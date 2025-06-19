"""
LERF implementation with efficient depth supervision integration.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
from torch.nn import Parameter
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.depth_nerfacto import DepthNerfactoModel, DepthNerfactoModelConfig
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap

from depthlerf.encoders.image_encoder import BaseImageEncoder
from depthlerf.depthlerf_field import DepthLERFField
from depthlerf.depthlerf_fieldheadnames import DepthLERFFieldHeadNames
from depthlerf.depthlerf_renderers import CLIPRenderer, MeanRenderer


@dataclass
class DepthLERFModelConfig(DepthNerfactoModelConfig):
    """Configuration for the LERF model"""
    
    _target: Type = field(default_factory=lambda: DepthLERFModel)
    clip_loss_weight: float = 0.1
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_lerf_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)


class DepthLERFModel(DepthNerfactoModel):
    """Memory-efficient LERF model with depth supervision"""
    
    config: DepthLERFModelConfig

    def populate_modules(self):
        """Set up the modules."""
        super().populate_modules()
        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()
        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        self.depthlerf_field = DepthLERFField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            clip_n_dims=self.image_encoder.embedding_dim,
        )


    def get_clip_embedding(self, ray_samples, weights, hashgrid_field, scales_shape):
        scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)
        
        clip_embedding_list = []
        for _, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.depthlerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())
            clip_embedding_list.append(clip_output)
        
        clip_embedding = torch.stack(clip_embedding_list, dim=1)
        
        return clip_embedding

    def get_max_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None):
        """Get maximum values across scales."""
        if preset_scales is not None:
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        n_phrases = len(self.image_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        
        for i, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.depthlerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())

            for j in range(n_phrases):
                if preset_scales is None or j == i:
                    probs = self.image_encoder.get_relevancy(clip_output, j)
                    pos_prob = probs[..., 0:1]
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_maxs[j] = scale
                        n_phrases_sims[j] = pos_prob

        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)

    def get_outputs(self, ray_bundle: RayBundle):
        """Memory efficient outputs computation"""
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        # Use the base nerfacto model to get depth supervision outputs
        base_outputs = super().get_outputs(ray_bundle)
        
        # Get the necessary components for LERF
        if self.training:
            weights = base_outputs['weights_list'][-1]
            ray_samples = base_outputs['ray_samples_list'][-1]
        else:
            # During inference, use proposal network
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
                ray_bundle, 
                density_fns=self.density_fns
            )
            field_outputs = self.field(ray_samples)
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            
            # Add to base_outputs for consistency
            if 'weights_list' not in base_outputs:
                base_outputs['weights_list'] = weights_list + [weights]
            if 'ray_samples_list' not in base_outputs:
                base_outputs['ray_samples_list'] = ray_samples_list + [ray_samples]

        # Process only necessary samples for LERF
        with torch.no_grad():
            lerf_weights, best_ids = torch.topk(weights, self.config.num_lerf_samples, dim=-2, sorted=False)

            def gather_fn(tens):
                return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

            dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
            lerf_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

            # Handle clip scales
            if self.training:
                clip_scales = ray_bundle.metadata["clip_scales"][..., None]
                dist = (lerf_samples.frustums.get_positions() - ray_bundle.origins[:, None, :]).norm(
                    dim=-1, keepdim=True
                )
                clip_scales = clip_scales * ray_bundle.metadata["height"] * (dist / ray_bundle.metadata["fy"])
            else:
                clip_scales = torch.ones_like(lerf_samples.spacing_starts, device=self.device)

        # Get LERF-specific outputs
        lerf_field_outputs = self.depthlerf_field.get_outputs(lerf_samples, clip_scales)
        base_outputs["clip"] = self.renderer_clip(
            embeds=lerf_field_outputs[DepthLERFFieldHeadNames.CLIP],
            weights=lerf_weights.detach()
        )
        base_outputs["dino"] = self.renderer_mean(
            embeds=lerf_field_outputs[DepthLERFFieldHeadNames.DINO],
            weights=lerf_weights.detach()
        )

        if not self.training:
            override_scales = ray_bundle.metadata.get("override_scales", None)
            with torch.no_grad():
                max_across, best_scales = self.get_max_across(
                    lerf_samples,
                    lerf_weights,
                    lerf_field_outputs[DepthLERFFieldHeadNames.HASHGRID],
                    clip_scales.shape,
                    preset_scales=override_scales,
                )
                base_outputs["raw_relevancy"] = max_across
                base_outputs["best_scales"] = best_scales.to(self.device)


        if not self.training:
            with torch.no_grad():
                clip_embedding = self.get_clip_embedding(
                    lerf_samples,
                    lerf_weights,
                    lerf_field_outputs[DepthLERFFieldHeadNames.HASHGRID],
                    clip_scales.shape
                )

            return base_outputs, clip_embedding
        return base_outputs



    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Process camera ray bundle outputs."""
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)

        # First pass to find best scales
        best_scales = None
        best_relevancies = None

        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs, _ = self.forward(ray_bundle=ray_bundle)

            if "best_scales" in outputs and "raw_relevancy" in outputs:
                if best_scales is None:
                    best_scales = outputs["best_scales"]
                    best_relevancies = [m.max() for m in outputs["raw_relevancy"]]
                else:
                    for phrase_i in range(outputs["best_scales"].shape[0]):
                        m = outputs["raw_relevancy"][phrase_i, ...].max()
                        if m > best_relevancies[phrase_i]:
                            best_scales[phrase_i] = outputs["best_scales"][phrase_i]
                            best_relevancies[phrase_i] = m

        # Second pass with best scales
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            if best_scales is not None:
                ray_bundle.metadata["override_scales"] = best_scales
            outputs, _ = self.forward(ray_bundle=ray_bundle)
            
            for output_name, output in outputs.items():
                if output_name == "best_scales":
                    continue
                if output_name == "raw_relevancy" and output.shape[0] > 0:
                    for r_id in range(output.shape[0]):
                        outputs_lists[f"relevancy_{r_id}"].append(output[r_id, ...])
                else:
                    outputs_lists[output_name].append(output)

        # Combine outputs
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not outputs_list or not torch.is_tensor(outputs_list[0]):
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)

        # Process relevancy outputs if available
        if best_scales is not None:
            for i in range(len(self.image_encoder.positives)):
                if f"relevancy_{i}" in outputs:
                    p_i = torch.clip(outputs[f"relevancy_{i}"] - 0.5, 0, 1)
                    outputs[f"composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo"))
                    if "rgb" in outputs and "relevancy_0" in outputs:
                        mask = (outputs["relevancy_0"] < 0.5).squeeze()
                        outputs[f"composited_{i}"][mask, :] = outputs["rgb"][mask, :]

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Combined loss computation."""
        # Get depth-related losses from parent
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            # Add LERF-specific losses
            unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["clip"], batch["clip"], delta=1.25, reduction="none"
            )
            loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()

            unreduced_dino = torch.nn.functional.mse_loss(
                outputs["dino"], batch["dino"], reduction="none"
            )
            loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()

        return loss_dict

    def get_metrics_dict(self, outputs, batch):
        """Combined metrics computation."""
        metrics_dict = super().get_metrics_dict(outputs, batch)
        return metrics_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get parameter groups for optimization."""
        param_groups = super().get_param_groups()
        param_groups["depthlerf"] = list(self.depthlerf_field.parameters())
        return param_groups