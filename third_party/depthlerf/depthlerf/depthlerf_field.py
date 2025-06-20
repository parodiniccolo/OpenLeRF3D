from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import torch
import open3d as o3d
from depthlerf.depthlerf_fieldheadnames import DepthLERFFieldHeadNames
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from jaxtyping import Float
import sys

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field

try:
    import tinycudann as tcnn
except ImportError:
    pass
except EnvironmentError as _exp:
    if "Unknown compute capability" not in _exp.args[0]:
        raise _exp
    print("Could not load tinycudann: " + str(_exp), file=sys.stderr)


class DepthLERFField(Field):
    def __init__(
        self,
        grid_layers,
        grid_sizes,
        grid_resolutions,
        clip_n_dims: int,
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ):
        super().__init__()
        assert len(grid_layers) == len(grid_sizes) and len(grid_resolutions) == len(grid_layers)
        self.spatial_distortion = spatial_distortion
        self.clip_encs = torch.nn.ModuleList(
            [
                DepthLERFField._get_encoding(
                    grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i], indim=3, hash_size=grid_sizes[i]
                )
                for i in range(len(grid_layers))
            ]
        )
        tot_out_dims = sum([e.n_output_dims for e in self.clip_encs])

        self.clip_net = tcnn.Network(
            n_input_dims=tot_out_dims + 1,
            n_output_dims=clip_n_dims,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 4,
            },
        )

        self.dino_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=384,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            },
        )

    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

    def get_outputs_mesh_vertices(self, batch_size: int, mesh_vertices, clip_scales):
        """
            batch_size: int
            mesh_vertices: Tensor Size (1, batch_size, 3)
            clip_scales: Tensor Size (1, batch_size, 1)
        """
        outputs = {}

        positions = mesh_vertices
        # positions = positions.to(self.device)
        # positions = self.spatial_distortion(positions)
        # positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)) for e in self.clip_encs]
        x = torch.concat(xs, dim=-1)

        outputs[DepthLERFFieldHeadNames.HASHGRID] = x.view(*(1, batch_size), -1)

        clip_pass = self.clip_net(torch.cat([x, clip_scales.view(-1, 1)],
                                            dim=-1)).view(*(1, batch_size), -1)

        outputs[DepthLERFFieldHeadNames.CLIP] = clip_pass / clip_pass.norm(dim=-1, keepdim=True)

        dino_pass = self.dino_net(x).view(*(1, batch_size), -1)
        outputs[DepthLERFFieldHeadNames.DINO] = dino_pass

        return outputs

    def get_outputs(self, ray_samples: RaySamples, clip_scales):
        # random scales, one scale
        outputs = {}

        # 4096 24 3
        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)) for e in self.clip_encs]
        x = torch.concat(xs, dim=-1)

        outputs[DepthLERFFieldHeadNames.HASHGRID] = x.view(*ray_samples.frustums.shape, -1)

        clip_pass = self.clip_net(torch.cat([x, clip_scales.view(-1, 1)], dim=-1)).view(*ray_samples.frustums.shape, -1)
        outputs[DepthLERFFieldHeadNames.CLIP] = clip_pass / clip_pass.norm(dim=-1, keepdim=True)
        # 4096 24 512
        dino_pass = self.dino_net(x).view(*ray_samples.frustums.shape, -1)
        outputs[DepthLERFFieldHeadNames.DINO] = dino_pass

        return outputs

    def get_output_from_hashgrid(self, ray_samples: RaySamples, hashgrid_field, scale):
        # designated scales, run outputs for each scale
        hashgrid_field = hashgrid_field.view(-1, self.clip_net.n_input_dims - 1)
        clip_pass = self.clip_net(torch.cat([hashgrid_field, scale.view(-1, 1)], dim=-1)).view(
            *ray_samples.frustums.shape, -1
        )
        output = clip_pass / clip_pass.norm(dim=-1, keepdim=True)
        # 4096 24 512
        return output
