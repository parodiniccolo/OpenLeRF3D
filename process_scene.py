"""
Scene analyzer module for analyzing NeRF scenes and computing relevancy scores.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import sys
from rich.console import Console
import matplotlib.pyplot as plt

import h5py
import numpy as np
import torch
import open_clip
import open3d as o3d
from attrs import define
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from sklearn.cluster import DBSCAN
from torch import Tensor
from scipy.spatial import cKDTree


CONSOLE = Console()

def read_ply(ply_path: str) -> np.ndarray:
    """Read vertices from a PLY file."""
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)

def read_instance_mask(mask_array, vertices):
    mask = []
    masked_vertices = []
    for index, vertex in zip(mask_array, vertices):
        mask.append(index)
        masked_vertices.append(np.append(vertex, index))  # Append original coordinates with index
    return mask, np.array(masked_vertices)

def apply_mask_colors(original_colors, masks, mask_scores, colormap, opacity=0.75):
    """Apply colored masks to the point cloud based on mask scores."""
    colors = original_colors.copy()
    
    # Normalize scores for coloring
    if mask_scores:
        max_score = max(mask_scores.values())
        normalized_scores = {k: v/max_score for k, v in mask_scores.items()}
        
        # Apply colors with opacity
        for mask_idx, score in normalized_scores.items():
            mask_array = np.array(masks[mask_idx], dtype=bool)
            overlay_color = np.array(colormap(score)[:3])
            original_color = colors[mask_array]
            
            # Blend colors with intensity-based opacity
            intensity_opacity = opacity * score
            blended_color = (1 - intensity_opacity) * original_color + intensity_opacity * overlay_color
            colors[mask_array] = blended_color
            
    return colors


class ShapeDescriptor:
    """Computes and stores shape descriptors for point clouds."""
    
    @staticmethod
    def compute(points: np.ndarray) -> Optional[Dict]:
        """
        Compute shape descriptors for a set of points.
        
        Args:
            points: Nx3 array of 3D points
            
        Returns:
            Dictionary containing shape descriptors or None if computation fails
        """
        if len(points) < 4:
            CONSOLE.print(f"[yellow]Warning: Not enough points for shape descriptors: {len(points)} points[/]")
            return None
            
        try:
            centroid = np.mean(points, axis=0)
            centered_points = points - centroid
            cov_matrix = np.cov(centered_points.T)
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            total_variance = np.sum(eigenvalues)
            if total_variance == 0:
                CONSOLE.print("[yellow]Warning: Zero total variance in shape descriptors[/]")
                return None
                
            normalized_eigenvalues = eigenvalues / total_variance
            
            return {
                'eigenvalues': normalized_eigenvalues,
                'eigenvectors': eigenvectors,
                'planarity': (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if eigenvalues[0] != 0 else 0,
                'linearity': (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if eigenvalues[0] != 0 else 0,
                'sphericity': eigenvalues[2] / eigenvalues[0] if eigenvalues[0] != 0 else 0,
                'centroid': centroid
            }
        except np.linalg.LinAlgError as e:
            CONSOLE.print(f"[yellow]Warning: Linear algebra error in shape descriptors: {e}[/]")
            return None

class MaskSelector:
    """Handles selection of masks based on point scores."""
    
    @staticmethod
    def find_masks_for_high_score_points(
        points: np.ndarray,
        possibility_array: np.ndarray,
        masked_vertices_list: List[np.ndarray],
        threshold: float = 0.55
    ) -> Tuple[Dict, List]:
        """
        Select masks based on points with high possibility scores and shape descriptors.
        
        Args:
            points: Nx3 array of point coordinates
            possibility_array: N-length array of scores
            masked_vertices_list: List of masked vertex arrays
            threshold: Score threshold for considering points
            
        Returns:
            Tuple of (best_masks_dict, positive_masks_list)
        """
        CONSOLE.print("\n[blue]Starting mask selection for high-score points:[/]")
        CONSOLE.print(f"Number of total points: {len(points)}")
        CONSOLE.print(f"Number of masks: {len(masked_vertices_list)}")
        
        high_score_indices = np.where(possibility_array >= threshold)[0]
        high_score_points = points[high_score_indices]
        high_scores = possibility_array[high_score_indices]
        
        if len(high_score_points) == 0:
            return {}, []
                    
        k = min(3, len(high_score_points))
        if k == 0:
            return {}, []
       
        # Compute shape descriptors for high-score points
        high_score_descriptors = ShapeDescriptor.compute(high_score_points)
        
        points_tree = cKDTree(high_score_points)
        mask_scores = {}
        
        for mask_idx, mask_points in enumerate(masked_vertices_list):
            mask_points = np.asarray(mask_points)
            if mask_points.shape[1] > 3:
                mask_points = mask_points[mask_points[:, 3] == 1][:, :3]
                    
            if len(mask_points) == 0:
                continue
                    
            try:
                distances, indices = points_tree.query(mask_points, k=k)
                if k == 1:
                    indices = indices.reshape(-1, 1)
                    distances = distances.reshape(-1, 1)
                        
                mask_point_scores = high_scores[indices]
                valid_distances = distances < 0.2
                
                if np.sum(valid_distances) == 0:
                    continue
                
                avg_score = np.mean(mask_point_scores[valid_distances] if k > 1 else mask_point_scores.flatten())
                coverage_ratio = np.sum(valid_distances) / len(mask_points)
                
                # Calculate mask score with shape descriptor information if available
                mask_score = avg_score * 0.75 + coverage_ratio * 0.25  # Default score calculation
                
                # Compute shape descriptors for mask points and use if available
                mask_descriptors = ShapeDescriptor.compute(mask_points)
                
                if mask_descriptors is not None and high_score_descriptors is not None:
                    # Calculate shape similarity based on eigenvalues
                    shape_similarity = 1.0 - np.mean(np.abs(
                        mask_descriptors['eigenvalues'] - high_score_descriptors['eigenvalues']
                    ))
                    
                    # Incorporate geometric properties into mask score
                    mask_score = (avg_score * 0.7) + (coverage_ratio * 0.15) + (shape_similarity * 0.15)
                    

                if mask_score > 0.60:
                    mask_scores[mask_idx] = mask_score

            except Exception as e:
                CONSOLE.print(f"[red]Error processing mask {mask_idx}: {str(e)}[/]")
                continue
        

        positive_masks = list(mask_scores.keys())
        if not positive_masks:
            return {}, []
                
        best_mask_idx = max(mask_scores.items(), key=lambda x: x[1])[0]
        return {
            0: {
                'best_mask_index': best_mask_idx,
                'close_mask_indices': positive_masks,
                'similarities': [mask_scores[idx] for idx in positive_masks],
                'point_scores': [mask_scores[idx] for idx in positive_masks]
            }
        }, positive_masks



@define
class SceneAnalyzer:
    """Handles analyzing NeRF scenes."""
    
    scene_name: str
    lerf_pipeline: Pipeline
    h5_dict: dict
    clip_model: Optional[torch.nn.Module]
    tokenizer: Optional[object]
    neg_embeds: Tensor
    negative_words_length: int
    axis_align_matrix: Optional[np.ndarray]

    def analyze_scene(self, query, config: Dict):
        """Perform scene analysis and evaluation."""
        scene_pcd = o3d.io.read_point_cloud(config['ply_path'])

        # Define rotation angles and create rotation matrix
        angles = (np.pi / 2, np.pi / 2, 0)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(angles)

        # Apply rotation to the point cloud
        #scene_pcd.rotate(rotation_matrix)
        original_colors = np.asarray(scene_pcd.colors)
        masks = np.asarray(torch.load(config['path_pred_masks'])).T
        ply_vertices = read_ply(config['ply_path'])
        
        # Apply rotation to the ply vertices (same rotation as applied to the point cloud)
        #ply_vertices = np.dot(rotation_matrix, ply_vertices.T).T


        instance_masks = []
        masked_vertices_list = []
        vertex_mask_indices = [-1] * len(ply_vertices)
        
        # Process masks
        for mask_index, mask in enumerate(masks):
            instance_mask, masked_vertices = read_instance_mask(mask, ply_vertices)
            instance_masks.append(instance_mask)
            masked_vertices_list.append(masked_vertices)
            
            for i, is_masked in enumerate(instance_mask):
                if is_masked == 1.0:
                    vertex_mask_indices[i] = mask_index
        
        #print("masked v list: ",masked_vertices_list)
        #print("masks: ", masks)

        # Process query and get results
        best_masks, positive_masks = self.process_query(query, masked_vertices_list, masks)
        
        if not best_masks or not positive_masks:
            print("best_masks", best_masks)
            print("pos masks; ", positive_masks)
            CONSOLE.print("[yellow]No matching masks found for the query.[/]")
            return
        
    

        # Create colormap and apply colors
        colormap = plt.get_cmap('turbo')
        
        # Get mask scores from best_masks
        mask_scores = {mask_idx: score for mask_idx, score in 
                      zip(best_masks[0]['close_mask_indices'], 
                          best_masks[0]['similarities'])} if best_masks else {}
        
        colors = apply_mask_colors(
            original_colors=original_colors,
            masks=masks,
            mask_scores=mask_scores,
            colormap=colormap,
            opacity=0.75
        )

        # Sanitize names for filesystem
        sanitized_scene_name = self.scene_name.replace(" ", "_")
        sanitized_query = query.replace(" ", "_")
        
        # Export results
        os.makedirs(f"results/{self.scene_name}/{self.scene_name}_close_mask_indices/", exist_ok=True)
        with open(f"results/{self.scene_name}/{self.scene_name}_close_mask_indices/{sanitized_query}.txt", 'w') as f:
            for mask_idx in positive_masks:
                f.write(f"{sanitized_query},{mask_idx}\n")

        # Create and save point cloud with masks
        pcd_with_masks = o3d.geometry.PointCloud()
        pcd_with_masks.points = o3d.utility.Vector3dVector(ply_vertices)
        pcd_with_masks.colors = o3d.utility.Vector3dVector(colors)

        # Save results
        os.makedirs(f"results/{self.scene_name}/{self.scene_name}_masked_pcd/", exist_ok=True)
        output_file_name = f"results/{self.scene_name}/{self.scene_name}_masked_pcd/{sanitized_scene_name}_{sanitized_query}_pointcloud_with_masks.ply"
        o3d.io.write_point_cloud(output_file_name, pcd_with_masks)
        
        CONSOLE.print(f"[green]Results saved to {output_file_name}[/]")
        
        # Generate heatmap visualization if needed
        centroids, scores, _, points, possibility_array, _, _ = self.find_centroids_bbox(query)
        if points is not None and possibility_array is not None and len(possibility_array) > 0:
            heatmap_colors = np.zeros((len(possibility_array), 4))
            threshold = 0.55

            # Fill colors based on possibility array values
            for i, value in enumerate(possibility_array):
                if value > threshold:
                    heatmap_colors[i] = colormap(value)
                else:
                    heatmap_colors[i] = [1.0, 1.0, 1.0, 1.0]

            # Process colors
            heatmap_colors = heatmap_colors.reshape(-1, 4)[:, :3]

            # Create point cloud with heatmap
            transformed_points = np.ascontiguousarray(points, dtype=np.float64)
            pcd_with_heatmap = o3d.geometry.PointCloud()
            pcd_with_heatmap.points = o3d.utility.Vector3dVector(transformed_points)
            pcd_with_heatmap.colors = o3d.utility.Vector3dVector(heatmap_colors)

            # Save heatmap visualization
            heatmap_output = f"results/{self.scene_name}/{self.scene_name}_masked_pcd/{sanitized_scene_name}_{sanitized_query}_nerf_pointcloud_heatmap.ply"
            o3d.io.write_point_cloud(heatmap_output, pcd_with_heatmap)
            CONSOLE.print(f"[green]Heatmap saved to {heatmap_output}[/]")

            # Create filted point cloud based on Threshold
            filtered = ~np.all(heatmap_colors == [1.0, 1.0, 1.0], axis=1)
            filtered_points = transformed_points[filtered]
            filtered_colors = heatmap_colors[filtered]
            pcd_with_heatmap_filtered = o3d.geometry.PointCloud()
            pcd_with_heatmap_filtered.points = o3d.utility.Vector3dVector(filtered_points)
            pcd_with_heatmap_filtered.colors = o3d.utility.Vector3dVector(filtered_colors)

            # Save filtered heatmap
            filtered_heatmap_output = f"results/{self.scene_name}/{self.scene_name}_masked_pcd/{sanitized_scene_name}_{sanitized_query}_nerf_pointcloud_heatmap_filtered.ply"
            o3d.io.write_point_cloud(filtered_heatmap_output, pcd_with_heatmap_filtered)
            CONSOLE.print(f"[green]Filtered heatmap saved to {filtered_heatmap_output}[/]")



    def process_query(self, query: str, masked_vertices_list: List, masks: np.ndarray):
        """Process a single query and return results."""
        centroids, scores, _, points, possibility_array, _, _ = self.find_centroids_bbox(query)
        
        
        if points is None or possibility_array is None or len(possibility_array) == 0:
            return None, None
            
        high_score_points = np.sum(possibility_array >= 0.55)
        if high_score_points < 3:
            return None, None
            
        return MaskSelector.find_masks_for_high_score_points(
            points=points,
            possibility_array=possibility_array,
            masked_vertices_list=masked_vertices_list,
            threshold=0.55
        )



    def get_relevancy(
        self,
        embed: torch.Tensor,
        positive_id: int,
        pos_embeds: Tensor,
        neg_embeds: Tensor,
        positive_words_length: int,
    ) -> torch.Tensor:
        """Computes relevancy scores for embeddings."""
        phrases_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        positive_vals = output[..., positive_id:positive_id + 1]
        negative_vals = output[..., positive_words_length:]
        repeated_pos = positive_vals.repeat(1, self.negative_words_length)

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        
        return torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(best_id.shape[0], self.negative_words_length, 2),
        )[:, 0, :]

    def find_centroids_bbox(
        self, 
        query: str,
    ) -> Tuple[List, List, List, np.ndarray, np.ndarray, List, List]:
        """Finds centroids and bounding boxes for a query."""
        positives = [query]
        
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in positives]).to("cuda")
            pos_embeds = self.clip_model.encode_text(tok_phrases)
            pos_embeds /= pos_embeds.norm(dim=-1, keepdim=True)

        scales_list = torch.linspace(0.0, 1.5, 30)
        best_scale = None
        probability_per_scale = None

        # Find best scale and probability
        for i, scale in enumerate(scales_list):
            clip_output = torch.from_numpy(self.h5_dict["clip_embeddings_per_scale"][i]).to("cuda")
            probs = self.get_relevancy(
                embed=clip_output,
                positive_id=0,
                pos_embeds=pos_embeds,
                neg_embeds=self.neg_embeds,
                positive_words_length=1,
            )
            pos_prob = probs[..., 0:1]

            if best_scale is None or pos_prob.max() > probability_per_scale.max():
                best_scale = scale
                probability_per_scale = pos_prob

        possibility_array = probability_per_scale.detach().cpu().numpy()
        top_indices = np.nonzero(possibility_array > 0.55)[0]

        if top_indices.shape[0] == 0:
            CONSOLE.print("[yellow]No points found for clustering.[/]")
            return [], [], [], [], [], [], []

        points = self.h5_dict["points"]
        origins = self.h5_dict["origins"]
        top_positions = points[top_indices]
        top_origins = origins[top_indices]
        top_values = possibility_array[top_indices].flatten()

        # Cluster points
        dbscan = DBSCAN(eps=0.05, min_samples=15)
        clusters = dbscan.fit(top_positions)
        labels = clusters.labels_

        centroids = []
        scores = []
        bboxes = []
        best_member_list = []
        origin_for_best_member_list = []
        values = []
        clusters_members = []

        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise
                continue

            members = top_positions[labels == cluster_id]
            centroid = np.mean(members, axis=0)
            centroids.append(centroid)

            # Calculate bounding box
            sx = np.max(members[:, 0]) - np.min(members[:, 0])
            sy = np.max(members[:, 1]) - np.min(members[:, 1])
            sz = np.max(members[:, 2]) - np.min(members[:, 2])
            bboxes.append((sx, sy, sz))

            # Get cluster values and members
            cluster_values = top_values[labels == cluster_id]
            cluster_origins = top_origins[labels == cluster_id]
            values.append(cluster_values)
            clusters_members.append(members)

            # Find best member
            best_index = np.argmax(cluster_values)
            scores.append(cluster_values[best_index])
            best_member_list.append(members[best_index])
            origin_for_best_member_list.append(cluster_origins[best_index])

        return centroids, scores, bboxes, points, possibility_array, values, clusters_members

class SceneAnalyzerFactory:
    """Factory class for creating SceneAnalyzer instances."""
    
    @staticmethod
    def initialize_lerf_pipeline(load_config: str, scene_name: str) -> Pipeline:
        """Initializes LERF pipeline."""
        initial_dir = os.getcwd()
        os.chdir(os.path.join("outputs", scene_name))
        _, lerf_pipeline, _, _ = eval_setup(
            Path(load_config),
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )
        os.chdir(initial_dir)
        return lerf_pipeline

    @staticmethod
    def load_h5_file(load_config: str) -> Dict:
        """Loads H5 file containing points and embeddings."""
        hdf5_file = h5py.File(load_config, "r")
        
        points = hdf5_file["points"]["points"][:]
        origins = hdf5_file["origins"]["origins"][:]
        directions = hdf5_file["directions"]["directions"][:]
        rgb = hdf5_file["rgb"]["rgb"][:]

        clip_embeddings_per_scale = []
        clips_group = hdf5_file["clip"]
        for i in range(30):
            clip_embeddings_per_scale.append(clips_group[f"scale_{i}"][:])

        hdf5_file.close()
        
        return {
            "points": points,
            "origins": origins,
            "directions": directions,
            "clip_embeddings_per_scale": clip_embeddings_per_scale,
            "rgb": rgb,
        }

def main():
    """
    Main entry point for the NeRF scene analysis tool.
    
    Usage:
        python script.py <scene_name> <query> <nerf_exported_mesh_path> <h5_file_path> <ply_path> 
                        <config_path> <dataparser_transforms> <path_pred_masks>
    """
    # Validate command line arguments
    if len(sys.argv) != 9:
        CONSOLE.print("[red]Error: Incorrect number of arguments[/]")
        CONSOLE.print("Usage: {} <scene_name> <query> <nerf_exported_mesh_path> <h5_file_path> <ply_path> "
              "<config_path> <dataparser_transforms> <path_pred_masks>".format(sys.argv[0]))
        sys.exit(1)
        
    # Extract command line arguments
    scene_name = sys.argv[1]
    query = sys.argv[2]
    config = {
        'nerf_exported_mesh_path': sys.argv[3],
        'h5_file_path': sys.argv[4],
        'ply_path': sys.argv[5],
        'config_path': sys.argv[6],
        'dataparser_transforms': sys.argv[7],
        'path_pred_masks': sys.argv[8]
    }
    
    try:
        # Initialize clip model
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            precision="fp16"
        )
        model.eval()
        model = model.to("cuda")
        tokenizer = open_clip.get_tokenizer("ViT-L-14")

        # Initialize negative embeddings
        negatives = ["object", "things", "stuff", "texture"]
        with torch.no_grad():
            tok_phrases = torch.cat([tokenizer(phrase) for phrase in negatives]).to("cuda")
            neg_embeds = model.encode_text(tok_phrases)
            neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)

        # Setup components
        lerf_pipeline = SceneAnalyzerFactory.initialize_lerf_pipeline(config['config_path'], scene_name)
        h5_dict = SceneAnalyzerFactory.load_h5_file(config['h5_file_path'])
        
        # Create analyzer
        analyzer = SceneAnalyzer(
            scene_name=scene_name,
            lerf_pipeline=lerf_pipeline,
            h5_dict=h5_dict,
            clip_model=model,
            tokenizer=tokenizer,
            neg_embeds=neg_embeds,
            negative_words_length=4,
            axis_align_matrix=None
        )
        
        # Run the scene analysis
        CONSOLE.print("[blue]Starting scene analysis...[/]")
        analyzer.analyze_scene(
            query=query,
            config=config
        )
        
    except Exception as e:
        CONSOLE.print(f"[red]Error during scene analysis: {str(e)}[/]")
        sys.exit(1)

if __name__ == "__main__":
    main()