import open3d as o3d
import numpy as np
import os
import json
import torch
from os.path import join, splitext
import argparse

def export_mesh(name, v, f, c=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_triangle_mesh(name, mesh)

def create_sphere_at_point(center, radius=0.01, color=[1, 0, 0]):
    """Create a sphere mesh at a given point with specified radius and color."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere

def save_scannet_superpoints_to_mesh(ply_path, 
                                     superpoints=None,
                                     use_ransac=False):
    """Save the superpoints segmentation results into mesh for visualization.
    Optionally, use RANSAC to get the centroid per unique index and color only the centroids.
    
    Args:
        ply_path: the path to the PLY file to process
        superpoints: list or array of superpoints indices
        use_ransac: boolean flag to use RANSAC for centroid computation
    """
    if superpoints is None:
        raise ValueError("Superpoints array must be provided")

    superpoints = np.array(superpoints)
    points_num = superpoints.shape[0]
    
    unique_superpoints = np.unique(superpoints)
    colors = np.ones((points_num, 3))  # Default color is white

    mesh = o3d.io.read_triangle_mesh(ply_path)
    v = np.array(mesh.vertices)
    f = np.array(mesh.triangles)
    
    centroids = []
    sphere_meshes = []

    if use_ransac:
        for sp in unique_superpoints:
            if sp == 0: continue
            sp_indices = np.where(superpoints == sp)[0]
            sp_points = v[sp_indices]
            if len(sp_points) < 3:
                # Not enough points to fit a plane
                continue
            
            # Perform RANSAC plane fitting
            sp_point_cloud = o3d.geometry.PointCloud()
            sp_point_cloud.points = o3d.utility.Vector3dVector(sp_points)
            plane_model, inliers = sp_point_cloud.segment_plane(distance_threshold=0.01,
                                                                ransac_n=3,
                                                                num_iterations=1000)
            inlier_points = sp_points[inliers]
            centroid = np.mean(inlier_points, axis=0)
            centroids.append(centroid)
            
            # Create a red sphere at the centroid
            sphere_mesh = create_sphere_at_point(centroid, radius=0.05, color=[1, 0, 0])  # Adjust radius as needed
            sphere_meshes.append(sphere_mesh)

    # Combine the original mesh and sphere meshes
    combined_mesh = mesh
    for sphere in sphere_meshes:
        combined_mesh += sphere

    # Save the combined mesh with spheres at centroids
    base_name, ext = splitext(ply_path)
    save_path = f'{base_name}_with_sp_centroids{ext}'
    o3d.io.write_triangle_mesh(save_path, combined_mesh)
    print(f'Saved combined mesh with centroids to {save_path}')

    if use_ransac:
        centroid_save_path = f'{base_name}_centroids.txt'
        np.savetxt(centroid_save_path, centroids, fmt='%.6f')
        print(f'Centroids saved to {centroid_save_path}')

def load_superpoints_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['segIndices']

def load_superpoints_pth(pth_path):
    data = torch.load(pth_path)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process ScanNet data and save it in evaluation format.')
    parser.add_argument('ply', type=str, help='The ply path of the scene to process')
    parser.add_argument('superpoints_path', type=str, help='The path to the file containing superpoints (.json or .pth)')
    parser.add_argument('--use_ransac', action='store_true', help='Use RANSAC to get the centroid per unique index')

    args = parser.parse_args()

    superpoints_path = args.superpoints_path
    if superpoints_path.endswith('.json'):
        superpoints = load_superpoints_json(superpoints_path)
    elif superpoints_path.endswith('.pth'):
        superpoints = load_superpoints_pth(superpoints_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .pth file.")

    save_scannet_superpoints_to_mesh(args.ply, superpoints=superpoints, use_ransac=args.use_ransac)
