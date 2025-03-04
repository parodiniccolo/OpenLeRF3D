import open3d as o3d
import numpy as np
import os
import json
import torch
from os.path import join, splitext
import tqdm
import argparse

def export_mesh(name, v, f, c=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_triangle_mesh(name, mesh)

def save_scannet_superpoints_to_mesh(ply_path, 
                                     superpoints=None):
    """Save the superpoints segmentation results into mesh for visualization.
    We assign random colors to each superpoint. 
    
    Args:
        scene_id: the id of scene to visualize 
        superpoints: list or array of superpoints indices
    """
    if superpoints is None:
        raise ValueError("Superpoints array must be provided")

    superpoints = np.array(superpoints)
    points_num = superpoints.shape[0]
    
    unique_superpoints = np.unique(superpoints)
    colors = np.ones((points_num, 3))

    for sp in unique_superpoints:
        if sp == 0: continue
        colors[superpoints == sp] = np.random.rand(3)

    #if len(unique_superpoints) > 1:
    #    first_sp = unique_superpoints[1]  # Skip 0 (background)
    #   colors[superpoints == first_sp] = np.random.rand(3)  # Random color for the first superpoint
    
    mesh = o3d.io.read_triangle_mesh(ply_path)
    v = np.array(mesh.vertices)
    f = np.array(mesh.triangles)

    c_label = colors

    # Create save path with _superpts suffix
    base_name, ext = splitext(ply_path)
    save_path = f'{base_name}_superpts{ext}'
    export_mesh(save_path, v, f, c_label)
    print('save to', save_path)

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

    args = parser.parse_args()

    superpoints_path = args.superpoints_path
    if superpoints_path.endswith('.json'):
        superpoints = load_superpoints_json(superpoints_path)
    elif superpoints_path.endswith('.pth'):
        superpoints = load_superpoints_pth(superpoints_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .pth file.")

    save_scannet_superpoints_to_mesh(args.ply, superpoints=superpoints)
