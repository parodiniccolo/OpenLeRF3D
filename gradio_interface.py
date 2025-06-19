import os
import sys
import time
import torch
import numpy as np
import open3d as o3d
import h5py
import open_clip
import gradio as gr
import tempfile
import trimesh
import copy
import json
import subprocess
import shutil
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.neighbors import NearestNeighbors

# Import necessary components from your existing code
from process_scene import (
    SceneAnalyzer, SceneAnalyzerFactory, ShapeDescriptor,
    read_ply
)

# Initialize Rich console
CONSOLE = Console()

# Global variables to store loaded models and data
scene_analyzer = None
scene_pcd = None
ply_vertices = None
original_colors = None
scene_config = {}
scene_mesh = None  # Store the loaded mesh
last_result_pcd = None  # Store the last generated point cloud for download

def densify_point_cloud(points, colors, density_factor=3.0, noise_std=0.01):
    """
    Create a denser point cloud by adding interpolated points around existing ones.
    
    Args:
        points: Original points array
        colors: Original colors array  
        density_factor: How many additional points to add per original point
        noise_std: Standard deviation of noise to add for variation
        
    Returns:
        Tuple of (dense_points, dense_colors)
    """
    CONSOLE.print(f"[blue]Densifying point cloud with factor {density_factor}...[/]")
    
    # Calculate number of new points to add
    num_original = len(points)
    num_new = int(num_original * density_factor)
    
    # Create new points by adding small random offsets to existing points
    # Randomly select points to duplicate
    indices = np.random.choice(num_original, num_new, replace=True)
    
    # Add small random noise around selected points
    noise = np.random.normal(0, noise_std, (num_new, 3))
    new_points = points[indices] + noise
    new_colors = colors[indices]
    
    # Combine original and new points
    dense_points = np.vstack([points, new_points])
    dense_colors = np.vstack([colors, new_colors])
    
    CONSOLE.print(f"[green]✓ Densified from {num_original} to {len(dense_points)} points[/]")
    return dense_points, dense_colors

def create_sphere_representation(points, colors, sphere_radius=0.02, sphere_resolution=8):
    """
    Create a sphere-based representation instead of just points for better visualization.
    
    Args:
        points: Points array
        colors: Colors array
        sphere_radius: Radius of each sphere
        sphere_resolution: Resolution of sphere mesh
        
    Returns:
        Combined trimesh scene
    """
    CONSOLE.print(f"[blue]Creating sphere representation with {len(points)} spheres...[/]")
    
    # Create a basic sphere mesh
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius)
    
    # Create a scene to hold all spheres
    scene = trimesh.Scene()
    
    # Add spheres at each point location with corresponding colors
    for i, (point, color) in enumerate(zip(points, colors)):
        # Create a copy of the sphere
        sphere_copy = sphere.copy()
        
        # Translate to the point location
        sphere_copy.apply_translation(point)
        
        # Set the color
        sphere_copy.visual.face_colors = np.tile(color, (len(sphere_copy.faces), 1))
        
        # Add to scene
        scene.add_geometry(sphere_copy, node_name=f"sphere_{i}")
        
        # Limit number of spheres to prevent memory issues
        if i >= 5000:  # Limit to 5000 spheres max
            CONSOLE.print(f"[yellow]Limited to {i+1} spheres to prevent memory issues[/]")
            break
    
    CONSOLE.print(f"[green]✓ Created sphere representation with {len(scene.geometry)} spheres[/]")
    return scene

def save_point_cloud_as_ply(points, colors, filename):
    """
    Save point cloud as PLY file for download.
    
    Args:
        points: Points array
        colors: Colors array  
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    try:
        CONSOLE.print(f"[blue]Saving point cloud as PLY: {filename}[/]")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save as PLY
        o3d.io.write_point_cloud(filename, pcd)
        
        CONSOLE.print(f"[green]✓ Saved point cloud to {filename}[/]")
        return filename
        
    except Exception as e:
        CONSOLE.print(f"[red]Error saving PLY file: {str(e)}[/]")
        return None

def check_and_run_training(scene_name, data_path, work_dir, lerf_type="lerf-big",
                          progress=gr.Progress()):
    """
    Check if necessary files for the scene exist, and run training if they don't.
    Returns three elements: main status, detailed logs, and point cloud path.
    """
    
    # Initialize status messages
    main_status = "Checking required files..."
    logs_html = "Starting training check...\n"
    point_cloud_path = None
    
    # Display in terminal
    CONSOLE.print(Panel(f"[bold cyan]Training Check:[/] {scene_name}", border_style="blue"))
    
    # Basic path checks
    if not os.path.exists(data_path):
        main_status = "Error: Data path not found"
        CONSOLE.print(f"[red]Error: Data path not found at {data_path}[/]")
        return main_status, logs_html, point_cloud_path
    
    # NeRF paths
    config_path = f"{work_dir}/outputs/{scene_name}/{lerf_type}/{scene_name}/config.yml"
    checkpoint_path = f"{work_dir}/outputs/{scene_name}/{lerf_type}/{scene_name}/nerfstudio_models/step-000029999.ckpt"
    nerf_pointcloud_path = f"{data_path}/point_cloud.ply"
    h5_path = f"{data_path}/embeddings_v2.h5"
    
    # Dictionary to track what needs to be done
    tasks = {
        "train_lerf": not os.path.exists(config_path) or not os.path.exists(checkpoint_path),
        "export_nerf": not os.path.exists(nerf_pointcloud_path) or not os.path.exists(h5_path)
    }
    
    # Display task status in terminal
    CONSOLE.print("[blue]Task Status:[/]")
    for task, needed in tasks.items():
        status_text = "[yellow]Needs Processing[/]" if needed else "[green]Already Complete[/]"
        CONSOLE.print(f"- {task.replace('_', ' ').title()}: {status_text}")
    
    # Check if everything is already done
    if not any(tasks.values()):
        main_status = "All necessary files already exist."
        CONSOLE.print("[green]All files exist. Ready to initialize the scene.[/]")
        return main_status, logs_html, point_cloud_path
    
    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Update UI with current status
    yield main_status, logs_html, point_cloud_path
    
    # Train LERF if needed
    if tasks["train_lerf"]:
        main_status = "Processing: Train LERF..."
        CONSOLE.print(f"\n[bold blue]Training {lerf_type}...[/]")
        CONSOLE.print(f"[blue]Will check for checkpoint: {checkpoint_path}[/]")
        logs_html += f"\nTraining {lerf_type}...\n"
        logs_html += f"Will check for checkpoint: {checkpoint_path}\n"
        yield main_status, logs_html, point_cloud_path
        
        try:
            cmd = [
                "ns-train", lerf_type,
                "--data", data_path,
                "--experiment-name", scene_name,
                "--timestamp", scene_name,
                "--output-dir", f"{work_dir}/outputs/",
                "--viewer.quit-on-train-completion", "True",
            ]
            
            CONSOLE.print(f"[blue]Running command: {' '.join(cmd)}[/]")
            logs_html += f"Running command: {' '.join(cmd)}\n"
            yield main_status, logs_html, point_cloud_path
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Process and display output in real-time to terminal
            for line in process.stdout:
                CONSOLE.print(line.strip())
                logs_html += line
                yield main_status, logs_html, point_cloud_path
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode != 0:
                # If there was an error, capture and display stderr
                error = process.stderr.read()
                CONSOLE.print(f"[red]Error in LERF training:[/]\n{error}")
                main_status = "Error in LERF training"
                logs_html += f"Error in LERF training:\n{error}\n"
                yield main_status, logs_html, point_cloud_path
                return main_status, logs_html, point_cloud_path
            
            # Verify checkpoint exists
            if os.path.exists(checkpoint_path):
                CONSOLE.print(f"[green]LERF training completed. Checkpoint found.[/]")
                logs_html += f"LERF training completed. Checkpoint found.\n"
            else:
                CONSOLE.print(f"[yellow]Warning: LERF training completed but checkpoint not found at expected location.[/]")
                logs_html += f"Warning: LERF training completed but checkpoint not found at expected location.\n"
            
        except subprocess.CalledProcessError as e:
            main_status = "Error in LERF training"
            CONSOLE.print(f"[red]Error in LERF training: {str(e)}[/]")
            logs_html += f"Error in LERF training: {str(e)}"
            return main_status, logs_html, point_cloud_path

    # Export NeRF if needed
    if tasks["export_nerf"]:
        main_status = "Processing: Export NeRF..."
        CONSOLE.print("\n[bold blue]Exporting NeRF pointcloud and CLIP embeddings...[/]")
        logs_html += "\nExporting NeRF pointcloud and CLIP embeddings...\n"
        yield main_status, logs_html, point_cloud_path
        
        try:
            cmd = [
                "ns-export", "pointcloud",
                "--num-points", "70000",
                "--remove-outliers", "False",
                "--normal-method", "open3d",
                "--reorient-normals", "False",
                "--save-world-frame", "False",
                "--load-config", config_path,
                "--output-dir", data_path,
                "--obb-center", "0.0000000000", "0.0000000000", "0.0000000000",
                "--obb-rotation", "0.0000000000", "0.0000000000", "0.0000000000",
                "--obb-scale", "5.0000000000", "5.0000000000", "5.0000000000",
            ]
            
            CONSOLE.print(f"[blue]Running command: {' '.join(cmd)}[/]")
            logs_html += f"Running command: {' '.join(cmd)}\n"
            yield main_status, logs_html, point_cloud_path
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Process and display output in real-time to terminal
            for line in process.stdout:
                CONSOLE.print(line.strip())
                logs_html += line
                yield main_status, logs_html, point_cloud_path
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode != 0:
                # If there was an error, capture and display stderr
                error = process.stderr.read()
                CONSOLE.print(f"[red]Error in NeRF export:[/]\n{error}")
                main_status = "Error in NeRF export"
                logs_html += f"Error in NeRF export:\n{error}\n"
                yield main_status, logs_html, point_cloud_path
                return main_status, logs_html, point_cloud_path
            
            CONSOLE.print(f"[green]NeRF export completed.[/]")
            logs_html += f"NeRF export completed.\n"
            
        except subprocess.CalledProcessError as e:
            main_status = "Error in NeRF export"
            CONSOLE.print(f"[red]Error in NeRF export: {str(e)}[/]")
            logs_html += f"Error in NeRF export: {str(e)}"
            return main_status, logs_html, point_cloud_path
    
    main_status = "Training complete!"
    CONSOLE.print("\n[bold green]All training tasks completed successfully![/] You can now initialize the scene for queries.")
    logs_html += "\nAll training tasks completed successfully! You can now initialize the scene for queries."
    
    yield main_status, logs_html, point_cloud_path
    return main_status, logs_html, point_cloud_path


def initialize_scene(scene_name, data_path, work_dir, lerf_type="lerf-big"):
    """Initialize the scene analyzer and load the scene point cloud."""
    global scene_analyzer, scene_pcd, ply_vertices, original_colors, scene_config, scene_mesh
    
    try:
        CONSOLE.print(Panel(f"[bold cyan]Loading Scene:[/] {scene_name}", border_style="blue"))
        
        # Generate paths based on the original run_query.py structure
        output_dir = data_path
        
        # Generate paths for checking if training has been completed
        config_path = f"{work_dir}/outputs/{scene_name}/{lerf_type}/{scene_name}/config.yml"
        checkpoint_path = f"{work_dir}/outputs/{scene_name}/{lerf_type}/{scene_name}/nerfstudio_models/step-000029999.ckpt"
        nerf_pointcloud_path = f"{output_dir}/point_cloud.ply"
        h5_path = f"{output_dir}/embeddings_v2.h5"
        
        # Check if essential files exist
        files_exist = (
            os.path.exists(output_dir) and
            os.path.exists(config_path) and
            os.path.exists(checkpoint_path) and
            os.path.exists(nerf_pointcloud_path) and
            os.path.exists(h5_path)
        )
        
        if not files_exist:
            missing_files = []
            if not os.path.exists(output_dir): missing_files.append("output_dir")
            if not os.path.exists(config_path): missing_files.append("config_path")
            if not os.path.exists(checkpoint_path): missing_files.append("checkpoint_path")
            if not os.path.exists(nerf_pointcloud_path): missing_files.append("nerf_pointcloud_path")
            if not os.path.exists(h5_path): missing_files.append("h5_path")
            
            CONSOLE.print(f"[yellow]Warning: Some required files are missing: {', '.join(missing_files)}[/]")
            return None, "Scene not fully trained. Please run the training pipeline first."
        
        # Use the exported point cloud
        ply_path = nerf_pointcloud_path
        CONSOLE.print(f"[green]Using exported NeRF point cloud: {ply_path}[/]")
        
        scene_config = {
            'scene_name': scene_name,
            'lerf_type': lerf_type,
            'data_path': data_path,
            'output_dir': output_dir,
            'ply_path': ply_path
        }
        
        # Create a paths table for display
        path_table = Table(title="File Paths", show_header=True, header_style="bold magenta")
        path_table.add_column("Path Type", style="cyan")
        path_table.add_column("Location", style="yellow", width=60)
        
        for key, value in scene_config.items():
            if isinstance(value, str) and os.path.exists(value):
                path_table.add_row(key, value)
        
        CONSOLE.print(path_table)
        
        # Load the point cloud from the PLY file
        try:
            CONSOLE.print(f"[blue]Loading point cloud from: {ply_path}[/]")
            scene_pcd = o3d.io.read_point_cloud(ply_path)
            if scene_pcd.is_empty():
                CONSOLE.print(f"[yellow]Warning: Loaded point cloud is empty.[/]")
            else:
                CONSOLE.print(f"[green]Successfully loaded point cloud with {len(scene_pcd.points)} points.[/]")
                
                # Store original colors for later use
                original_colors = np.asarray(scene_pcd.colors)
                
                # Store vertices for later use
                ply_vertices = np.asarray(scene_pcd.points)
        except Exception as e:
            CONSOLE.print(f"[red]Error loading point cloud: {str(e)}[/]")
            scene_pcd = None
        
        # Export point cloud as temporary GLB file for visualization
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
            point_cloud_path = tmp.name

        if scene_pcd is not None:
            # Create a trimesh point cloud for visualization with densification
            CONSOLE.print("[blue]Creating enhanced point cloud visualization...[/]")
            
            # Rotate the point cloud 90 degrees on the x-axis first
            CONSOLE.print("[blue]Rotating point cloud to Y up for correct visualization[/]")
            rotation = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi/2 ,0, 0])  # 90 degrees in radians
            scene_pcd.rotate(rotation, center=np.array([0, 0, 0]))
            
            # Get points and colors
            points = np.asarray(scene_pcd.points)
            colors = np.asarray(scene_pcd.colors)
            
            # Densify the point cloud for better visualization
            dense_points, dense_colors = densify_point_cloud(points, colors, density_factor=2.0)
            
            # Create point cloud with denser representation
            cloud = trimesh.points.PointCloud(dense_points, colors=dense_colors)
            
            # Create a scene and export to GLB
            scene = trimesh.Scene(cloud)
            scene.export(point_cloud_path)
            CONSOLE.print(f"[green]Exported enhanced point cloud to temporary GLB: {point_cloud_path}[/]")
            
            return point_cloud_path, "Scene loaded from point cloud. Continuing initialization in background..."
        else:
            CONSOLE.print(f"[red]Error: Could not load point cloud from {ply_path}[/]")
            return None, f"Error: Could not load point cloud from {ply_path}"
        
    except Exception as e:
        import traceback
        error_msg = f"Error loading scene: {str(e)}"
        CONSOLE.print(f"[red]{error_msg}[/]")
        traceback.print_exc()
        return None, error_msg


def complete_initialization(scene_name, data_path, work_dir, lerf_type="lerf-big"):
    """Complete the initialization process after the model is loaded."""
    global scene_analyzer, scene_config
    
    try:
        CONSOLE.print(Panel(f"[bold cyan]Completing Scene Initialization:[/] {scene_name}", border_style="blue"))
        
        # Generate complete paths
        output_dir = data_path
        
        # Update scene config with complete paths
        scene_config.update({
            'config_path': f"{work_dir}/outputs/{scene_name}/{lerf_type}/{scene_name}/config.yml",
            'dataparser_transforms': f"{work_dir}/outputs/{scene_name}/{lerf_type}/{scene_name}/dataparser_transforms.json",
            'nerf_exported_mesh_path': f"{output_dir}/point_cloud.ply",
            'h5_file_path': f"{output_dir}/embeddings_v2.h5"
        })
        
        # Check if all required files exist
        required_files = [
            scene_config['config_path'],
            scene_config['nerf_exported_mesh_path'],
            scene_config['h5_file_path']
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            error_msg = "Missing required files for initialization. Please run the training pipeline first."
            CONSOLE.print(f"[red]{error_msg}[/]")
            CONSOLE.print("[red]Missing files:[/]")
            for f in missing_files:
                CONSOLE.print(f"[red]- {f}[/]")
            return error_msg
        
        # Create a paths table for display
        path_table = Table(title="Initialization Paths", show_header=True, header_style="bold magenta")
        path_table.add_column("Path Type", style="cyan")
        path_table.add_column("Location", style="yellow", width=60)
        path_table.add_column("Status", style="green")
        
        for key, value in scene_config.items():
            if isinstance(value, str) and (key.endswith('_path') or key.endswith('_transforms')):
                status = "✓ Found" if os.path.exists(value) else "✗ Missing"
                status_style = "green" if os.path.exists(value) else "red"
                path_table.add_row(key, value, f"[{status_style}]{status}[/{status_style}]")
        
        CONSOLE.print(path_table)
        
        # Initialize CLIP model
        CONSOLE.print("[blue]Initializing CLIP model...[/]")
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            precision="fp16"
        )
        model.eval()
        model = model.to("cuda")
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        CONSOLE.print("[green]✓ CLIP model initialized[/]")

        # Initialize negative embeddings
        CONSOLE.print("[blue]Initializing negative embeddings...[/]")
        negatives = ["object", "things", "stuff", "texture"]
        with torch.no_grad():
            tok_phrases = torch.cat([tokenizer(phrase) for phrase in negatives]).to("cuda")
            neg_embeds = model.encode_text(tok_phrases)
            neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)
        CONSOLE.print("[green]✓ Negative embeddings created[/]")

        # Initialize LERF pipeline
        CONSOLE.print("[blue]Initializing LERF pipeline...[/]")
        lerf_pipeline = SceneAnalyzerFactory.initialize_lerf_pipeline(
            scene_config['config_path'], 
            scene_name
        )
        CONSOLE.print("[green]✓ LERF pipeline initialized[/]")
        
        # Load H5 file
        CONSOLE.print("[blue]Loading H5 file...[/]")
        h5_dict = SceneAnalyzerFactory.load_h5_file(scene_config['h5_file_path'])
        CONSOLE.print("[green]✓ H5 file loaded[/]")
        
        # Create the scene analyzer
        CONSOLE.print("[blue]Creating scene analyzer...[/]")
        scene_analyzer = SceneAnalyzer(
            scene_name=scene_name,
            lerf_pipeline=lerf_pipeline,
            h5_dict=h5_dict,
            clip_model=model,
            tokenizer=tokenizer,
            neg_embeds=neg_embeds,
            negative_words_length=4,
            axis_align_matrix=None
        )
        
        CONSOLE.print("[bold green]✓ Scene initialized successfully![/]")
        
        return "Scene initialized successfully and ready for queries!"
        
    except Exception as e:
        import traceback
        error_msg = f"Error completing initialization: {str(e)}"
        CONSOLE.print(f"[red]{error_msg}[/]")
        traceback.print_exc()
        return error_msg

def normalize_values(values, min_val=None, max_val=None):
    """
    Normalize values to 0-1 range using min-max normalization.
    
    Args:
        values: Array of values to normalize
        min_val: Optional minimum value (if None, uses min of values)
        max_val: Optional maximum value (if None, uses max of values)
        
    Returns:
        Array of normalized values in range 0-1
    """
    if min_val is None:
        min_val = np.min(values)
    if max_val is None:
        max_val = np.max(values)
        
    # Prevent division by zero
    if max_val == min_val:
        CONSOLE.print("[yellow]Warning: Min and max values are equal, cannot normalize.[/]")
        return np.ones_like(values) * 0.5  # Return middle value
        
    # Apply min-max normalization
    return (values - min_val) / (max_val - min_val)

def generate_heatmap_visualization(query, points, possibility_array, threshold=0.55, colormap=None, normalize=False, use_spheres=False, density_factor=2.0):
    """
    Generate a heatmap visualization based on query results.
    
    Args:
        query: The text query
        points: Points array from find_centroids_bbox
        possibility_array: Possibility array from find_centroids_bbox
        threshold: Confidence threshold for showing colored points
        colormap: Matplotlib colormap to use
        normalize: Whether to apply global min-max normalization
        use_spheres: Whether to use sphere representation instead of points
        density_factor: Factor for point cloud densification
        
    Returns:
        Tuple of (model_path, status_message, ply_path)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import open3d as o3d
    import tempfile
    import trimesh
    import copy
    
    global last_result_pcd
    
    if colormap is None:
        colormap = plt.get_cmap('turbo')
    
    try:
        CONSOLE.print("[blue]Generating enhanced heatmap visualization...[/]")
        
        # Apply normalization if requested
        if normalize:
            CONSOLE.print("[blue]Applying global min-max normalization...[/]")
            min_val = np.min(possibility_array)
            max_val = np.max(possibility_array)
            normalized_values = normalize_values(possibility_array, min_val, max_val)
            CONSOLE.print(f"[blue]Normalized range: [{min_val:.4f}, {max_val:.4f}] -> [0, 1][/]")
            color_values = normalized_values
        else:
            CONSOLE.print("[blue]Using raw confidence scores (no normalization)...[/]")
            color_values = possibility_array
        
        # Initialize the heatmap colors array (RGBA)
        heatmap_colors = np.zeros((len(color_values), 4))
        
        # Create color mapping based on possibility values for ALL points
        for i, value in enumerate(color_values):
            # Color all points based on their possibility score
            heatmap_colors[i] = colormap(value)
        
        # Convert RGBA to RGB for Open3D
        heatmap_colors = heatmap_colors.reshape(-1, 4)[:, :3]
        
        # Densify the point cloud for better visualization
        if density_factor > 1.0:
            dense_points, dense_colors = densify_point_cloud(points, heatmap_colors, density_factor=density_factor)
        else:
            dense_points, dense_colors = points, heatmap_colors
        
        # Create point cloud with heatmap colors
        CONSOLE.print("[blue]Creating enhanced heatmap point cloud...[/]")
        transformed_points = np.ascontiguousarray(dense_points, dtype=np.float64)
        pcd_with_heatmap = o3d.geometry.PointCloud()
        pcd_with_heatmap.points = o3d.utility.Vector3dVector(transformed_points)
        pcd_with_heatmap.colors = o3d.utility.Vector3dVector(dense_colors)

        # Rotate the point cloud 90 degrees on the x-axis
        CONSOLE.print("[blue]Rotating point cloud to Y up for correct visualization[/]")
        rotation = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi/2, 0, 0])  # 90 degrees in radians
        pcd_with_heatmap.rotate(rotation, center=np.array([0, 0, 0]))
        
        # Store for download
        last_result_pcd = pcd_with_heatmap
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
            glb_path = tmp.name
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            ply_path = tmp.name
        
        # Save PLY file for download
        o3d.io.write_point_cloud(ply_path, pcd_with_heatmap)
        
        if use_spheres and len(dense_points) < 5000:  # Only use spheres for smaller point clouds
            # Create sphere representation
            CONSOLE.print("[blue]Creating sphere representation...[/]")
            scene = create_sphere_representation(
                np.asarray(pcd_with_heatmap.points), 
                np.asarray(pcd_with_heatmap.colors)
            )
            scene.export(glb_path)
        else:
            # Convert to trimesh for GLB export
            CONSOLE.print("[blue]Converting to trimesh for GLB export...[/]")
            cloud = trimesh.points.PointCloud(
                np.asarray(pcd_with_heatmap.points), 
                colors=np.asarray(pcd_with_heatmap.colors)
            )
            
            # Create a scene and export to GLB
            scene = trimesh.Scene(cloud)
            scene.export(glb_path)
        
        # Prepare result message
        high_score_points = np.sum(possibility_array >= threshold)
        norm_text = "with global min-max normalization" if normalize else "without normalization"
        density_text = f"(densified {density_factor}x)" if density_factor > 1.0 else ""
        sphere_text = "with spheres" if use_spheres else "with points"
        
        result_msg = (f"Enhanced Heatmap visualization {norm_text} {density_text} {sphere_text} for query: '{query}' (threshold: {threshold})\n"
                     f"Found {high_score_points} points above threshold\n"
                     f"Visualization has {len(dense_points)} points")
        
        CONSOLE.print(f"[green]✓ {result_msg}[/]")
        CONSOLE.print(f"[green]✓ Exported heatmap to: {glb_path}[/]")
        CONSOLE.print(f"[green]✓ Saved PLY to: {ply_path}[/]")
        return glb_path, result_msg, ply_path
        
    except Exception as e:
        import traceback
        error_msg = f"Error generating heatmap: {str(e)}"
        CONSOLE.print(f"[red]{error_msg}[/]")
        traceback.print_exc()
        
        # In case of error, return the best available model path
        CONSOLE.print("[yellow]Falling back to original PLY file due to error[/]")
        return scene_config.get('ply_path'), error_msg, None
        
def generate_filtered_heatmap_visualization(query, points, possibility_array, threshold=0.55, colormap=None, normalize=False, use_spheres=False, density_factor=2.0):
    """
    Generate a heatmap visualization based on query results, filtering out low-confidence points.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import open3d as o3d
    import tempfile
    import trimesh
    import copy
    
    global last_result_pcd
    
    if colormap is None:
        colormap = plt.get_cmap('turbo')
    
    try:
        CONSOLE.print("[blue]Generating enhanced filtered heatmap visualization...[/]")
        
        # Filter points that exceed the threshold
        high_score_indices = np.where(possibility_array >= threshold)[0]
        high_score_points = points[high_score_indices]
        high_score_values = possibility_array[high_score_indices]
        
        if len(high_score_points) == 0:
            CONSOLE.print(f"[yellow]No points found above threshold {threshold}. Try lowering the threshold.[/]")
            return scene_config.get('ply_path'), f"No points found above threshold {threshold}. Try lowering the threshold.", None
        
        CONSOLE.print(f"[blue]Found {len(high_score_points)} points above threshold {threshold}[/]")
        
        # Apply normalization if requested
        if normalize:
            CONSOLE.print("[blue]Applying global min-max normalization...[/]")
            # Normalize based on the entire possibility_array, not just high scores
            min_val = np.min(possibility_array)
            max_val = np.max(possibility_array)
            normalized_values = normalize_values(high_score_values, min_val, max_val)
            CONSOLE.print(f"[blue]Normalized range: [{min_val:.4f}, {max_val:.4f}] -> [0, 1][/]")
            color_values = normalized_values
        else:
            CONSOLE.print("[blue]Using raw confidence scores (no normalization)...[/]")
            color_values = high_score_values
        
        # Initialize the heatmap colors array (RGBA)
        heatmap_colors = np.zeros((len(color_values), 4))
        
        # Create color mapping based on possibility values for high-confidence points
        for i, value in enumerate(color_values):
            heatmap_colors[i] = colormap(value)
        
        # Convert RGBA to RGB for Open3D
        heatmap_colors = heatmap_colors.reshape(-1, 4)[:, :3]
        
        # Densify the point cloud for better visualization
        if density_factor > 1.0:
            dense_points, dense_colors = densify_point_cloud(high_score_points, heatmap_colors, density_factor=density_factor)
        else:
            dense_points, dense_colors = high_score_points, heatmap_colors

        # Create point cloud with heatmap colors
        CONSOLE.print("[blue]Creating enhanced filtered point cloud...[/]")
        pcd_with_heatmap = o3d.geometry.PointCloud()
        pcd_with_heatmap.points = o3d.utility.Vector3dVector(dense_points)
        pcd_with_heatmap.colors = o3d.utility.Vector3dVector(dense_colors)
        

        # Rotate the point cloud 90 degrees on the x-axis
        CONSOLE.print("[blue]Rotating point cloud to Y up for correct visualization[/]")
        rotation = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi/2, 0, 0])  # 90 degrees in radians
        pcd_with_heatmap.rotate(rotation, center=np.array([0, 0, 0]))

        # Store for download
        last_result_pcd = pcd_with_heatmap
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
            glb_path = tmp.name
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            ply_path = tmp.name
        
        # Save PLY file for download
        o3d.io.write_point_cloud(ply_path, pcd_with_heatmap)
        
        if use_spheres and len(dense_points) < 5000:  # Only use spheres for smaller point clouds
            # Create sphere representation
            CONSOLE.print("[blue]Creating sphere representation...[/]")
            scene = create_sphere_representation(
                np.asarray(pcd_with_heatmap.points), 
                np.asarray(pcd_with_heatmap.colors)
            )
            scene.export(glb_path)
        else:
            # Convert to trimesh for GLB export
            CONSOLE.print("[blue]Converting to trimesh for GLB export...[/]")
            cloud = trimesh.points.PointCloud(
                np.asarray(pcd_with_heatmap.points), 
                colors=np.asarray(pcd_with_heatmap.colors)
            )
            
            # Create a scene and export to GLB
            scene = trimesh.Scene(cloud)
            scene.export(glb_path)
        
        # Prepare result message
        norm_text = "with global min-max normalization" if normalize else "without normalization"
        density_text = f"(densified {density_factor}x)" if density_factor > 1.0 else ""
        sphere_text = "with spheres" if use_spheres else "with points"
        
        result_msg = (f"Enhanced Filtered Heatmap visualization {norm_text} {density_text} {sphere_text} for query: '{query}' (threshold: {threshold})\n"
                     f"Found {len(high_score_points)} points above threshold\n"
                     f"Visualization has {len(dense_points)} points")
        
        CONSOLE.print(f"[green]✓ {result_msg}[/]")
        CONSOLE.print(f"[green]✓ Exported filtered heatmap to: {glb_path}[/]")
        CONSOLE.print(f"[green]✓ Saved PLY to: {ply_path}[/]")
        return glb_path, result_msg, ply_path
        
    except Exception as e:
        import traceback
        error_msg = f"Error generating filtered heatmap: {str(e)}"
        CONSOLE.print(f"[red]{error_msg}[/]")
        traceback.print_exc()
        
        # In case of error, return the best available model path
        CONSOLE.print("[yellow]Falling back to original PLY file due to error[/]")
        return scene_config.get('ply_path'), error_msg, None
    

def generate_mixed_color_visualization(query, points, possibility_array, threshold=0.55, colormap=None, normalize=False, use_spheres=False, density_factor=2.0):
    """
    Generate a visualization that preserves original point cloud colors but applies 
    colormap highlighting only to points above the threshold.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import open3d as o3d
    import tempfile
    import trimesh
    import copy
    
    global last_result_pcd
    
    if colormap is None:
        colormap = plt.get_cmap('turbo')
    
    try:
        CONSOLE.print("[blue]Generating enhanced mixed color visualization...[/]")
        
        # Try to load original colors from the nerfstudio point_cloud.ply file
        nerf_ply_path = scene_config.get('nerf_exported_mesh_path')
        original_colors = None
        
        if nerf_ply_path and os.path.exists(nerf_ply_path):
            CONSOLE.print(f"[blue]Loading original colors from: {nerf_ply_path}[/]")
            try:
                # Load the PLY file using Open3D
                original_pcd = o3d.io.read_point_cloud(nerf_ply_path)
                original_points = np.asarray(original_pcd.points)
                original_pcd_colors = np.asarray(original_pcd.colors)
                
                # Check if the point counts match
                if len(original_points) == len(points):
                    CONSOLE.print(f"[green]Successfully loaded original colors from NeRF point cloud.[/]")
                    original_colors = original_pcd_colors
                else:
                    CONSOLE.print(f"[yellow]Point count mismatch: Original PLY has {len(original_points)} points, but query result has {len(points)} points.[/]")
                    
                    # Try to use KD-tree for point matching if counts don't match
                    CONSOLE.print(f"[blue]Attempting to match points using KD-tree...[/]")
                    try:
                        import sklearn.neighbors
                        # Build KD-tree on original points
                        tree = sklearn.neighbors.KDTree(original_points)
                        # Find nearest neighbors for each point in our query result
                        distances, indices = tree.query(points, k=1)
                        # Get colors from matched points
                        matched_colors = original_pcd_colors[indices.flatten()]
                        CONSOLE.print(f"[green]Successfully matched colors using KD-tree.[/]")
                        original_colors = matched_colors
                    except Exception as e:
                        CONSOLE.print(f"[yellow]Error matching points: {str(e)}[/]")
                        
            except Exception as e:
                CONSOLE.print(f"[yellow]Error loading original colors from PLY file: {str(e)}[/]")
                
        # Fall back to global original_colors if available
        if original_colors is None and 'original_colors' in globals() and globals()['original_colors'] is not None:
            CONSOLE.print("[blue]Using global original_colors variable.[/]")
            global_colors = globals()['original_colors']
            
            # Check if global original_colors has the right size
            if len(global_colors) == len(points):
                original_colors = global_colors
            else:
                CONSOLE.print(f"[yellow]Size mismatch: global original_colors has {len(global_colors)} points, but query result has {len(points)} points.[/]")
        
        # Create a default color if original colors are not available
        if original_colors is None:
            CONSOLE.print("[yellow]Original colors not available, using default gray.[/]")
            original_colors = np.ones((len(points), 3)) * 0.7  # Default gray
        
        # Debug: Check shape of original colors
        CONSOLE.print(f"[blue]Original colors shape: {original_colors.shape}[/]")
        
        # Make a copy of the original colors to modify
        mixed_colors = original_colors.copy()
        
        # Apply normalization to possibility scores if requested
        if normalize:
            CONSOLE.print("[blue]Applying global min-max normalization...[/]")
            min_val = np.min(possibility_array)
            max_val = np.max(possibility_array)
            normalized_values = normalize_values(possibility_array, min_val, max_val)
            CONSOLE.print(f"[blue]Normalized range: [{min_val:.4f}, {max_val:.4f}] -> [0, 1][/]")
            color_values = normalized_values
        else:
            CONSOLE.print("[blue]Using raw confidence scores (no normalization)...[/]")
            color_values = possibility_array
        
        # Find high-confidence points
        high_score_indices = np.where(possibility_array >= threshold)[0]
        high_score_count = len(high_score_indices)
        
        if high_score_count == 0:
            CONSOLE.print(f"[yellow]No points found above threshold {threshold}. Try lowering the threshold.[/]")
            return scene_config.get('ply_path'), f"No points found above threshold {threshold}. Try lowering the threshold.", None
        
        CONSOLE.print(f"[blue]Found {high_score_count} points above threshold {threshold}[/]")
        
        # DEBUG: print shape of arrays before color assignment
        CONSOLE.print(f"[blue]Mixed colors shape: {mixed_colors.shape}[/]")
        
        CONSOLE.print(f"[blue]Using turbo colormap highlighting for {high_score_count} high confidence points[/]")
        
        # Apply colors based on confidence score
        for i in high_score_indices:
            # Map the score to a color using the actual colormap
            norm_score = min(max(color_values[i], 0.0), 1.0)  # Clamp to 0-1
            # Use the actual turbo colormap
            rgba_color = colormap(norm_score)
            
            # Handle the case where colormap returns (1, 4) shape
            if rgba_color.shape == (1, 4):
                # Extract the single row and get RGB
                color_row = rgba_color[0]  # Get the first (only) row
                heatmap_color = [color_row[0], color_row[1], color_row[2]]
            elif len(rgba_color) >= 3:
                # Direct array case
                heatmap_color = [rgba_color[0], rgba_color[1], rgba_color[2]]
            else:
                # Fallback
                rgba_color = plt.cm.turbo(norm_score)
                heatmap_color = [rgba_color[0], rgba_color[1], rgba_color[2]]
            
            # Assign the color
            mixed_colors[i] = heatmap_color
        
        # Densify the point cloud for better visualization
        if density_factor > 1.0:
            # For mixed colors, we need to be careful about densification
            # We'll densify but preserve the highlight pattern
            dense_points, dense_colors = densify_point_cloud(points, mixed_colors, density_factor=density_factor)
        else:
            dense_points, dense_colors = points, mixed_colors
        
        # Create point cloud with mixed colors
        CONSOLE.print("[blue]Creating enhanced mixed color point cloud...[/]")
        pcd_with_mixed_colors = o3d.geometry.PointCloud()
        pcd_with_mixed_colors.points = o3d.utility.Vector3dVector(dense_points)
        pcd_with_mixed_colors.colors = o3d.utility.Vector3dVector(dense_colors)
        
        # Rotate the point cloud 90 degrees on the x-axis
        CONSOLE.print("[blue]Rotating point cloud to Y up for correct visualization[/]")
        rotation = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi/2, 0, 0])  # 90 degrees in radians
        pcd_with_mixed_colors.rotate(rotation, center=np.array([0, 0, 0]))

        # Store for download
        last_result_pcd = pcd_with_mixed_colors
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
            glb_path = tmp.name
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            ply_path = tmp.name
        
        # Save PLY file for download
        o3d.io.write_point_cloud(ply_path, pcd_with_mixed_colors)
        
        if use_spheres and len(dense_points) < 5000:  # Only use spheres for smaller point clouds
            # Create sphere representation
            CONSOLE.print("[blue]Creating sphere representation...[/]")
            scene = create_sphere_representation(
                np.asarray(pcd_with_mixed_colors.points), 
                np.asarray(pcd_with_mixed_colors.colors)
            )
            scene.export(glb_path)
        else:
            # Convert to trimesh for GLB export
            CONSOLE.print("[blue]Converting to trimesh for GLB export...[/]")
            cloud = trimesh.points.PointCloud(
                np.asarray(pcd_with_mixed_colors.points), 
                colors=np.asarray(pcd_with_mixed_colors.colors)
            )
            
            # Create a scene and export to GLB
            scene = trimesh.Scene(cloud)
            scene.export(glb_path)
        
        # Prepare result message
        norm_text = "with global min-max normalization" if normalize else "without normalization"
        density_text = f"(densified {density_factor}x)" if density_factor > 1.0 else ""
        sphere_text = "with spheres" if use_spheres else "with points"
        
        result_msg = (f"Enhanced Mixed color visualization {norm_text} {density_text} {sphere_text} for query: '{query}' (threshold: {threshold})\n"
                     f"Found {high_score_count} points above threshold (highlighted with colormap)\n"
                     f"Visualization has {len(dense_points)} points")
        
        CONSOLE.print(f"[green]✓ {result_msg}[/]")
        CONSOLE.print(f"[green]✓ Exported mixed color visualization to: {glb_path}[/]")
        CONSOLE.print(f"[green]✓ Saved PLY to: {ply_path}[/]")
        return glb_path, result_msg, ply_path
        
    except Exception as e:
        import traceback
        error_msg = f"Error generating mixed color visualization: {str(e)}"
        CONSOLE.print(f"[red]{error_msg}[/]")
        traceback.print_exc()
        
        # In case of error, return the best available model path
        CONSOLE.print("[yellow]Falling back to original PLY file due to error[/]")
        return scene_config.get('ply_path'), error_msg, None


def generate_filtered_original_colors_visualization(query, points, possibility_array, threshold=0.55, colormap=None, normalize=False, use_spheres=False, density_factor=2.0):
    """
    Generate a visualization showing only points above threshold with their original colors (no heatmap coloring).
    
    Args:
        query: The text query
        points: Points array from find_centroids_bbox
        possibility_array: Possibility array from find_centroids_bbox
        threshold: Confidence threshold for filtering points
        colormap: Not used in this visualization (kept for consistency)
        normalize: Not used in this visualization (kept for consistency)
        use_spheres: Whether to use sphere representation instead of points  
        density_factor: Factor for point cloud densification
        
    Returns:
        Tuple of (model_path, status_message, ply_path)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import open3d as o3d
    import tempfile
    import trimesh
    import copy
    
    global last_result_pcd
    
    try:
        CONSOLE.print("[blue]Generating filtered original colors visualization...[/]")
        
        # Filter points that exceed the threshold
        high_score_indices = np.where(possibility_array >= threshold)[0]
        high_score_points = points[high_score_indices]
        
        if len(high_score_points) == 0:
            CONSOLE.print(f"[yellow]No points found above threshold {threshold}. Try lowering the threshold.[/]")
            return scene_config.get('ply_path'), f"No points found above threshold {threshold}. Try lowering the threshold.", None
        
        CONSOLE.print(f"[blue]Found {len(high_score_points)} points above threshold {threshold}[/]")
        
        # Try to load original colors from the nerfstudio point_cloud.ply file
        nerf_ply_path = scene_config.get('nerf_exported_mesh_path')
        original_colors = None
        
        if nerf_ply_path and os.path.exists(nerf_ply_path):
            CONSOLE.print(f"[blue]Loading original colors from: {nerf_ply_path}[/]")
            try:
                # Load the PLY file using Open3D
                original_pcd = o3d.io.read_point_cloud(nerf_ply_path)
                original_points = np.asarray(original_pcd.points)
                original_pcd_colors = np.asarray(original_pcd.colors)
                
                # Check if the point counts match
                if len(original_points) == len(points):
                    CONSOLE.print(f"[green]Successfully loaded original colors from NeRF point cloud.[/]")
                    original_colors = original_pcd_colors
                else:
                    CONSOLE.print(f"[yellow]Point count mismatch: Original PLY has {len(original_points)} points, but query result has {len(points)} points.[/]")
                    
                    # Try to use KD-tree for point matching if counts don't match
                    CONSOLE.print(f"[blue]Attempting to match points using KD-tree...[/]")
                    try:
                        import sklearn.neighbors
                        # Build KD-tree on original points
                        tree = sklearn.neighbors.KDTree(original_points)
                        # Find nearest neighbors for each point in our query result
                        distances, indices = tree.query(points, k=1)
                        # Get colors from matched points
                        matched_colors = original_pcd_colors[indices.flatten()]
                        CONSOLE.print(f"[green]Successfully matched colors using KD-tree.[/]")
                        original_colors = matched_colors
                    except Exception as e:
                        CONSOLE.print(f"[yellow]Error matching points: {str(e)}[/]")
                        
            except Exception as e:
                CONSOLE.print(f"[yellow]Error loading original colors from PLY file: {str(e)}[/]")
                
        # Fall back to global original_colors if available
        if original_colors is None and 'original_colors' in globals() and globals()['original_colors'] is not None:
            CONSOLE.print("[blue]Using global original_colors variable.[/]")
            global_colors = globals()['original_colors']
            
            # Check if global original_colors has the right size
            if len(global_colors) == len(points):
                original_colors = global_colors
            else:
                CONSOLE.print(f"[yellow]Size mismatch: global original_colors has {len(global_colors)} points, but query result has {len(points)} points.[/]")
        
        # Create a default color if original colors are not available
        if original_colors is None:
            CONSOLE.print("[yellow]Original colors not available, using default gray.[/]")
            original_colors = np.ones((len(points), 3)) * 0.7  # Default gray
        
        # Get the original colors for the filtered high-confidence points
        filtered_original_colors = original_colors[high_score_indices]
        
        CONSOLE.print(f"[blue]Using original colors for {len(high_score_points)} filtered points[/]")
        
        # Densify the point cloud for better visualization
        if density_factor > 1.0:
            dense_points, dense_colors = densify_point_cloud(high_score_points, filtered_original_colors, density_factor=density_factor)
        else:
            dense_points, dense_colors = high_score_points, filtered_original_colors

        # Create point cloud with original colors
        CONSOLE.print("[blue]Creating filtered original colors point cloud...[/]")
        pcd_with_original_colors = o3d.geometry.PointCloud()
        pcd_with_original_colors.points = o3d.utility.Vector3dVector(dense_points)
        pcd_with_original_colors.colors = o3d.utility.Vector3dVector(dense_colors)
        
        # Rotate the point cloud 90 degrees on the x-axis
        CONSOLE.print("[blue]Rotating point cloud to Y up for correct visualization[/]")
        rotation = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi/2, 0, 0])  # 90 degrees in radians
        pcd_with_original_colors.rotate(rotation, center=np.array([0, 0, 0]))

        # Store for download
        last_result_pcd = pcd_with_original_colors
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
            glb_path = tmp.name
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            ply_path = tmp.name
        
        # Save PLY file for download
        o3d.io.write_point_cloud(ply_path, pcd_with_original_colors)
        
        if use_spheres and len(dense_points) < 5000:  # Only use spheres for smaller point clouds
            # Create sphere representation
            CONSOLE.print("[blue]Creating sphere representation...[/]")
            scene = create_sphere_representation(
                np.asarray(pcd_with_original_colors.points), 
                np.asarray(pcd_with_original_colors.colors)
            )
            scene.export(glb_path)
        else:
            # Convert to trimesh for GLB export
            CONSOLE.print("[blue]Converting to trimesh for GLB export...[/]")
            cloud = trimesh.points.PointCloud(
                np.asarray(pcd_with_original_colors.points), 
                colors=np.asarray(pcd_with_original_colors.colors)
            )
            
            # Create a scene and export to GLB
            scene = trimesh.Scene(cloud)
            scene.export(glb_path)
        
        # Prepare result message
        density_text = f"(densified {density_factor}x)" if density_factor > 1.0 else ""
        sphere_text = "with spheres" if use_spheres else "with points"
        
        result_msg = (f"Filtered Original Colors visualization {density_text} {sphere_text} for query: '{query}' (threshold: {threshold})\n"
                     f"Found {len(high_score_points)} points above threshold (shown with original colors)\n"
                     f"Visualization has {len(dense_points)} points")
        
        CONSOLE.print(f"[green]✓ {result_msg}[/]")
        CONSOLE.print(f"[green]✓ Exported filtered original colors visualization to: {glb_path}[/]")
        CONSOLE.print(f"[green]✓ Saved PLY to: {ply_path}[/]")
        return glb_path, result_msg, ply_path
        
    except Exception as e:
        import traceback
        error_msg = f"Error generating filtered original colors visualization: {str(e)}"
        CONSOLE.print(f"[red]{error_msg}[/]")
        traceback.print_exc()
        
        # In case of error, return the best available model path
        CONSOLE.print("[yellow]Falling back to original PLY file due to error[/]")
        return scene_config.get('ply_path'), error_msg, None


def process_query(query, threshold=0.55, viz_mode="Heatmap", normalize=False, use_spheres=False, density_factor=2.0):
    """
    Process a text query using the specified confidence threshold and visualization mode.
    
    Args:
        query: The text query to search in the scene
        threshold: Confidence threshold for filtering results (0.0 to 1.0)
        viz_mode: Visualization mode ("Heatmap", "Filtered Heatmap", or "Mixed Colors")
        normalize: Whether to apply global min-max normalization for heatmaps
        use_spheres: Whether to use sphere representation instead of points  
        density_factor: Factor for point cloud densification
    
    Returns:
        Tuple of (model_path, status_message, ply_path)
    """
    global scene_analyzer, scene_pcd, ply_vertices, original_colors, scene_mesh
    
    if scene_analyzer is None:
        # Try to return the best available model path
        model_path = scene_config.get('ply_path')
        CONSOLE.print("[red]Error: Scene not initialized. Please initialize the scene first.[/]")
        if model_path and os.path.exists(model_path):
            return model_path, "Error: Scene not initialized. Please initialize the scene first.", None
        return None, "Error: Scene not initialized. Please initialize the scene first.", None
    
    if not query or query.strip() == "":
        CONSOLE.print("[yellow]Please enter a valid query.[/]")
        # Return the best available model path
        return scene_config.get('ply_path'), "Please enter a valid query.", None
    
    try:
        norm_text = "with normalization" if normalize else "without normalization"
        sphere_text = "with spheres" if use_spheres else "with points"
        density_text = f"(densified {density_factor}x)" if density_factor > 1.0 else ""
        
        CONSOLE.print(Panel(f"[bold cyan]Processing Query:[/] {query} (threshold: {threshold}, mode: {viz_mode} {norm_text} {sphere_text} {density_text})", border_style="blue"))
        
        # Get query results using the scene analyzer
        CONSOLE.print("[blue]Running query through scene analyzer...[/]")
        centroids, scores, _, points, possibility_array, _, _ = scene_analyzer.find_centroids_bbox(query)
        
        if points is None or possibility_array is None or len(possibility_array) == 0:
            CONSOLE.print("[yellow]No matching points found for the query.[/]")
            return scene_config.get('ply_path'), "No matching points found for the query.", None
        
        # Create colormap for visualizations
        import matplotlib.pyplot as plt
        colormap = plt.get_cmap('turbo')
        
        # Statistics about results
        high_score_count = np.sum(possibility_array >= threshold)
        avg_score = np.mean(possibility_array)
        max_score = np.max(possibility_array)
        min_score = np.min(possibility_array)
        
        CONSOLE.print(f"[green]Query results:[/]")
        CONSOLE.print(f"[green]- Points found: {len(points)}[/]")
        CONSOLE.print(f"[green]- Points above threshold ({threshold}): {high_score_count}[/]")
        CONSOLE.print(f"[green]- Score range: [{min_score:.4f}, {max_score:.4f}][/]")
        CONSOLE.print(f"[green]- Average score: {avg_score:.4f}[/]")
        
        # Handle different visualization modes
        if viz_mode == "Heatmap":
            CONSOLE.print("[blue]Selected visualization mode: Enhanced Heatmap[/]")
            return generate_heatmap_visualization(
                query=query,
                points=points,
                possibility_array=possibility_array,
                threshold=threshold,
                colormap=colormap,
                normalize=normalize,
                use_spheres=use_spheres,
                density_factor=density_factor
            )
        elif viz_mode == "Mixed Colors":
            CONSOLE.print("[blue]Selected visualization mode: Enhanced Mixed Colors[/]")
            return generate_mixed_color_visualization(
                query=query,
                points=points,
                possibility_array=possibility_array,
                threshold=threshold,
                colormap=colormap,
                normalize=normalize,
                use_spheres=use_spheres,
                density_factor=density_factor
            )
        elif viz_mode == "Filtered Heatmap":
            CONSOLE.print("[blue]Selected visualization mode: Enhanced Filtered Heatmap[/]")
            return generate_filtered_heatmap_visualization(
                query=query,
                points=points,
                possibility_array=possibility_array,
                threshold=threshold,
                colormap=colormap,
                normalize=normalize,
                use_spheres=use_spheres,
                density_factor=density_factor
            )
        elif viz_mode == "Filtered Original Colors":
            CONSOLE.print("[blue]Selected visualization mode: Filtered Original Colors[/]")
            return generate_filtered_original_colors_visualization(
                query=query,
                points=points,
                possibility_array=possibility_array,
                threshold=threshold,
                colormap=colormap,
                normalize=normalize,
                use_spheres=use_spheres,
                density_factor=density_factor
            )
        else:
            # Default to heatmap if unrecognized mode
            CONSOLE.print(f"[yellow]Unrecognized visualization mode: {viz_mode}. Defaulting to Enhanced Heatmap.[/]")
            return generate_heatmap_visualization(
                query=query,
                points=points,
                possibility_array=possibility_array,
                threshold=threshold,
                colormap=colormap,
                normalize=normalize,
                use_spheres=use_spheres,
                density_factor=density_factor
            )
            
    except Exception as e:
        import traceback
        error_msg = f"Error processing query: {str(e)}"
        CONSOLE.print(f"[red]{error_msg}[/]")
        traceback.print_exc()
        
        # In case of error, return the best available model path
        CONSOLE.print("[yellow]Falling back to original PLY file due to error[/]")
        return scene_config.get('ply_path'), error_msg, None

def download_last_result():
    """Download the last generated point cloud result as PLY file."""
    global last_result_pcd
    
    if last_result_pcd is None:
        CONSOLE.print("[yellow]No result available for download. Please run a query first.[/]")
        return None
    
    try:
        # Create a temporary file for download
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            ply_path = tmp.name
        
        # Save the point cloud
        o3d.io.write_point_cloud(ply_path, last_result_pcd)
        
        CONSOLE.print(f"[green]✓ Prepared result for download: {ply_path}[/]")
        return ply_path
        
    except Exception as e:
        CONSOLE.print(f"[red]Error preparing download: {str(e)}[/]")
        return None

def create_gradio_interface():
    """Create and launch the Gradio interface for the LERF system."""
    with gr.Blocks(title="OpenLeRF3D Interface") as interface:
        gr.Markdown("# OpenLeRF3D Interface")
        
        # Global variable to store the threshold value
        query_threshold = gr.State(value=0.55)
        download_path = gr.State(value=None)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Scene initialization controls
                with gr.Group():
                    gr.Markdown("### Scene Configuration")
                    scene_name = gr.Textbox(label="Scene Name", value="nerfstudio_scene")
                    
                    # NeRF Studio data path input
                    data_path = gr.Textbox(
                        label="NeRF Studio Data Path",
                        value="",
                        placeholder="Path to the NeRF Studio data directory"
                    )
                    
                    work_dir = gr.Textbox(
                        label="Work Directory", 
                        value="/media/dc-04-vol03/Niccolo/openlerf3d_git"
                    )
                    
                    with gr.Accordion("Advanced Options", open=False):
                        lerf_type = gr.Dropdown(
                            label="LERF Type", 
                            choices=["lerf-big", "lerf", "depthlerf-big", "depthlerf"], 
                            value="lerf-big"
                        )
                    
                    # Add a "Run Training" button
                    train_button = gr.Button("Run Training Pipeline", variant="secondary")
                    init_button = gr.Button("Initialize Scene", variant="primary")
                
                # Status display - simple text status
                status_main = gr.Textbox(
                    label="Status", 
                    value="Not initialized", 
                    elem_id="status_main"
                )
                
                # Detailed logs in collapsible section
                with gr.Accordion("Detailed Logs", open=False, elem_id="logs_accordion"):
                    logs_display = gr.Textbox(
                        value="No logs available", 
                        elem_id="logs_display",
                        lines=15
                    )
                
                # Query controls
                with gr.Group():
                    gr.Markdown("### Query")
                    query_input = gr.Textbox(
                        label="Text Query", 
                        placeholder="Enter text to search in the scene...",
                        value="window"
                    )
                    
                    # Move the Run Query button here, right after the text input
                    query_button = gr.Button("Run Query", variant="primary")
                    
                    # Visualization mode selector
                    viz_mode = gr.Radio(
                        choices=["Heatmap", "Filtered Heatmap", "Mixed Colors", "Filtered Original Colors"], 
                        value="Heatmap", 
                        label="Visualization Mode",
                        info="Heatmap: All points colored by confidence | Filtered Heatmap: Only high-confidence points with heatmap colors | Mixed Colors: Original colors + highlighted matches | Filtered Original Colors: Only high-confidence points with original colors"
                    )
                    
                    # Threshold slider in the Query section
                    threshold_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=0.9, 
                        value=0.55, 
                        step=0.05, 
                        label="Confidence Threshold", 
                        info="Higher values are more strict (fewer but more confident matches)"
                    )
                    
                    # Add the normalization toggle
                    normalize_toggle = gr.Checkbox(
                        value=False,
                        label="Enable Global Min-Max Normalization",
                        info="Rescale confidence scores to 0-1 range for better visualization"
                    )
                    
                    # Add visualization enhancement options
                    with gr.Accordion("Visualization Enhancements", open=True):
                        use_spheres = gr.Checkbox(
                            value=False,
                            label="Use Sphere Representation (for smaller point clouds)",
                            info="Render points as small spheres instead of simple points"
                        )
                        
                        density_factor = gr.Slider(
                            minimum=1.0,
                            maximum=5.0,
                            value=1.0,
                            step=0.5,
                            label="Point Density Factor",
                            info="Multiply point count for denser visualization (higher = denser)"
                        )
                    
                    # Update the query_threshold State when the slider changes
                    threshold_slider.change(
                        fn=lambda x: x,
                        inputs=[threshold_slider],
                        outputs=[query_threshold]
                    )
                
                # Download section
                with gr.Group():
                    gr.Markdown("### Download Results")
                    download_button = gr.Button("Download Last Result as PLY", variant="secondary")
                    download_file = gr.File(
                        label="Download File",
                        visible=False
                    )
                
                # Examples section
                gr.Markdown("### Example Queries")
                examples = [
                    "window",
                    "door", 
                    "wall",
                    "floor",
                    "ceiling",
                    "furniture"
                ]
                for example in examples:
                    example_button = gr.Button(example)
                    example_button.click(
                        fn=lambda ex=example: ex,
                        inputs=[],
                        outputs=[query_input]
                    )
            
            # Visualization area - optimized for GLB files
            with gr.Column(scale=2):
                point_cloud_display = gr.Model3D(
                    label="Scene Visualization",
                    clear_color=[1.0, 1.0, 1.0, 1.0],  # White background
                    camera_position=[1, 1, 1],         # Set initial camera position
                    height=900,                        # Set a fixed height
                    display_mode="solid"               # Use solid display mode for better rendering
                )
        
        # Set up event handlers
        # Training button - run the training pipeline
        train_button.click(
            fn=check_and_run_training,
            inputs=[scene_name, data_path, work_dir, lerf_type],
            outputs=[status_main, logs_display, point_cloud_display]
        )
        
        # First display the model, then complete initialization in the background
        init_button.click(
            fn=initialize_scene,
            inputs=[scene_name, data_path, work_dir, lerf_type],
            outputs=[point_cloud_display, status_main]
        ).then(
            fn=complete_initialization,
            inputs=[scene_name, data_path, work_dir, lerf_type],
            outputs=[status_main]
        )
        
        # Use the modified process_query function that accepts all enhancement parameters
        def handle_query(query, threshold, viz_mode, normalize, use_spheres, density_factor):
            glb_path, status, ply_path = process_query(query, threshold, viz_mode, normalize, use_spheres, density_factor)
            return glb_path, status, ply_path
        
        query_button.click(
            fn=handle_query,
            inputs=[query_input, query_threshold, viz_mode, normalize_toggle, use_spheres, density_factor],
            outputs=[point_cloud_display, status_main, download_path]
        )
        
        # Download functionality
        def prepare_download():
            ply_path = download_last_result()
            if ply_path:
                return gr.update(value=ply_path, visible=True)
            else:
                return gr.update(visible=False)
        
        download_button.click(
            fn=prepare_download,
            inputs=[],
            outputs=[download_file]
        )
    
    # Add enhanced CSS to improve display
    css = """
    #status_main {
        min-height: 30px;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 8px;
        background-color: #fafafa;
        margin-bottom: 5px;
        font-weight: bold;
    }
    
    #logs_display {
        font-family: monospace;
        font-size: 0.9em;
    }
    
    .gradio-container {
        max-width: 100% !important;
    }
    """
    
    interface.css = css
    return interface


def apply_mask_to_mesh(mesh, colormap, opacity=0.75):
    """Apply colors to a mesh's vertices."""
    if mesh is None:
        return None
    
    CONSOLE.print("[blue]Applying mask colors to mesh...[/]")
    
    # Get the original mesh vertex colors
    vertices = mesh.vertices
    
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        original_colors = mesh.visual.vertex_colors[:, :3].astype(float) / 255.0
    else:
        # If no vertex colors, use a default color
        original_colors = np.ones((len(vertices), 3)) * 0.7
    
    # Initialize the output colors with the original colors
    colors = original_colors.copy()
    
    CONSOLE.print(f"[blue]Mesh has {len(vertices)} vertices[/]")
    
    # Create a new mesh with the updated colors
    new_mesh = mesh.copy()
    
    # Convert colors to RGBA uint8 format for trimesh
    colors_rgba = np.ones((len(colors), 4)) * 255
    colors_rgba[:, :3] = colors * 255
    
    # Update the vertex colors
    new_mesh.visual.vertex_colors = colors_rgba.astype(np.uint8)
    
    CONSOLE.print("[green]✓ Successfully applied mask colors to mesh[/]")
    return new_mesh


if __name__ == "__main__":
    # Set CUDA device if needed
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Create and launch the interface
    CONSOLE.print(Panel("[bold green]Starting Enhanced OpenLeRF3D Interface[/]", border_style="green"))
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", share=True)