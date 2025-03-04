import os
import subprocess
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import datasets.replica as replica

# Initialize Rich console
CONSOLE = Console()

def display_parameters(params):
    """Display parameters in a formatted table"""
    table = Table(title="Processing Parameters", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")
    
    for key, value in params.items():
        table.add_row(key, str(value))
    
    CONSOLE.print(table)

def run_query(scene_name):
    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Verify that the environment variable is set
    CONSOLE.print(Panel(f"[bold cyan]CUDA_VISIBLE_DEVICES:[/] {os.getenv('CUDA_VISIBLE_DEVICES')}", 
                       title="Environment", border_style="blue"))

    #----- Input parameters -----#
    WORK_DIR = "/media/dc-04-vol03/Niccolo/openlerf3d_replica"
    LERF = "lerf-big"  # or "lerf"
    MASK = "sam"  # or sam-hq
    data_dir = f"{WORK_DIR}/data"
    DATA_NERF_PATH = f"{WORK_DIR}/data/replica_{scene_name}"
    SCENE_NAME = scene_name

    CONSOLE.print(f"\n[bold green]Processing Scene:[/] {SCENE_NAME}")

    # Processing parameters
    params = {
        "VIEW_FREQ": 1,  # sample frequency of the frames
        "THRES_CONNECT": "0.7,0.5,5",  # dynamic threshold for region growing
        "MAX_NEIGHBOR_DISTANCE": 2,  # farthest distance to take neighbors into account
        "THRES_MERGE": 1000,  # merge small groups with less than THRES_MERGE points
        "DIS_DECAY": 0.5,  # decay rate of the distance weight
        "SIMILAR_METRIC": "2-norm",  # metric for similarity measurement
        "MASK_NAME": MASK,  # mask name for loading mask
        "ALIAS_MASK_NAME": MASK  # mask name for saving results
    }

    display_parameters(params)

    # Generate paths
    TEXT_HEAD = "demo_scannet"
    TEXT = f"{TEXT_HEAD}_{params['VIEW_FREQ']}view"
    HEAD = (f"{TEXT_HEAD}_{params['VIEW_FREQ']}view_merge{params['THRES_MERGE']}_"
           f"{params['SIMILAR_METRIC']}_{params['ALIAS_MASK_NAME']}_"
           f"connect({params['THRES_CONNECT']})_depth{params['MAX_NEIGHBOR_DISTANCE']}")

    paths = {
        "CONFIG_PATH": f"{WORK_DIR}/outputs/{SCENE_NAME}/{LERF}/{SCENE_NAME}/config.yml",
        "DATAPARSER_TRANSFORMS": f"{WORK_DIR}/outputs/{SCENE_NAME}/{LERF}/{SCENE_NAME}/dataparser_transforms.json",
        "PLY_PATH": f"{DATA_NERF_PATH}/{SCENE_NAME}_mesh.ply",
        "PATH_PRED_MASKS": f"{WORK_DIR}/data/results/{HEAD}/class_agnostic_masks/{SCENE_NAME}_masks.pt",
        "NERF_EXPORTED_MESH_PATH": f"{DATA_NERF_PATH}/point_cloud.ply",
        "H5_FILE_PATH": f"{DATA_NERF_PATH}/embeddings_v2.h5"
    }

    # Display paths in a table
    path_table = Table(title="File Paths", show_header=True, header_style="bold magenta")
    path_table.add_column("Path Type", style="cyan")
    path_table.add_column("Location", style="yellow", width=60)
    
    for key, value in paths.items():
        path_table.add_row(key, value)
    
    CONSOLE.print(path_table)

    # Run the Python script
    CONSOLE.print("\n[bold yellow]Executing Process[/]")
    CONSOLE.print("[cyan]→[/] Running process_scene.py...")
    
    try:
        subprocess.run([
            "python", "process_scene.py",
            SCENE_NAME, paths["NERF_EXPORTED_MESH_PATH"], paths["H5_FILE_PATH"],
            paths["PLY_PATH"], paths["CONFIG_PATH"], paths["DATAPARSER_TRANSFORMS"],
            paths["PATH_PRED_MASKS"],
        ], check=True)
        CONSOLE.print("[green]✓ Process completed successfully[/]")
    except subprocess.CalledProcessError as e:
        CONSOLE.print(f"[red]✗ Process failed with error code {e.returncode}[/]")
        raise

if __name__ == "__main__":
    CONSOLE.print(Panel("[bold blue]Starting Replica Scene Processing[/]", 
                       border_style="blue"))
    
    for i, scene_name in enumerate(replica.scenes, 1):
        CONSOLE.print(f"\n[bold]Processing Scene {i}/{len(replica.scenes)}[/]")
        CONSOLE.rule(f"Scene: {scene_name}")
        run_query(scene_name)
        CONSOLE.print(f"[dim]Completed scene: {scene_name}[/]")