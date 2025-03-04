from third_party.SAI3D.sam_auto_generation_replica import prepare_2d_mask as prepare_2d_mask_sam
from third_party.SAI3D.sam_sem_auto_generation_replica import prepare_2d_mask as prepare_2d_mask_semantic_sam
import subprocess
import os 
import time
import argparse

import datasets.replica as replica
from rich.console import Console

# Initialize Rich console
CONSOLE = Console()

def run_training(scene_name):
    workdir = "/media/dc-04-vol03/Niccolo/openlerf3d_replica"
    lerf    = "lerf-big"
    mask    = "sam" # "sam-hq" or "sam" or "semantic-sam"

    #Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Verify that the environment variable is set
    CONSOLE.print(f"[bold cyan]CUDA_VISIBLE_DEVICES:[/] {os.getenv('CUDA_VISIBLE_DEVICES')}")
    CONSOLE.print(f"[bold cyan]SCENE_NAME:[/] {scene_name}")
        
    data_dir = f"{workdir}/data"

    if mask == "semantic-sam":
        prepare_2d_mask = prepare_2d_mask_semantic_sam
    # elif mask == "sam-hq":
        # prepare_2d_mask = prepare_2d_mask_sam_hq  
    elif mask == "sam":
        prepare_2d_mask = prepare_2d_mask_sam
    else:
        raise ValueError(f"Unknown mask type: {mask}")
    
    prepare_2d_mask(f"{data_dir}/replica_{scene_name}/", view_freq=1, scene_id=scene_name, downsample_factor=1)

    # get superpoint
    segmentator_path = "./third_party/SAI3D/Segmentator/segmentator"
    ply_path = f"{data_dir}/replica_{scene_name}/{scene_name}_mesh.ply"
    obj_path = f"{data_dir}/replica_{scene_name}/{scene_name}_mesh.obj"
    arg1 = "0.01"
    arg2 = "800"

    command = [segmentator_path, obj_path, arg1, arg2]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        CONSOLE.print("[green]✓ Segmentor executed successfully[/]")
        CONSOLE.print("[dim]Output:[/]", result.stdout)
    else:
        CONSOLE.print("[red]✗ Segmentor failed[/] with return code", result.returncode)
        CONSOLE.print("[red]Error:[/]", result.stderr)
        raise Exception("Segmentor failed")

    # Mask proposal
    VIEW_FREQ             = 1
    THRES_CONNECT         = "0.7,0.5,5"
    MAX_NEIGHBOR_DISTANCE = 2
    THRES_MERGE          = 1000
    DIS_DECAY            = 0.5
    SIMILAR_METRIC       = "2-norm"
    MASK_NAME            = mask
    ALIAS_MASK_NAME      = mask

    TEXT_HEAD = "demo_scannet"
    TEXT      = f"{TEXT_HEAD}_{VIEW_FREQ}view"
    HEAD      = f"{TEXT_HEAD}_{VIEW_FREQ}view_merge{THRES_MERGE}_{SIMILAR_METRIC}_{ALIAS_MASK_NAME}_connect({THRES_CONNECT})_depth{MAX_NEIGHBOR_DISTANCE}"
    EVAL_DIR  = f"{data_dir}/results/{HEAD}"

    REPLICA_PATH = f"{data_dir}/"

    CONSOLE.print("\n[bold yellow]SAI3D Processing[/]")
    CONSOLE.print("[cyan]→[/] Extracting class agnostic masks...")
    subprocess.run([
        "python", "./third_party/SAI3D/sai3d_nerfstudio.py",
        "--base_dir", REPLICA_PATH,
        "--scene_id", scene_name,
        "--thres_merge", str(THRES_MERGE),
        "--similar_metric", SIMILAR_METRIC,
        "--thres_connect", THRES_CONNECT,
        "--mask_name", MASK_NAME,
        "--max_neighbor_distance", str(MAX_NEIGHBOR_DISTANCE),
        "--view_freq", str(VIEW_FREQ),
        "--use_torch",
        "--dis_decay", str(DIS_DECAY),
        "--eval_dir", EVAL_DIR
    ])

    CONSOLE.print("[cyan]→[/] Processing .pt mask file")
    subprocess.run([
        "python", "./third_party/SAI3D/helpers/format_convertion.py",
        "--app=1",
        "--base_dir", f"{data_dir}/results/{HEAD}",
        "--out_dir", f"{data_dir}/results/{HEAD}/class_agnostic_masks"
    ])

    CONSOLE.print("[green]✓ SAI3D Processing Complete[/]")
    CONSOLE.print(f"[dim]Masks saved to {data_dir}/results/{HEAD}/class_agnostic_masks/[/]")

    subprocess.run([
        "python", "./third_party/SAI3D/helpers/visualize_replica.py",
        scene_name, f"{data_dir}/results/{HEAD}", REPLICA_PATH
    ])

    # Training LERF
    CONSOLE.print("\n[bold yellow]LERF Training[/]")
    DATA_NERF_PATH = f"{workdir}/data/replica_{scene_name}"

    CONSOLE.print(f"[cyan]→[/] Training Replica {lerf}...")
    subprocess.run([
        "ns-train", lerf,
        "--experiment-name", scene_name,
        "--timestamp", scene_name,
        "--output-dir", f"{workdir}/outputs/",
        "--viewer.quit-on-train-completion", "True",
        "--data", DATA_NERF_PATH,
    ])
    CONSOLE.print(f"[green]✓ {lerf} Training Complete[/]")

    CONFIG_PATH = f"{workdir}/outputs/{scene_name}/{lerf}/{scene_name}/config.yml"
    CONSOLE.print("[cyan]→[/] Exporting NeRF pointcloud and .h5 CLIP embeddings...")
    subprocess.run([
        "ns-export", "pointcloud",
        "--num-points", "50000",
        "--remove-outliers", "True",
        "--normal-method", "open3d",
        "--reorient-normals", "True",
        "--save-world-frame", "False",
        "--load-config", CONFIG_PATH,
        "--output-dir", DATA_NERF_PATH,
        "--obb-center", "0.0000000000", "0.0000000000", "0.0000000000",
        "--obb-rotation", "0.0000000000", "0.0000000000", "0.0000000000",
        "--obb-scale", "5.0000000000", "5.0000000000", "5.0000000000",
    ])
    CONSOLE.print("[green]✓ Export Complete[/]")

if __name__ == "__main__":

    for scene_name in replica.scenes:
        run_training(scene_name)