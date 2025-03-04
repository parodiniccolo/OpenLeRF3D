"""
Convert the replica dataset into nerfstudio format.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import json
import os
import pymeshlab
import cv2

import replica

from nerfstudio.process_data import process_data_utils
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

def process_depth_images(input_dir, output_dir):
    """
    Process all depth images in the input directory and save them to the output directory.
    
    Args:
        input_dir (str): Path to directory containing depth images
        output_dir (str): Path to directory where processed images will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all files in input directory
    files = os.listdir(input_dir)
    
    # Filter for common image extensions
    image_extensions = ('.png')
    image_files = [f for f in files if f.lower().endswith(image_extensions)]
    
    print(f"Found {len(image_files)} images to process")
    
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Read image as uint16
            depth_image = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH)
            
            if depth_image is None:
                print(f"Failed to read image: {filename}")
                continue
                
            # Apply the transformation
            processed_image = depth_image / (1000.0 * 6.58)
            
            # Save the processed image
            cv2.imwrite(output_path, processed_image)
            print(f"Processed and saved: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print("Processing Depth Maps complete!")


def process_replica(data: Path, output_dir: Path):
    """Process Replica data into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Record3D poses into the nerfstudio format.
    """

    # convert mesh to triangle mesh (open3d can only read triangle meshes)
    mesh_path = data / '..' / f'{data.name}_mesh.ply'
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    ms.apply_filter('meshing_poly_to_tri')
    os.makedirs(output_dir, exist_ok=True)
    ms.save_current_mesh(str(output_dir / mesh_path.name), save_vertex_normal=True)
    # Also save as OBJ with the same base name
    obj_path = str(output_dir / mesh_path.name.replace('.ply', '.obj'))
    ms.save_current_mesh(obj_path)  

    verbose = True
    num_downscales = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size = 400
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)

    summary_log = []

    replica_image_dir = data / "results"

    if not replica_image_dir.exists():
        raise ValueError(f"Image directory {replica_image_dir} doesn't exist")

    replica_image_filenames = []
    replica_depth_filenames = []
    for f in replica_image_dir.iterdir():
        if f.stem.startswith('frame'):  # removes possible duplicate images (for example, 123(3).jpg)
            if f.suffix.lower() in [".jpg"]:
                replica_image_filenames.append(f)
        if f.stem.startswith('depth'):  # removes possible duplicate images (for example, 123(3).jpg)
            if f.suffix.lower() in [".png"]:
                replica_depth_filenames.append(f)

    replica_image_filenames = sorted(replica_image_filenames)
    replica_depth_filenames = sorted(replica_depth_filenames)
    assert(len(replica_image_filenames) == len(replica_depth_filenames))
    num_images = len(replica_image_filenames)

    idx = np.arange(num_images)
    if max_dataset_size != -1 and num_images > max_dataset_size:
        idx = np.round(np.linspace(0, num_images - 1, max_dataset_size)).astype(int)

    replica_image_filenames = list(np.array(replica_image_filenames)[idx])
    replica_depth_filenames = list(np.array(replica_depth_filenames)[idx])

    # Copy images to output directory
    copied_image_paths = process_data_utils.copy_images_list(
        replica_image_filenames,
        image_dir=image_dir,
        verbose=verbose,
        num_downscales=num_downscales,
    )

    # Process and copy depth images
    # First, create a temporary directory for processed depth images
    temp_depth_dir = output_dir / "temp_depths"
    temp_depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Process the depth images
    process_depth_images(str(replica_image_dir), str(temp_depth_dir))
    
    # Now copy the processed depth images with downscaling
    processed_depth_files = [temp_depth_dir / f.name for f in replica_depth_filenames]
    copied_depth_paths = process_data_utils.copy_images_list(
        processed_depth_files,
        image_dir=depth_dir,
        verbose=verbose,
        num_downscales=num_downscales,
    )
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(str(temp_depth_dir))

    assert(len(copied_image_paths) == len(copied_depth_paths))
    num_frames = len(copied_image_paths)

    copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
    copied_depth_paths = [Path("depths/" + copied_depth_path.name) for copied_depth_path in copied_depth_paths]
    summary_log.append(f"Used {num_frames} images out of {num_images} total")
    if max_dataset_size > 0:
        summary_log.append(
            "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
            f"larger than the current value ({max_dataset_size}), or -1 to use all images."
        )

    traj_path = data / "traj.txt"
    replica_to_json(copied_image_paths, copied_depth_paths, traj_path, output_dir, indices=idx)


def replica_to_json(images_paths: List[Path], depth_paths: List[Path], trajectory_txt: Path, output_dir: Path, indices: np.ndarray) -> int:
    """Converts Replica's metadata and image paths to a JSON file.

    Args:
        images_paths: list if image paths.
        traj_path: Path to the Replica trajectory file.
        output_dir: Path to the output directory.
        indices: Indices to sample the metadata_path. Should be the same length as images_paths.

    Returns:
        The number of registered images.
    """

    assert len(images_paths) == len(indices)
    assert len(depth_paths) == len(indices)
    poses_data = process_txt(trajectory_txt)
    poses_data = np.array(
            [np.array(
                [float(v) for v in p.split()]).reshape((4, 4)) for p in poses_data]
        )

    # Set up rotation matrix
    rot_x = np.eye(4)
    a = np.pi
    rot_x[1, 1] = np.cos(a)
    rot_x[2, 2] = np.cos(a)
    rot_x[1, 2] = -np.sin(a)
    rot_x[2, 1] = np.sin(a)

    camera_to_worlds = poses_data[indices] @ rot_x

    frames = []
    for i, (im_path, depth_path) in enumerate(zip(images_paths, depth_paths)):
        c2w = camera_to_worlds[i]
        frame = {
            "file_path": im_path.as_posix(),
            "depth_file_path": depth_path.as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    with open(trajectory_txt.parents[1] / 'cam_params.json') as file:
        cam_params = json.load(file)

    # Camera intrinsics
    focal_length = cam_params['camera']['fx']
    h = cam_params['camera']['h']
    w = cam_params['camera']['w']
    cx, cy = w / 2.0, h / 2.0

    out = {
        "fl_x": focal_length,
        "fl_y": focal_length,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "camera_model": CAMERA_MODELS["perspective"].name,
    }

    out["frames"] = frames
    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    return len(frames)


if __name__ == "__main__":
    for scene in replica.scenes:
      data = f'data/Replica/{scene}'
      output_dir = f'data/replica_{scene}'
      process_replica(Path(data), Path(output_dir))
