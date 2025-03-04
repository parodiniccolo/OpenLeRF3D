import os
import tqdm
import cv2
import glob
import argparse
from natsort import natsorted
from third_party.SAI3D.helpers.sam_utils import get_samhq_by_iou, get_sam_by_area, num_to_natural, viz_mask, my_prepare_image, my_prepare_image_hq
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def seg_scannet(base_dir, view_freq, downsample_factor, scene_id=None):
    # Load SAM model
    sam_model_type = 'vit_h'
    sam_ckpt = '/media/dc-04-vol03/Niccolo/openlerf3d/third_party/SAI3D/SAM/checkpoints/sam_vit_h_4b8939.pth'
    sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt)
    device = "cuda"
    sam.to(device=device)

    # Create mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)

    os.makedirs(os.path.join(base_dir, '2D_masks'), exist_ok=True)
    color_base = os.path.join(base_dir, "images")
    color_paths = natsorted(glob.glob(os.path.join(color_base, '*.jpg')))
    for color_path in tqdm.tqdm(color_paths):
        color_name = os.path.basename(color_path)
        num = int(color_name[-9:-4])
        if num % view_freq != 0:
            continue
        original_image, input_image = my_prepare_image(image_pth=color_path, downsample_factor=downsample_factor)
        labels = get_samhq_by_iou(original_image, mask_generator)
        # labels = get_sam_by_area(input_image,mask_generator)
        color_mask = viz_mask(labels)
        labels = num_to_natural(labels) + 1  # 0 is background

        save_path = os.path.join(base_dir, '2D_masks')
        if (not os.path.exists(save_path)):
            os.mkdir(save_path)
        save_path = os.path.join(save_path, 'sam')
        if (not os.path.exists(save_path)):
            os.mkdir(save_path)
        # cv2.imwrite(os.path.join(save_path,color_name),original_image)
        cv2.imwrite(os.path.join(
            save_path, f'maskcolor_{color_name[:-4]}.png'), color_mask)
        cv2.imwrite(os.path.join(
            save_path, f'maskraw_{color_name[:-4]}.png'), labels)

def prepare_2d_mask(data_dir, view_freq, scene_id, downsample_factor=1):
    if view_freq is None:
        view_freq = 1
    seg_scannet(data_dir, view_freq, downsample_factor, scene_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--view_freq', type=int, default=5,
                        help='sample freuqncy for views')
    parser.add_argument('--downsample_factor', type=int, default=1,
                        help='Downsample factor for image preparation')
    args = parser.parse_args()

    seg_scannet(base_dir=args.data_dir, view_freq=args.view_freq, downsample_factor=args.downsample_factor)