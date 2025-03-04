import os
import tqdm
import cv2
import glob
import argparse
from natsort import natsorted
from third_party.SAI3D.helpers.sam_utils import get_samhq_by_iou, get_sam_by_area, num_to_natural, viz_mask, my_prepare_image, my_prepare_image_hq
from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator

def seg_scannet(base_dir, view_freq, scene_id=None):
    if scene_id:
        seg_split = [scene_id]
    else:
        with open(os.path.join(base_dir, 'Tasks/Benchmark/scannetv2_val.txt'), 'r') as f:
            val_split = f.readlines()
        val_split = [s.strip() for s in val_split]
        seg_split = sorted(val_split)

    all_color_base = os.path.join(base_dir, 'posed_images')

    level = [3,]  # instance level


    model_type='vit_h' 
    ckpt='/media/dc-04-vol03/Niccolo/econom/third_party/SAI3D/SAM-HQ/checkpoints/sam_hq_vit_h.pth'
    
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device=device)

    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam,
    #     #points_per_side=32,
    #     #pred_iou_thresh=0.8,
    #     #stability_score_thresh=0.9,
    #     #crop_n_layers=1,
    #     #crop_n_points_downscale_factor=2,
    #     #min_mask_region_area=10,  # Requires open-cv to run post-processing
    # )


    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24,          # Slightly reduced from default 32 to get less granular segments
        pred_iou_thresh=0.9,         # Increased from 0.8 for more precision
        stability_score_thresh=0.95,  # Increased from 0.9 for more stable masks
        min_mask_region_area=100,    # Increased to remove smaller details
        crop_n_layers=0,             # Reduced to avoid capturing fine details at different scales
    )

    """
    mask_generator = SemanticSamAutomaticMaskGenerator(
        sam_model,
        level=level)  # model_type: 'L' / 'T', depends on your checkpoint
    """
    os.makedirs(os.path.join(base_dir, '2D_masks'), exist_ok=True)
    for scene_id in tqdm.tqdm(seg_split):
        color_base = os.path.join(all_color_base, scene_id)
        color_paths = natsorted(glob.glob(os.path.join(color_base, '*.jpg')))
        for color_path in tqdm.tqdm(color_paths, desc=scene_id):
            color_name = os.path.basename(color_path)
            num = int(color_name[-9:-4])
            if num % view_freq != 0:
                continue
            original_image, input_image = my_prepare_image(image_pth=color_path)

            labels = get_samhq_by_iou(original_image, mask_generator)
            # labels = get_sam_by_area(input_image,mask_generator)
            color_mask = viz_mask(labels)
            labels = num_to_natural(labels) + 1  # 0 is background

            save_path = os.path.join(base_dir, '2D_masks', scene_id)
            if (not os.path.exists(save_path)):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, 'sam-hq')
            if (not os.path.exists(save_path)):
                os.mkdir(save_path)
            # cv2.imwrite(os.path.join(save_path,color_name),original_image)
            cv2.imwrite(os.path.join(
                save_path, f'maskcolor_{color_name[:-4]}.png'), color_mask)
            cv2.imwrite(os.path.join(
                save_path, f'maskraw_{color_name[:-4]}.png'), labels)

def prepare_2d_mask(data_dir, view_freq, scene_id):
    if view_freq is None:
        view_freq = 1
    seg_scannet(data_dir, view_freq, scene_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--view_freq', type=int, default=5,
                        help='sample freuqncy for views')
    args = parser.parse_args()

    seg_scannet(base_dir=args.data_dir, view_freq=args.view_freq)
