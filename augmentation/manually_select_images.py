import argparse
import json
import os

import pandas as pd
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image, write_png


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, default='outputs/blip2_post.json')
    parser.add_argument('--generated_data_root', type=str, default='outputs/generation')
    parser.add_argument('--output_bbox_root', type=str, default='outputs/bbox')
    args = parser.parse_args()
    os.makedirs(args.output_bbox_root, exist_ok=True)
    return args


def main():
    args = parse_args()
    # DEBUG
    original_data_root = 'dataset/train'
    copy_original_images_dir = 'outputs/original'
    os.makedirs(copy_original_images_dir, exist_ok=True)

    with open(args.annotation_path, 'r') as f:
        ann_list = json.load(f)

    for _, row in enumerate(ann_list):
        # DEBUG
        copied_file_name = os.path.join(copy_original_images_dir, f'{row["category_id"]}_{row["image_id"]:04d}.png')
        os.system(f'cp {os.path.join(original_data_root, row["file_name"])} {copied_file_name}')

        generated_img = read_image(os.path.join(
            args.generated_data_root,
            f'{row["category_id"]}_{row["image_id"]:04d}.png'
        ))
        _, h, w = generated_img.shape
        upscaled_bboxes = [[bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h] for bbox in row['bbox']]
        draw = draw_bounding_boxes(
            generated_img,
            torch.tensor(upscaled_bboxes),
            labels=[row['label']] * len(upscaled_bboxes),
            width=5
        )
        write_png(draw, os.path.join(
            args.output_bbox_root,
            f'{row["category_id"]}_{row["image_id"]:04d}.png'
        ))


def copy_selected_images_and_annotations(df_selected: pd.DataFrame, args: argparse.Namespace):
    """Copy selected images and annotations to the output directory.

    Args:
        df_selected (pd.DataFrame):
            annotation dataframe with information including selected images and corresponding bboxes
        args (argparse.Namespace): arguments
    """
    output_dir = os.path.join(args.output_dir, 'selected')
    os.makedirs(output_dir, exist_ok=True)

    # Copy selected images.
    for img_path in df_selected['file_name'].unique():
        os.system(f'cp {os.path.join(args.data_root, img_path)} {output_dir}')

    with open(args.annotation_path, 'r') as f:
        ann_json = json.load(f)
    selected_ann = {
        'categories': ann_json['categories'],
        'images': [img for img in ann_json['images'] if img['id'] in df_selected['image_id'].unique()],
        'annotations': [ann for ann in ann_json['annotations'] if ann['image_id'] in df_selected['image_id'].unique()],
    }
    with open(os.path.join(output_dir, 'selected.json'), 'w') as f:
        json.dump(selected_ann, f, indent=4)


if __name__ == '__main__':
    main()
