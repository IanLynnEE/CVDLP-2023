import argparse
import json
import os

import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image, write_png


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotation_path', type=str, default='outputs/blip2_post.json')
    parser.add_argument('-i', '--data_root', type=str, default='outputs/generation')
    parser.add_argument('-o', '--output', type=str, default='outputs/bbox')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args


def main():
    args = parse_args()

    with open(args.annotation_path, 'r') as f:
        ann_list = json.load(f)
    assert isinstance(ann_list, list), f'annotation file should be a list, but got {type(ann_list)}'

    for _, row in enumerate(ann_list):
        generated_img = read_image(os.path.join(
            args.data_root,
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
            args.output,
            f'{row["category_id"]}_{row["image_id"]:04d}.png'
        ))


if __name__ == '__main__':
    main()
