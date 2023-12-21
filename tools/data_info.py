"""Get the information of the dataset.

This script is used to get the information of the dataset, including the
number of images, the number of categories, the number of images in each
category, image sizes, statistics of images, etc.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Get the information of the '
                                     'dataset.')
    parser.add_argument('--data_root', default='dataset/train')
    parser.add_argument('--annotation_path',
                        default='dataset/annotations/train.json')
    parser.add_argument('--img_info', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.annotation_path, 'r') as f:
        ann_json = json.load(f)

    # Create dataframes.
    df_ann = pd.DataFrame.from_records(ann_json['annotations'])
    df_ann.set_index('id', inplace=True)                            # bbox_id

    # Get the number of images per category.
    df_category = df_ann.groupby('category_id').count()
    df_category = df_category[['image_id']]
    df_category.columns = ['num_images']
    df_category.to_csv('dataset/category2.csv')

    # Get sizes of bboxes.
    df_bbox_info = df_ann[['bbox']].copy()
    df_bbox_info['width'] = df_bbox_info['bbox'].apply(lambda x: x[2])
    df_bbox_info['height'] = df_bbox_info['bbox'].apply(lambda x: x[3])
    df_bbox_info = df_bbox_info[['width', 'height']]
    df_bbox_info.describe().to_csv('dataset/bbox_info_2.csv')

    if args.img_info:
        df_image_info = get_image_info(args.data_root, ann_json)
        df_image_info.describe().to_csv('dataset/image_info.csv')
    return


def get_image_info(data_root: str, ann: dict):
    """Get the information of the images, including
    (width, height, mean_r, mean_g, mean_b, std_r, std_g, std_b).

    Args:
        data_root (str): path to the dataset
        ann (dict): dict of the annotation file
    """
    image_info: list[dict[str, float]] = []
    for image in tqdm(ann['images']):
        image_path = os.path.join(data_root, image['file_name'])
        with Image.open(image_path) as img:
            width, height = img.size
            img = np.array(img)
            mean_r, mean_g, mean_b = img.mean(axis=(0, 1))
            std_r, std_g, std_b = img.std(axis=(0, 1))
        image_info.append({
            'width': width,
            'height': height,
            'mean_r': mean_r,
            'mean_g': mean_g,
            'mean_b': mean_b,
            'std_r': std_r,
            'std_g': std_g,
            'std_b': std_b
        })
    return pd.DataFrame.from_records(image_info)


if __name__ == '__main__':
    main()
