"""Get the information of the dataset.

This script is used to get the information of the dataset, including the
number of images, the number of categories, the number of images in each
category, image sizes, statistics of images, etc.
"""

import sys
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def main():
    data_root: str = sys.argv[1]
    annotation_path: str = sys.argv[2]

    # Get classes.
    with open(annotation_path, 'r') as f:
        ann = json.load(f)
        categories = [category['name'] for category in ann['categories']]

    # Get the number of images per category.
    num_images_per_category = {i: 0 for i in range(len(categories))}
    for image in ann['annotations']:
        num_images_per_category[image['category_id']] += 1
    print('Number of images per category:', num_images_per_category)

    # Get bbox information, such as the number of bboxes per image,
    # the bbox sizes, etc.
    bbox_info: list[dict[str, float]] = []
    for bboxes in ann['annotations']:
        bbox_info.append({
            'width': bboxes['bbox'][2],
            'height': bboxes['bbox'][3]
        })
    bbox_info_df = pd.DataFrame.from_records(bbox_info)
    bbox_info_df.describe().to_csv('bbox_info.csv')

    # Get image information.
    # (width, height, mean_r, mean_g, mean_b, std_r, std_g, std_b)
    image_info: list[dict[str, float]] = []
    for image in tqdm(ann['images']):
        image_path = data_root + '/' + image['file_name']
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

    # Get statistics of images.
    image_info_df = pd.DataFrame.from_records(image_info)
    image_info_df.describe().to_csv('image_info.csv')


if __name__ == '__main__':
    main()
