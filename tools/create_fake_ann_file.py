"""Create a fake annotation file.

MMDetection requires the annotation file even for test dataset.
Moreover, we need to convert the prediction file from coco format to a specific
format for this howework. Thus, creating a fake annotation file is very useful.

Usage:
    python create_fake_ann_file.py <img_dir> <annotation_dir>
"""

import json
import os
import sys

from PIL import Image


def main():
    assert len(sys.argv) == 3
    assert os.path.isdir(sys.argv[1])
    assert os.path.isdir(sys.argv[2])
    assert os.path.isfile(os.path.join(sys.argv[2], 'train.json'))

    # Get classes from annotations in annotation_dir.
    with open(os.path.join(sys.argv[2], 'train.json')) as f:
        train_ann = json.load(f)

    _create_fake_annotation(sys.argv[1], sys.argv[2], train_ann['categories'])


def _create_fake_annotation(img_dir: str, annotation_dir: str, classes: list):
    images = []
    for id, file_name in enumerate(os.listdir(img_dir)):
        if not file_name.endswith('.jpg'):
            continue
        # This is slow, but whatever.
        image = Image.open(os.path.join(img_dir, file_name))
        images.append({
            'id': id,
            'file_name': file_name,
            'width': image.width,
            'height': image.height,
        })

    out = os.path.join(annotation_dir, os.path.basename(img_dir) + '.json')
    with open(out, 'w') as f:
        json.dump({
            'categories': classes,
            'images': images,
        }, f, indent=4)


if __name__ == '__main__':
    main()
