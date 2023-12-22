import argparse
import json
import os
import shutil
from copy import deepcopy

from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_annotation', type=str, default='dataset/annotations/train.json')
    parser.add_argument('--origin_data_root', type=str, default='dataset/train')
    parser.add_argument('-g', '--generated_data_root', type=str, required=True)
    parser.add_argument('-i', '--selected_data_root', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    gen_ann, ori_ann = get_annotation_of_selected_images(args.origin_annotation, args.selected_data_root)
    os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'selected_ann.json'), 'w') as f:
        json.dump(gen_ann, f, indent=4)
    with open(os.path.join(args.output_dir, 'original_selected_ann.json'), 'w') as f:
        json.dump(ori_ann, f, indent=4)

    for row in gen_ann['images']:
        shutil.copy(
            os.path.join(args.generated_data_root, row['file_name']),
            os.path.join(args.output_dir, row['file_name'])
        )
        shutil.copy(
            os.path.join(args.origin_data_root, row['o_file_name']),
            os.path.join(args.output_dir, row['o_file_name'])
        )
    get_fid(args, gen_ann)


def get_fid(args: argparse.Namespace, ann: dict):
    generated_resized_dir = os.path.join(args.output_dir, 'generated_resized')
    original_resized_dir = os.path.join(args.output_dir, 'original_resized')
    os.makedirs(generated_resized_dir)
    os.makedirs(original_resized_dir)

    # Resize image to 512x512.
    for row in tqdm(ann['images'], total=len(ann['images'])):
        img = Image.open(os.path.join(args.generated_data_root, row['file_name']))
        img = img.resize((512, 512))
        img.save(os.path.join(generated_resized_dir, row['file_name']))

        img = Image.open(os.path.join(args.origin_data_root, row['o_file_name']))
        img = img.resize((512, 512))
        img.save(os.path.join(original_resized_dir, row['o_file_name']))

    os.system(f'python -m pytorch_fid --device cuda:0 {generated_resized_dir} {original_resized_dir}')
    shutil.rmtree(generated_resized_dir)
    shutil.rmtree(original_resized_dir)


def get_annotation_of_selected_images(ori_ann: str, selected_dir: str):
    with open(ori_ann, 'r') as f:
        ann = json.load(f)

    img_id_list = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(selected_dir) if f.endswith('.png')]
    selected_ann = {
        'categories': ann['categories'],
        'images': [row for row in ann['images'] if row['id'] in img_id_list],
        'annotations': [row for row in ann['annotations'] if row['image_id'] in img_id_list],
    }
    original_selected_ann = deepcopy(selected_ann)
    img_id_to_cat_id = {row['image_id']: row['category_id'] for row in selected_ann['annotations']}

    for row in selected_ann['images']:
        row['o_file_name'] = row['file_name']
        row['file_name'] = f'{img_id_to_cat_id[row["id"]]}_{row["id"]:04d}.png'
    return selected_ann, original_selected_ann


if __name__ == '__main__':
    main()
