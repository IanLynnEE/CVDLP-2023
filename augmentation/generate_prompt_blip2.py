"""Generate prompt by Blip2 for selected images.

For advanced generative models, we need to provide a prompt to generate
high-quality images. Most of the object detection datasets do not provide
captions for images. Therefore, we need to generate prompts for images first.

In order to generate high-quality images, we need to select images with
only one category of bboxes and less than 7 bboxes per image. Then we can
aggregate bboxes image by image and generate prompts for them.

The generated prompts might not be perfect. Here, we provide a sample of
post processing. However, it's better to check the generated prompts manually.
"""

import argparse
import os
import json

import torch
import pandas as pd
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Salesforce/blip2-opt-6.7b')
    parser.add_argument('--precision', type=str, choices=['float16', 'float32'], default='float16')
    parser.add_argument('--data_root', type=str, default='dataset/train')
    parser.add_argument('--annotation_path', type=str, default='dataset/annotations/train.json')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()
    args.precision: torch.dtype = getattr(torch, args.precision)
    os.makedirs(args.output_dir, exist_ok=True)
    return args


@torch.no_grad()
def main():
    args = parse_args()
    quantized = True if torch.cuda.get_device_properties(device).total_memory < 16e9 else False

    df = prepare_annotation(select_images(args.annotation_path))

    processor = Blip2Processor.from_pretrained(args.model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=args.precision,
        load_in_8bit=quantized,
    )
    if not quantized:
        model = model.to(device)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        img = Image.open(os.path.join(args.data_root, row['file_name']))

        # Blip2ForConditionalGeneration will not take prompt.
        inputs = processor(img, return_tensors='pt').to(device, dtype=args.precision)
        generated_ids = model.generate(**inputs, max_new_tokens=30)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        df.loc[i, 'generated_text'] = generated_text

    df_post = post_process(df)
    df_post.to_json(os.path.join(args.output_dir, 'blip2_post.json'), orient='records', indent=4)

    df_post.to_csv(os.path.join(args.output_dir, 'blip2_post.csv'), index=False)
    return


def select_images(annotation_path: str):
    """Select images with only one category of bboxes and less than 7 bboxes per image.

    Args:
        annotation_path (str): path to the annotation file

    Returns:
        DataFrame: annotation dataframe with information including selected images and corresponding bboxes
    """
    with open(annotation_path, 'r') as f:
        ann_json = json.load(f)

    # For `df_images`, the index actually means `image_id`. Set it as index for later joining.
    df_images = pd.DataFrame.from_records(ann_json['images'], index='id')
    df_ann = pd.DataFrame.from_records(ann_json['annotations'])

    # Join two dataframes by the `image_id` in `df_ann`.
    df = df_ann.join(df_images, on='image_id', how='left')

    # Select images with only one category of bboxes.
    df_single_category = df.groupby('image_id').filter(
        lambda x: len(x['category_id'].unique()) == 1
    )

    # Select images with less than 7 bboxes except for `category_id == 2`.
    df_selected = df_single_category.groupby('image_id').filter(
        lambda x: len(x) < 7 or 2 in x['category_id'].unique()
    )

    # Map category_id to category name.
    df_selected['label'] = df_selected['category_id'].map({
        cat['id']: cat['name'] for cat in ann_json['categories']
    })
    return df_selected.reset_index(drop=True)


def prepare_annotation(df_selected: pd.DataFrame):
    """Prepare an annotation with bboxes normalized and aggregated image by image.

    Args:
        df_input (pd.DataFrame): annotation dataframe with information including images and bboxes

    Returns:
        pd.DataFrame: annotation dataframe in the above format
    """
    df = df_selected[['image_id', 'file_name', 'category_id', 'label', 'height', 'width', 'bbox']].copy()

    # Normalize bbox coordinates and convert into the format of [x1, y1, x2, y2].
    df['bbox'] = df.apply(
        lambda x: [
            x['bbox'][0] / x['width'],
            x['bbox'][1] / x['height'],
            (x['bbox'][0] + x['bbox'][2]) / x['width'],
            (x['bbox'][1] + x['bbox'][3]) / x['height'],
        ],
        axis=1
    )

    # Since the image should only contain one category of bboxes, we can aggregate them image by image.
    df = df.groupby('image_id').agg({
        k: 'first' if k != 'bbox' else lambda x: list(x) for k in df.columns
    })
    df['generated_text'] = ''
    df['prompt'] = ''
    return df.reset_index(drop=True)


def post_process(df: pd.DataFrame):
    """Post process the generated text. Replace text, remove rows, etc.

    Args:
        df (pd.DataFrame): annotation dataframe with generated text

    Returns:
        pd.DataFrame: annotation dataframe with post processed generated text
    """
    # Replace 'sting ray' with 'stingray'.
    df['generated_text'] = df['generated_text'].str.replace('sting ray', 'stingray')

    # For label 'puffin', if 'bird', 'pengiun' or 'seal' is in the generated text, replace it with 'puffin'.
    df['generated_text'] = df.apply(
        lambda x: x['generated_text'].replace('bird', 'puffin').replace('penguin', 'puffin').replace('seal', 'puffin')
        if x['label'] == 'puffin' else x['generated_text'],
        axis=1
    )

    # If the label is not in the generated text, but 'fish' is in it, replace 'fish' with the label.
    df['generated_text'] = df['generated_text'].str.replace('fish tank', 'special_text_for_later_processing')
    df['generated_text'] = df.apply(
        lambda x: x['generated_text'].replace('fish', x['label'])
        if x['label'] not in x['generated_text'] and 'fish' in x['generated_text']
        else x['generated_text'],
        axis=1
    )
    df['generated_text'] = df['generated_text'].str.replace('tank', 'fish tank')
    df['generated_text'] = df['generated_text'].str.replace('special_text_for_later_processing', 'fish tank')

    # Remove r'a (?:man|person) (?:is )?\w*ing (?:in|at) '.
    regex_str = r'a (?:man|person) (?:is )?\w*ing (?:in|at) '
    df['generated_text'] = df['generated_text'].str.replace(regex_str, '', regex=True)

    # Remove 'a group of' and 'many'.
    df['generated_text'] = df['generated_text'].str.replace('a group of ', '')
    df['generated_text'] = df['generated_text'].str.replace('many ', '')

    # Remove 'a ' in the beginning.
    df['generated_text'] = df['generated_text'].str.replace(r'^a ', '', regex=True)

    # Delete row if the label is not in the generated text.
    df['label_in_text'] = df.apply(lambda x: x['label'] in x['generated_text'], axis=1)
    df = df[df['label_in_text']].drop(columns=['label_in_text'])

    # Place this as `prompt`.
    df['prompt'] = df.apply(
        lambda x:
        f'{x["generated_text"]}, {x["label"]}, marine, ocean, HD quality, high detail, '
        f'height: {x["height"]}, width: {x["width"]}',
        axis=1
    )
    return df


if __name__ == '__main__':
    main()
