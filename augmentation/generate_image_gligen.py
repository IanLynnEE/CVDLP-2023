"""Generate synthetic images by GLIGEN.

GLIGEN is a generative model that can generate images from text and bounding
boxes. We can use it to generate synthetic images for object detection. The
generated images can be used for data augmentation during training.

Two modes are supported: inpainting and generation. In inpainting mode, the
height and width of the generated images cannot be specified directly.
We resize the generated images to the original size of the input image instead.
"""


import argparse
import json
import os
from warnings import filterwarnings

import torch
from diffusers import StableDiffusionGLIGENPipeline
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, choices=['inpainting', 'generation'], default='generation')
    parser.add_argument('--data_root', type=str, default='dataset/train')
    parser.add_argument('--annotation_path', type=str, default='outputs/blip2_post.json')
    parser.add_argument('-o', '--output_root', type=str, default='outputs')
    args = parser.parse_args()

    args.model_name = f'masterful/gligen-1-4-{args.mode}-text-box'
    args.output_dir = os.path.join(args.output_root, args.mode)
    os.makedirs(args.output_dir, exist_ok=True)
    return args


@torch.no_grad()
def main():
    args = parse_args()
    filterwarnings('ignore', category=FutureWarning)
    inpainting = True if args.mode == 'inpainting' else False

    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        args.model_name,
        variant='fp16',
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    with open(args.annotation_path, 'r') as f:
        ann_list = json.load(f)

    for _, row in tqdm(enumerate(ann_list), total=len(ann_list)):
        img = Image.open(os.path.join(args.data_root, row['file_name'])).convert('RGB') if inpainting else None

        generated_images = pipe(
            prompt=row['generated_text'],
            height=row['height'] if not inpainting else None,
            width=row['width'] if not inpainting else None,
            gligen_phrases=['one' + row['label'] for _ in row['bbox']],
            gligen_boxes=row['bbox'],
            gligen_inpaint_image=img,
            gligen_scheduled_sampling_beta=1,
            output_type='pil',
            num_inference_steps=50,
        ).images

        # Resize to the original size.
        if inpainting:
            generated_images[0] = generated_images[0].resize((row['width'], row['height']))

        generated_images[0].save(os.path.join(
            args.output_dir,
            f'{row["category_id"]}_{row["image_id"]:04d}.png'
        ))


if __name__ == '__main__':
    main()
