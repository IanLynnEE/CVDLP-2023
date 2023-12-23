# Data Augmentation for Object Detection by BLIP-2 and GLIGEN

This is a guide for using generative models to augment the dataset. The code is based on [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) and [GLIGEN](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/gligen).

## Installation

While installing dependencies within the same virtual environment as the main project is possible, it is recommended to install in a new virtual environment. Python >= 3.11 and PyTorch >= 2.1.2 are recommended to extract the potential of your hardware. Older versions may also work, but the VRAM usage might be significantly higher.

**Step 0.** Create and activate a virtual environment (optional). Python 3.11 is recommended.

**Step 1.** Install PyTorch and torchvision. Please install PyTorch v2.1.2 though the official [installation guide](https://pytorch.org/get-started).

**Step 2.** Install Diffusers by HuggingFace.

```shell
pip install "diffusers[torch]" Transformers
```

**Step 3.** Install Pandas.

```shell
pip install pandas
```

**Step 4.** Install PyTorch-FID (optional).

```shell
pip install pytorch-fid
```

If the [ValueError](https://github.com/mseitzer/pytorch-fid/issues/103) has not been fixed, please install SciPy v1.11.1.

```shell
pip install scipy==1.11.1
```



## Generate Images

Diffusion-based generative models usually require a prompt to generate images. We will first select some images from the dataset, and then generate captions for them. The captions (with the images) will be sent to the generative model.

### Select Images

In order to generate high-quality images, we need to select images that are suitable for the generative model. Some criteria are recommended for selecting images:

- The image should contain only a few objects (e.g. less than 7 objects).
- The objects should be in the same category (e.g. all dogs).

[An example of selecting images by script](./generate_prompt_blip2.py#select_images) is integrated in the following step, so you don't need to run it separately. However, you may need to modify it according to your needs.

### Generate Captions

We use [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) to generate captions for the images.

```shell
python generate_prompt_blip2.py \
    -i <path/to/images/directory> \
    -a <path/to/annotations.json> \
    -o <path/to/output/directory>
```

The generated captions may contain some errors, so the post-processing step is required. [An example of post-processing](./generate_prompt_blip2.py#post_process) is integrated in the script above, you may need to modify it according to your needs.

The generated captions (after post-processing) will be saved in `<path/to/output/directory>/blip2_post.json`.

### Generate Images

We use [GLIGEN](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/gligen) to generate images from the captions.

```shell
python generate_image_gligen.py \
    --mode <generation|inpainting>
    -i <path/to/images/directory> \
    -p <path/to/captions.json> \
    -o <path/to/output/directory> \
```

The generated images will be saved in `<path/to/output/directory>/<mode>/`.

### Manually Select Images

Not all generated images are suitable for augmentation. You can manually select images from the generated images based on the quality of the images and the imbalance ratio of the dataset. Please not change the file names of the selected images if you want to use the annotations generated in the next step.

A scrpit for drawing bounding boxes on the generated images for better selection:

```shell
python draw_bboxes.py \
    -g <path/to/generated/images/directory> \
    -p <path/to/captions.json> \
    -o <path/to/output/directory>
```

### Finalization

To train/fine-tune the object detection model, the annotations for the selected images are required. The annotations should be the same as the corresponding original images, so we need to parse the annotations of the original images and select the annotations for the selected images.

If you keep the file names of the selected images unchanged (disregarding if them contain bounding boxes), you can use the following script to generate the annotations for training/fine-tuning.

```shell
python finalize_selected_images.py \
    --original_annotation <path/to/original/annotations.json> \
    --original_data_root <path/to/original/images/directory> \
    -g <path/to/generated/images/directory> \
    -i <path/to/selected/images/directory> \
    -o <path/to/output/directory>
```

The generated images, the corresponding original images, and the annotations of them will be saved in `<path/to/output/directory>`.



## Fine-tune Object Detection Model

The generated images together with the corresponding original images usually are not enough for training the model for a new domain from a pretrained model. We recommend to load a model that had been trained your custom dataset, and then further fine-tune it with the generated images, which could be a balanced dataset if you select images carefully. The idea is similar to [Dynamic Curriculum Learning for Imbalanced Data Classification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Dynamic_Curriculum_Learning_for_Imbalanced_Data_Classification_ICCV_2019_paper.pdf)
