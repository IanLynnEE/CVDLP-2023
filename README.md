# Fine-tune Co-DETR on a custom dataset

This project shows how to fine-tune a pretrained [Co-DETR](https://github.com/Sense-X/Co-DETR) model on a custom dataset.

## Installation

### Install MMDetection

This project is based on [MMDetection v3.2.0](https://github.com/open-mmlab/mmdetection/releases/tag/v3.2.0). Please follow the official [installation guide](https://mmdetection.readthedocs.io/en/v3.2.0/get_started.html). For example, you can install it via pip:

**Step 0.** Create and activate a virtual environment (optional). Python 3.8 is recommended.

**Step 1.** Install PyTorch and torchvision. Please install PyTorch v1.13.1 though the official [installation guide](https://pytorch.org/get-started/previous-versions/).

**Step 2.** Install MMDetection.

```shell
pip install -U openmim
mim install "mmengine==0.9.0"
mim install "mmcv==2.0.1"
mim install "mmdet==3.2.0"
```

Note: MMDetection requires [MMCV](https://github.com/open-mmlab/mmcv), which is only compiled on PyTorch 1.x.0. Please make sure that the PyTorch version matches the requirements of MMCV.

**Step 3.** Install [FairScale](https://github.com/facebookresearch/fairscale) and TorchMetrics.

```shell
pip install fairscale
pip install torchmetrics
```

**Step 4.** Add the current project path to the PYTHONPATH.

```shell
export PYTHONPATH=<path_to_project>:$PYTHONPATH
```

### Install Diffusers (Optional)

If the imbalance of the dataset is severe, a guideline for using generative models to augment the dataset is provided in [Augmentation]('./augmentation/README.md).



## Prepare a custom dataset

**Step 1.** Prepare the dataset in the following manner:

```
dataset
├── annotations
│   ├── train.json
│   └── val.json
├── train
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── val
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── test
    ├── 000000.jpg
    ├── 000001.jpg
    └── ...
```

The annotation files are expected to be in the COCO format.

**Step 3.** Create a fake annotation file for the test set.

```shell
python tools/create_fake_ann_file.py dataset/test dataset/annotations/
```

**Step 4.** Modify `classes` in `configs/codino/co_dino_5scale_r50_1xb1_48e.py` to match the classes in your dataset.



## Experiments

All experiments are conducted on a single NVIDIA GeForce RTX 3060 GPU except for the augmentation part, which is conducted on a single NVIDIA GeForce RTX 4080 GPU.

### Fine-tune the model

If the dataset is filed in the above manner, you can fine-tune the model by running the following command:

```shell
python tools/train.py configs/codino/co_dino_5scale_r50_1xb1_48e.py
```

Checkpoints and logs will be saved in the `work_dirs/` folder.

### Inference

To get the prediction results on the test set, run the following command:

```shell
python tools/test.py configs/codino/co_dino_5scale_r50_1xb1_48e.py <path_to_checkpoint>
```

The prediction results will be saved in the `work_dirs/test.bbox.json` file, and it's in the COCO submission format. To convert it to the required format for `evaluate.py`, run the following command:

```shell
python tools/convert_results.py work_dirs/test.bbox.json dataset/annotations/test.json
```

The converted results will be saved in the `work_dirs/test.bbox.converted.json` file.

Note: Validation results are monitored during training. If independent validation is required, please check `tools/validate.sh`.

### Visualize the results

To visualize the results, run the following command:

```shell
python tools/demo.py <image_path> configs/codino/co_dino_5scale_r50_1xb1_48e.py --weights <path_to_checkpoint>
```

Results will be saved under `outputs/`.


### Further fine-tune the model by augmented data

Please refer to [Augmentation]('./augmentation/README.md') for details.


## Acknowledgement

All credits go to the authors of [Co-DETR](https://github.com/Sense-X/Co-DETR) and [OpenMMLab](https://openmmlab.com).