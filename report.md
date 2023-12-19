# Homework 1 for CVDLP 2023 Spring

[Co-DETR](https://github.com/Sense-X/Co-DETR) was used for this homework. After fine-tuning on the custom dataset, the model achieved 0.54 mAP on the validation set.



## Model Architecture

[Co-DETR](https://github.com/Sense-X/Co-DETR) is a Transformer-based object detection model, a modified version of [DETR](https://github.com/facebookresearch/detr). The authors proposed a new training scheme to improve the performance. The main idea is to use multiple auxiliary heads to supervise the encoder. In addition, the authors proposed to extract the positive coordinates from these auxiliary heads to improve the training efficiency of positive samples in the decoder. In inference, these auxiliary heads are discarded and thus the method introduces no additional parameters and computational cost to the original detector. The model architecture is shown below.

![Co-DETR](https://raw.githubusercontent.com/Sense-X/Co-DETR/main/figures/framework.png)



## Implementation Details

Limited by the computational resources, I only trained the model with ResNet-50 backbone. Training on NVIDIA RTX 3060 GPU also limits the availible augmentation methods. A short summary of the training details is shown below.

- Backbone: ResNet-50 with pretrained weights on ImageNet
- Batch size: 1
- Learning rate: 2e-4
- Optimizer: AdamW with weight decay 1e-4
- Scheduler: MultiStepLR with milestones [8, 16, 32] and gamma 0.1
- Training epochs: 48
- Augmentation: Following the original implementation of DETR, only different scales are used.

### Backbone

The weights of the backbone are from [torchvision](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html), and the learning rates for the backbone are multiplied by 0.1 except for the batch normalization layers, the first stage, and the stem stage, which were frozen during training. All stages are used as output features.

### Pre-trained Weights

The pre-trained weights can be downloaded from the official release of [MMDetection](https://github.com/open-mmlab/mmdetection/blob/v3.2.0/projects/CO-DETR/README.md). The weights are trained on COCO dataset.

### Loss Function

With the loss function of the $l$-th decoder layer of the original DETR denoted as $\tilde{\mathcal{L}}_l^{dec}$, the loss function of Co-DETR is:

$$
\mathcal{L}^{global} = \sum_{l=1}^L \left(
    \tilde{\mathcal{L}}_l^{dec}
    + \lambda_1 \sum_{i=1}^K \mathcal{L}_{i, l}^{dec}
    + \lambda_2 \sum_{i=1}^K \mathcal{L}_i^{enc}
\right),
$$

where the loss of the $l$-th decoder layer in the $i$-th auxiliary branch is:

$$
\mathcal{L}_{i, l}^{dec} = \tilde{\mathcal{L}} \left(
    \tilde{P}_{i, l},\ P_i^{pos}
\right),
$$

and the loss of the $i$-th auxiliary branch in the encoder is:

$$
\mathcal{L}_i^{enc} = \mathcal{L}_i \left(
    \hat{P}_i^{pos},\ P_i^{pos}
\right) + \mathcal{L}_i \left(
    \hat{P}_i^{neg},\ P_i^{neg}
\right).
$$

$\lambda_1$ and $\lambda_2$ are set to 1.0 and 2.0, respectively. For more details, please refer to the original paper.

### Augmentation

Noticing that the image size of the custom dataset is much larger than that of COCO, I tried different augmentation methods. With the limited computational resources, the best augmentation method I found is not reducing the resolution of images in resizing. The augmentation includes:

- Random horizontal flip with probability 0.5
- Random choice of the following methods:
  - Randomly resize to (576, 608, 640, 672, 704, 736, 768, 800, 832) with the shorter side
  - Randomly crop to (384, 600) with the shorter side, and Randomly resize to (576, 608, 640, 672, 704, 736, 768, 800, 832) with the shorter side.

By enlarging the image size, slightly better performance can be achieved. This might come from the nature of the Transformer-based models, which are superior in detecting large objects. However, I run out of VRAM when the shorter side is larger than 832.

I've also tried to add vertical flip, as the dataset contains signicant amount of images taken underwater. An intuitive idea is that those images can be flipped vertically since the orientation of the objects are not important. However, the performance is worse than that without vertical flip.

<div style="page-break-after: always;"></div>

## Results on Validation Set

| Model   | mAP    | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|---------|--------|-----------------|-----------------|----------------|----------------|----------------|
| Co-DETR | 0.5418 | 0.8150          | 0.5830          | 0.2374         | 0.4287         | 0.6743         |

I find that the performance is unstable and may fluctuate by about 0.01 mAP. Additionally, training with more epochs might improve the performance, particularly the AP<sub>S</sub> and AP<sub>M</sub>. However, since mAP is the metric used in the competition, I picked the model stopping at 33 epochs, which has the highest mAP on the validation set, as the final model.



## Results Visualization

![IMG_8404_predict](./outputs/vis/IMG_8404_jpg.rf.265b89e862a375f6b89f781ea60ed480.jpg)

<div style="page-break-after: always;"></div>

## Other Attempts

The authors of [Conditional DETR](https://github.com/Atten4Vis/ConditionalDETR) proposed to use a conditional transformer to improve the performance of DETR. The idea is to make the training focus on the extremity areas of the objects. This method do make the model converge faster, and the main idea is not conflicting with Co-DETR. It will be interesting to combine these two methods in the future.

In conditional DETR, the authors also revealed that the performance is sensitive to the backbone. The official site of the Co-DETR also supports this claim. The authors of Co-DETR did release pretrained models with Swin-L backbone, but I cannot run the model on my GPU. In hope of using other backbones, I first tried to train the conditional DETR model from scratch. If the model can converge, other backbones will be options for Co-DETR. Sadly, the limited number of training samlpes makes the model not converge. Thus, while there might be chances to reach 0.6 mAP by using a better backbone for Co-DETR, more computational resources are needed to utilize the pretrained weights.

Since the conditional DETR is much faster to train, lots of experiments were done on it, including different learning rates, different augmentation methods, and different backbones. Configuration examples are in the [configs](./configs) folder.