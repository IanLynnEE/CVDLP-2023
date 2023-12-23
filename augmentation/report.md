# Homework 3 for CVDLP 2023 Spring

[Co-DETR](https://github.com/Sense-X/Co-DETR), [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) and [GLIGEN](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/gligen) was used for this homework. After further fine-tuning the model by generated dataset, the model achieved 0.546 mAP on the validation set.

## Image Captioning

### Test Two Models

[blip2-opt-6.7b](https://huggingface.co/Salesforce/blip2-opt-6.7b) and [blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) were tested. An example of captioning is shown below.

| Model | Caption                                                     |
|-------|-------------------------------------------------------------|
| 6.7b  | penguin swimming in the water with its head above the water |
| 2.7b  | penguin swimming in the water at the zoo                    |

[blip2-opt-6.7b](https://huggingface.co/Salesforce/blip2-opt-6.7b) was selected for the following steps.

### Templates of Prompts for Generative Models

Two templates were tested. The first one is `{caption}, {label}`, the second one is `{caption}, {label}, marine, ocean, HD quality, high detail, height: {height}, width: {width}` with carefully hand-crafted post-processing.



## Generate Images

### Text Only Generation (Generation Task)

Two templates of prompts listed above were tested. The second one outperforms the first one regarding the FID score.

### Image Grounding Generation (Inpainting)

Images with the second template of prompts were used. The FID score improved compared to the text only generation.



## Performance on Generated Images (FID Score)

The generated images were selected individually for each method. 20 images per class (140 images in total) were selected. The FID score was calculated between the generated images and the corresponding original images. The results are shown below.

|        | Text Grounding | Text Grounding | Image Grounding |
|--------|----------------|----------------|-----------------|
| Prompt | Template 1     | Template 2     | Template 2      |
| FID    | 152.506        | 140.487        | 110.366         |



## Performance of Object Detection

Generated images together with the corresponding original images constructed a new balanced dataset. The dataset is too small to train from a pretrained model (from other datasets). Therefore, the model was fine-tuned for 4 epochs from a trained model which was trained on our original custom dataset for 33 epochs. The results are shown below.

|        | Before | Text Grounding | Image Grounding |
|--------|--------|----------------|-----------------|
| mAP    | 0.5418 | 0.5469         | 0.5467          |

The performance of the model did improve after fine-tuning on the generated dataset. However, the advantage of image grounding is not shown in the results compared to text grounding. The reason might be that the text-grounding images are more diverse, and I picked the images too carefully for the image grounding.

<div style="page-break-after: always;"></div>

##  Visualization

<!--
mogrify -resize 768 +repage ./*.png
convert \
\( 1_0023.png 1_0039.png 1_0054.png 1_0158.png 1_0436.png -append \) \
\( 2_0148.png 2_0150.png 2_0167.png 2_0385.png 2_0402.png -append \) \
\( 3_0051.png 3_0084.png 3_0277.png 3_0299.png 3_0327.png -append \) \
\( 4_0090.png 4_0140.png 4_0280.png 4_0408.png 4_0419.png -append \) \
\( 5_0062.png 5_0135.png 5_0312.png 5_0315.png 5_0442.png -append \) \
\( 6_0113.png 6_0188.png 6_0191.png 6_0214.png 6_0342.png -append \) \
\( 7_0058.png 7_0270.png 7_0296.png 7_0337.png 7_0380.png -append \) +append inpaint_example.png
-->

![inpaint_example](./inpaint_example.jpeg)