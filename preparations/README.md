# Preparing the Training

LION requires various datasets for training and inference.
The following is the required datasets as suggested from the [original repository](https://github.com/JiuTian-VL/JiuTian-LION).

## To-Do List

- [x] [LION Train data](https://huggingface.co/datasets/daybreaksly/LION-data-train)
- [x] [OCR-VQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_)
  - [Hugging Face](https://huggingface.co/datasets/howard-hou/OCR-VQA)
- [x] [coco-2014](https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3)
  - [Hugging Face](https://huggingface.co/datasets/visual-layer/coco-2014-vl-enriched)
- [x] [coco-2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
  - [Hugging Face](https://huggingface.co/datasets/rafaelpadilla/coco2017)
- [x] [okvqa-2014](https://okvqa.allenai.org/download.html)
  - [Hugging Face](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA)
- [x] [textcaps](https://textvqa.org/textcaps/dataset/)
  - [Hugging Face](https://huggingface.co/datasets/HuggingFaceM4/TextCaps)
- [x] [vqav2-2014](https://visualqa.org/download.html)
  - [Hugging Face](https://huggingface.co/datasets/HuggingFaceM4/VQAv2)
- [x] [visual_genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)
  - [Hugging Face](https://huggingface.co/datasets/ranjaykrishna/visual_genome)

## Dataset Directory Tree

```tree
/path/to/data/images/
├── OCR-VQA/images
├── coco/images/train2014
├── coco_2017/train2017
├── okvqa/images/train/train2014
├── textcaps/images/train_images
├── vqav2/images/train2014
├── visual_genome/VG_100K
└── visual_genome/VG_100K_2
```

## Checkpoints

| Version         | Checkpoint                                                                        |
| --------------- | --------------------------------------------------------------------------------- |
| LION-FlanT5-XL  | [daybreaksly/LION-FlanT5-XL](https://huggingface.co/daybreaksly/LION-FlanT5-XL)   |
| LION-FlanT5-XXL | [daybreaksly/LION-FlanT5-XXL](https://huggingface.co/daybreaksly/LION-FlanT5-XXL) |

## Usage

### Prepare models

- [x] [eva_vit_g](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)
- [x] [ram_swin_large_14m](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth)
- [ ] [FlanT5-XL](https://huggingface.co/google/flan-t5-xl).
- [x] [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
