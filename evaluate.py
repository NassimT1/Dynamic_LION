import os
import torch
import argparse
import logging
from models.lion_t5 import LIONT5InstructAdapter  # register
from preprocessors.lion_preprocessors import ImageEvalProcessor
from common.registry import registry
from omegaconf import OmegaConf
from datetime import datetime
from PIL import Image


def build_model(cfg):
    model_cfg = cfg.model
    model_cls = registry.get_model_class("lion_t5")
    return model_cls.from_config(model_cfg)


def build_dataset(ds_cfg):
    from datasets.jsonl_vqa import JSONLVQADataset

    ds = JSONLVQADataset(
        ann_path=ds_cfg.ann_path,
        vis_root=ds_cfg.get("vis_root", None),
        is_train=ds_cfg.get("is_train", False),
    )
    # optional sampling ratio for multi-dataset mixing
    ratio = float(ds_cfg.get("sample_ratio", 1.0))
    setattr(ds, "sample_ratio", ratio)
    return ds


def build_eval_datasets(cfg):
    # Support either a list under cfg.train_datasets or single cfg.train_dataset
    if cfg.get("eval_datasets", None) is not None:
        return [build_dataset(ds_cfg) for ds_cfg in cfg.eval_datasets]
    elif cfg.get("eval_dataset", None) is not None:
        return [build_dataset(cfg.eval_dataset)]
    else:
        raise ValueError(
            "Config must define eval_datasets (list) or eval_dataset (single)"
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg-path",
        default="configs/eval.yaml",
        required=True,
        help="Path to evaluation config YAML",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg_path)

    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    processor = ImageEvalProcessor()

    path1 = "images/coco/images/train2014/COCO_train2014_000000024935.jpg"
    path2 = "images/coco/images/test2014/COCO_test2014_000000000001.jpg"
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")
    proc1 = processor(img1)
    proc2 = processor(img2)

    t1 = model.generate_tags_with_scores(img1)
    t2 = model.generate_tags_with_scores(img2)

    question = "Please describe the image using a single short sentence."

    output1 = model.generate({
        "image": proc1.unsqueeze(0).cuda(),
        "question": [question],
        "tags_for_dynamic_prompt": t1,
        "category": "image_level",
    })
    output2 = model.generate({
        "image": proc2.unsqueeze(0).cuda(),
        "question": [question],
        "tags_for_dynamic_prompt": t2,
        "category": "image_level",
    })

    print(output1)
    print(output2)


if __name__ == "__main__":
    main()
