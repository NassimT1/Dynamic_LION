import os
import argparse
import logging
from models.lion_t5 import LIONT5InstructAdapter  # register
from common.registry import registry
from omegaconf import OmegaConf
from datetime import datetime


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
    # out_base = cfg.run.get("output_dir", "outputs/lion_train")
    # job_id = datetime.now().strftime("eval_%Y%m%d%H")
    # out_dir = os.path.join(out_base, job_id)
    # os.makedirs(out_dir, exist_ok=True)

    # accelerator = Accelerator(
    #     log_with=["tensorboard"],
    #     project_dir=out_dir,
    #     mixed_precision=cfg.run.get("mixed_precision", "bf16"),
    # )
    # accelerator.init_trackers(project_name="lion_train")

    # logging.basicConfig(level=logging.INFO)
    model = build_model(cfg)
    # eval_datasets = build_eval_datasets(cfg)


if __name__ == "__main__":
    main()
