import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

from PIL import Image
import torch
from torch.utils.data import Dataset

from common.registry import registry
from ram.transform import get_transform
from preprocessors.lion_preprocessors import ImageTrainProcessor, ImageEvalProcessor


@dataclass
class JSONLVQASample:
    image_path: str
    question: str
    answer: str
    category: str = "image_level"  # or "region_level"
    tags: Optional[str] = None


class JSONLVQADataset(Dataset):
    def __init__(self, ann_path: Union[str, List[str]], vis_root: Optional[str], is_train: bool = True) -> None:
        # Support a single file or a list of files in JSON format.
        self.vis_root = vis_root
        self.is_train = is_train
        self.samples: List[Dict[str, Any]] = []

        if isinstance(ann_path, str):
            ann_paths = [ann_path]
        else:
            ann_paths = list(ann_path)

        for p in ann_paths:
            assert os.path.exists(p), f"Annotation file not found: {p}"
            self._load_ann_file(p)

        self.processor = ImageTrainProcessor.from_config() if is_train else ImageEvalProcessor.from_config()
        self.ram_processor = get_transform()

    def _load_ann_file(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
            assert isinstance(data, list), f"Expected a list of records in {path}"
            self.samples.extend(data)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        # Accept Combined_VQA style keys: image_path OR image_id (fallback), and our own "image"
        img_path = rec.get("image") or rec.get("image_path") or rec.get("image_id")
        assert img_path, f"Missing image path for record {idx}"
        if self.vis_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.vis_root, img_path)
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            image = self.processor(im)
            ram_image = self.ram_processor(im)

        q = rec.get("question")
        a = rec.get("answer", "")
        # Normalize answer to string if provided as a list
        if isinstance(a, list):
            a = a[0] if len(a) > 0 else ""
        # Map Combined_VQA categories (e.g., "ref"/"norm") to expected router names
        raw_cat = rec.get("category", "image_level")
        cat = self._normalize_category(raw_cat)
        tags = rec.get("tags", None)
        if isinstance(tags, list):
            tags = ", ".join(tags)
        sample = {
            "image": image,
            "ram_image": ram_image,
            "question": q,
            "answer": a,
            "category": [cat],  # keep batch shape expectation
        }
        if tags is not None:
            sample["tags"] = [tags]
        return sample

    @staticmethod
    def _normalize_category(cat: Any) -> str:
        # Accept various synonyms: Combined_VQA may use {"ref", "norm"}
        if not isinstance(cat, str):
            return "image_level"
        c = cat.lower()
        if c in {"ref", "region", "region_level", "grounding"}:
            return "region_level"
        # default to image-level for other cases like "norm" or unknown
        return "image_level"

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # images: stack; strings: list; category/tags as list of lists
        images = torch.stack([b["image"] for b in batch])
        ram_image = torch.stack([b["ram_image"] for b in batch])
        questions = [b["question"] for b in batch]
        answers = [b["answer"] for b in batch]
        categories = [b["category"][0] for b in batch]
        # Router supports a single route index across the module; use the first
        batch_category = [categories[0]]
        out = {
            "image": images,
            "ram_image": ram_image,
            "question": questions,
            "answer": answers,
            "category": batch_category,
        }
        if "tags" in batch[0]:
            out["tags"] = [b.get("tags", [None])[0] for b in batch]
        return out
