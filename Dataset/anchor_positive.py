import json
import pandas as pd
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer


def load_npz(path: str):
    df = np.load(path, allow_pickle=True)
    return df


def load_dataset(path: str = "Dataset/MS_COCO_2017_tags_embeddings.parquet"):
    df = pd.read_parquet(path)
    return df


def parse_json(s: str):
    return np.fromiter(json.loads(s), dtype=np.float32)


def convert_dataset(model: AutoTokenizer, src_path: str, dest_path: str):
    df = load_dataset(src_path)
    tags = df["image_tag_string"]
    v_caps = df["V_caption"].map(parse_json)

    assert len(tags) == len(v_caps)

    text_embs = []
    vis_caps = []
    i = 0
    for tag, v_cap in tqdm(zip(tags, v_caps), total=len(tags), desc="Encoding dataset"):
        embeddings = string_to_embeddings(model, tag)
        embeddings = {k: v.squeeze(0) for k, v in embeddings.items()}
        text_embs.append(embeddings)

        v_cap = np.array(v_cap, dtype=np.float32)
        vis_caps.append(v_cap)
        i += 1
        if i == 2:
            break

    out_df = np.savez(
        dest_path,
        text_embs=np.array(text_embs),
        vis_caps=np.array(vis_caps),
    )
    return out_df


def string_to_embeddings(model, sentence: str):
    """Outputs (768, ) Tensor"""
    tag_embs = model(
        sentence,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return tag_embs


if __name__ == "__main__":
    model = AutoTokenizer.from_pretrained("bert-base-uncased")
    src_path = "Dataset/MS_COCO_2017_tags_embeddings.parquet"
    dest_path = "Dataset/anchor_pos.npz"
    convert_dataset(model, src_path, dest_path)
    df = load_npz(dest_path)
