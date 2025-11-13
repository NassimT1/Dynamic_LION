import json
import pandas as pd
import numpy as np

from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_dataset(path: str = "Dataset/MS_COCO_2017_tags_embeddings.parquet"):
    df = pd.read_parquet(path)
    return df


def parse_json(s: str):
    return np.fromiter(json.loads(s), dtype=np.float32)


def convert_dataset_into_triplet(
    model: SentenceTransformer, src_path: str, dest_path: str
):
    df = load_dataset(src_path)
    tags = df["image_tag_string"]
    v_caps = df["V_caption"].map(parse_json)

    assert len(tags) == len(v_caps)

    text_embs = []
    vis_caps = []
    for tag, v_cap in tqdm(zip(tags, v_caps), total=len(tags), desc="Encoding dataset"):
        embeddings = string_to_embeddings(model, tag)
        text_embs.append(embeddings)

        v_cap = np.array(v_cap, dtype=np.float32)
        vis_caps.append(v_cap)

    text_embs = np.array(text_embs)
    vis_caps = np.array(vis_caps)

    out_df = pd.DataFrame(
        {
            "tag_embeddings": text_embs,
            "v_cap_embeddings": vis_caps,
        }
    )

    out_df.to_parquet(dest_path, index=False)
    return out_df


def string_to_embeddings(model: SentenceTransformer, sentence: str):
    """Outputs (768, ) Tensor"""
    embeddings = model.encode(sentence)
    return embeddings


if __name__ == "__main__":
    model = SentenceTransformer("all-mpnet-base-v2")
    src_path = "Dataset/MS_COCO_2017_tags_embeddings.parquet"
    convert_dataset_into_triplet(model, src_path, "Dataset/anchor_pos.parquet")
