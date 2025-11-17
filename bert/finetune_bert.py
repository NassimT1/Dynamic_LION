import json
import yaml
import torch
import pandas as pd
import numpy as np

from tag_dataset import TagEmbeddingsDataset
from bert_model import BertTagEmbeddings
from trainer import Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def load_dataset(path: str = "Dataset/MS_COCO_2017_tags_embeddings.parquet"):
    df = pd.read_parquet(path)
    return df


def parse_json(s: str):
    return np.fromiter(json.loads(s), dtype=np.float32)


def split_dataset(dataset: pd.DataFrame, test_ratio: float = 0.1):
    df = dataset
    tags = df["image_tag_string"]
    tags = np.array(tags)
    v_caps = np.array(df["V_caption"].map(parse_json))

    idx = int(len(df) * test_ratio)
    train_tags = tags[idx:]
    test_tags = tags[:idx]
    train_v_caps = v_caps[idx:]
    test_v_caps = v_caps[:idx]

    return train_tags, test_tags, train_v_caps, test_v_caps


def prepare_dataset(
    path: str, test_ratio: float, batch_size: int, tokenizer: AutoTokenizer
):
    dset = load_dataset(path)
    train_tags, test_tags, train_v_caps, test_v_caps = split_dataset(dset, test_ratio)

    train_dset = TagEmbeddingsDataset(train_tags, train_v_caps, tokenizer)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)

    test_dset = TagEmbeddingsDataset(test_tags, test_v_caps, tokenizer)
    test_loader = DataLoader(test_dset, batch_size=batch_size)
    return train_loader, test_loader


def prepare_models(model_id: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bert = BertTagEmbeddings(model_id)
    bert.to(device)
    return tokenizer, bert


if __name__ == "__main__":
    with open("bert/fine-tune.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device} loaded")

    n_epochs = cfg["n_epochs"]
    tokenizer, bert = prepare_models(cfg["model_id"], device)

    train_loader, val_loader = prepare_dataset(
        cfg["dataset_path"], cfg["test_ratio"], cfg["batch_size"], tokenizer
    )

    triplet_loss = torch.nn.TripletMarginLoss()
    adamw = torch.optim.AdamW(bert.parameters(), lr=5e-5)
    metrics = {
        "cosine_similarity": torch.nn.CosineSimilarity(dim=1, eps=1e-6),
    }

    trainer = Trainer(
        model=bert,
        n_epochs=n_epochs,
        validate_every=5,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=triplet_loss,
        optimizer=adamw,
        metrics=metrics,
        log_path="bert/bert_hist.csv",
    )

    trainer.fit()

    torch.save(bert.state_dict(), "bert/bert_embeddings.pt")
