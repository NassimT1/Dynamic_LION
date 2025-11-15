import json
import yaml
import torch
import pandas as pd
import numpy as np

from tag_dataset import TagEmbeddingsDataset
from bert_model import BertTagEmbeddings
from torch.utils.data import DataLoader
from tqdm import tqdm
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


def prepare_dataset(path: str, test_ratio: float):
    dset = load_dataset(path)
    return split_dataset(dset, test_ratio)


if __name__ == "__main__":
    with open("bert/fine-tune.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    train_tags, test_tags, train_v_caps, test_v_caps = prepare_dataset(
        cfg["dataset_path"], cfg["test_ratio"]
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])

    train_dset = TagEmbeddingsDataset(train_tags, train_v_caps, tokenizer)
    train_loader = DataLoader(train_dset, batch_size=cfg["batch_size"], shuffle=True)

    test_dset = TagEmbeddingsDataset(test_tags, test_v_caps, tokenizer)
    test_loader = DataLoader(train_dset, batch_size=cfg["batch_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device} loaded")

    bert = BertTagEmbeddings(cfg["model_id"])
    bert.to(device)

    loss_fn = torch.nn.TripletMarginLoss()
    optimizer = torch.optim.AdamW(bert.parameters(), lr=5e-5)

    epochs = 1
    for i in tqdm(range(epochs)):
        bert.train()
        total_loss = 0
        for tag_embs, embs, embs_neg in train_loader:
            tag_embs = {k: v.to(device) for k, v in tag_embs.items()}
            embs = embs.to(device)

            preds = bert(tag_embs)
            loss = loss_fn(preds, embs, embs_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {i}: train loss = {total_loss / len(train_loader):.4f}")

        bert.eval()
        with torch.no_grad():
            val_loss = 0
            for tokens, embs, embs_neg in test_loader:
                tokens = {k: v.to(device) for k, v in tokens.items()}
                embs = embs.to(device)
                preds = bert(**tokens)
                val_loss += loss_fn(preds, embs, embs_neg).item()
            print(f"Validation loss = {val_loss / len(test_loader):.4f}")

    torch.save(bert.state_dict(), "bert_embeddings.pt")
