import yaml
import torch
import pandas as pd
import numpy as np

from tag_dataset import TagEmbeddingsDataset
from bert_model import BertTagEmbeddings
from trainer import Trainer
from torch.utils.data import DataLoader


def load_npz(path: str):
    df = np.load(path, allow_pickle=True)
    return df


def split_dataset(dataset: pd.DataFrame, dset_size: int, test_ratio: float = 0.1):
    df = dataset
    tags = df["text_embs"]
    tags = tags[:dset_size]
    v_caps = df["vis_caps"]
    v_caps = v_caps[:dset_size]

    idx = int(len(tags) * test_ratio)
    train_tags = tags[idx:]
    test_tags = tags[:idx]
    train_v_caps = v_caps[idx:]
    test_v_caps = v_caps[:idx]

    return train_tags, test_tags, train_v_caps, test_v_caps


def prepare_dataset(path: str, test_ratio: float, dset_size: int, batch_size: int):
    dset = load_npz(path)
    train_tags, test_tags, train_v_caps, test_v_caps = split_dataset(
        dset, dset_size, test_ratio
    )

    train_dset = TagEmbeddingsDataset(train_tags, train_v_caps)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)

    test_dset = TagEmbeddingsDataset(test_tags, test_v_caps)
    test_loader = DataLoader(test_dset, batch_size=batch_size)
    return train_loader, test_loader


def prepare_model(model_id: str, device: str = "cuda"):
    bert = BertTagEmbeddings(model_id)
    bert.to(device)
    return bert


if __name__ == "__main__":
    with open("bert/fine-tune.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device} loaded")

    n_epochs = cfg["n_epochs"]
    bert = prepare_model(cfg["model_id"], device)

    train_loader, val_loader = prepare_dataset(
        cfg["dataset_path"], cfg["test_ratio"], cfg["dset_size"], cfg["batch_size"]
    )

    triplet_loss = torch.nn.TripletMarginLoss()
    adamw = torch.optim.AdamW(bert.parameters(), lr=cfg["lr"])
    metrics = {
        "cosine_similarity": torch.nn.CosineSimilarity(dim=1, eps=1e-6),
    }

    trainer = Trainer(
        model=bert,
        n_epochs=n_epochs,
        validate_every=1,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=triplet_loss,
        optimizer=adamw,
        metrics=metrics,
        log_path="bert/logs",
    )

    trainer.fit()

    torch.save(bert.state_dict(), "bert/bert_embeddings.pt")
