import torch

from torch.utils.data import Dataset


class TagEmbeddingsDataset(Dataset):
    def __init__(self, tags, v_caps, tokenizer):
        self.tags = tags
        self.v_caps = v_caps
        assert len(self.tags) == len(v_caps)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        tags = self.tags[idx]
        tokens = self.tokenizer(
            tags,
            truncation=True,
            padding="max_length",
            max_length=32,
            return_tensors="pt",
        )
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        embeddings = torch.tensor(self.v_caps[idx], dtype=torch.float32)
        return tokens, embeddings
