import torch
import random

from torch.utils.data import Dataset


class TagEmbeddingsDataset(Dataset):
    def __init__(self, tags, v_caps, tokenizer):
        self.tags = tags
        self.v_caps = v_caps
        assert len(self.tags) == len(v_caps)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tags)

    def rand_idx(self):
        return random.randint(0, self.__len__())

    def __getitem__(self, idx):
        tags = self.tags[idx]
        tag_embs = self.tokenizer(
            tags,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tag_embs = {k: v.squeeze(0) for k, v in tag_embs.items()}
        cap_embeddings = torch.tensor(self.v_caps[idx], dtype=torch.float32)
        rand_idx = self.rand_idx()
        cap_neg_embeddings = torch.tensor(self.v_caps[rand_idx], dtype=torch.float32)
        return tag_embs, cap_embeddings, cap_neg_embeddings
