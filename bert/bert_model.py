import torch.nn as nn

from transformers import AutoModel


class BertTagEmbeddings(nn.Module):
    def __init__(
        self,
        model_id: str = "bert-base-uncased",
        in_dim: int = 768,
        out_dim: int = 2048,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.bert = AutoModel.from_pretrained(model_id)
        self.linear = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        bert_out = self.bert(**x)
        class_embeddings = bert_out.last_hidden_state[:, 0, :]
        output = self.linear(class_embeddings)
        return output
