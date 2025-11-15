import torch.nn as nn

from transformers import AutoModel


class BertTagEmbeddings(nn.Module):
    def __init__(
        self,
        model_id: str = "bert-base-uncased",
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_id)

    def forward(self, x):
        bert_out = self.bert(**x)
        output = bert_out.last_hidden_state[:, 0, :]
        return output
