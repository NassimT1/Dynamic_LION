import torch.nn as nn

from transformers import AutoModel


class BertTagEmbeddings(nn.Module):
    """
    Input: token embeddings size of 512
    Output: embeddings size of 768
    Given some token embedding vector including an information
        of image tags and corresponding confidence scores information,
        outputs an embedding vector representing visual captions of the image.
    """

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
