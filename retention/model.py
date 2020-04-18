import abc
from typing import Dict, Optional

import torch
import torch.nn as nn
from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder


class RetentionModel(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            dropout: float,
            hidden_dim: int,
    ):
        super().__init__(vocab)
        self.hidden_dim = hidden_dim
        self.num_labels = 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_labels),
        )
        self.accuracy = CategoricalAccuracy()
        self.loss = torch.nn.CrossEntropyLoss()

    @abc.abstractmethod
    def forward(self, *inputs):
        pass

    def hidden_to_output(
        self,
        # (batch_size, hidden_size)
        hidden: torch.LongTensor,
        label: torch.IntTensor = None,
    ):
        # (batch_size, n_classes)
        logits = self.classifier(hidden)
        # (batch_size, n_classes)
        probs = nn.functional.softmax(logits, dim=-1)

        output_dict = {
            'logits': logits,
            'probs': probs,
            'preds': torch.argmax(logits, 1)
        }

        if label is not None:
            loss = self.loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self.accuracy(logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset=reset)}


@Model.register("bert_retention_model")
class BERTRetentionModel(RetentionModel):

    def __init__(
            self,
            vocab: Vocabulary,
            dropout: float,
            extra_hidden_dim: int,
            qid_embedder: TextFieldEmbedder,
            uid_embedder: TextFieldEmbedder,
            bert_pooling: str = 'cls',
            bert_finetune: bool = False,
            model_name_or_path: str = "bert-base-uncased",
            return_embedding: bool = False,
    ) -> None:
        bert = PretrainedTransformerEmbedder(model_name_or_path, requires_grad=bert_finetune)
        if not bert_finetune:
            for param in bert.parameters():
                param.requires_grad = False
        hidden_dim = (
            bert.get_output_dim()
            + uid_embedder.get_output_dim()
            + qid_embedder.get_output_dim()
            + extra_hidden_dim
        )
        super().__init__(
            vocab=vocab,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        self.bert = bert
        self.uid_embedder = uid_embedder
        self.qid_embedder = qid_embedder
        self.model_name_or_path = model_name_or_path
        self.bert_pooling = bert_pooling
        self.return_embedding = return_embedding

    def forward(
            self,
            tokens: Dict[str, torch.LongTensor],
            uid: Dict[str, torch.Tensor],
            qid: Dict[str, torch.Tensor],
            features: torch.Tensor,
            label: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        input_ids = tokens["text"]
        if self.bert_pooling == "cls":
            # CLS token is always the first
            bert_emb = self.bert(input_ids)[:, 0, :]
        elif self.bert_pooling == "mean":
            mask = (input_ids != 0).long()[:, :, None]
            bert_seq_emb = self.bert(input_ids)
            bert_emb = util.masked_mean(bert_seq_emb, mask, dim=1)
        else:
            raise ValueError("Invalid pooling option")

        uid_emb = self.uid_embedder(uid)[:, 0, :]
        qid_emb = self.qid_embedder(qid)[:, 0, :]

        hidden = torch.cat((
            bert_emb,
            uid_emb,
            qid_emb,
            features), dim=-1)
        output_dict = self.hidden_to_output(hidden, label)

        if self.return_embedding:
            output_dict['bert_emb'] = bert_emb

        return output_dict
