from typing import Dict
import numpy as np
import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    Seq2VecEncoder, TextFieldEmbedder, 
)
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask


@Model.register('baseline')
class Baseline(Model):
    def __init__(self, 
                vocab: Vocabulary,
                text_field_embedder: TextFieldEmbedder,
                seq2vec_encoder: Seq2VecEncoder
):
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._seq2vec_encoder = seq2vec_encoder
        self._classification_layer = torch.nn.Linear(in_features = TextFieldEmbedder.get_output_dim() + 4,
                                            out_features = 1)
        self._loss = nn.BCEWithLogitsLoss()

    # Change these to match the text_to_instance argument names
    def forward(self,
                text: Dict[str, torch.Tensor],
                answer: torch.Tensor,
                metadata: Dict,
                question_features: np.ndarray,
                user_features: np.ndarray,
                overall_accuracy_per_user: np.ndarray,
                ):
        # This is where all the modeling stuff goes
        # AllenNLP requires that if there are training labels,
        # that a dictionary with key "loss" be returned.
        # You can stuff anything else in the dictionary that might be helpful
        # later.
        mask = get_text_field_mask(text)
        question_embeddings = self._text_field_embedder(text)
        question_encoder_out = self._seq2vec_encoder(question_embeddings, mask)
        encoder_out = np.concatenate((question_encoder_out, question_features, user_features), asix = None)
        logits = self._classification_layer(encoder_out)
        output = {"logits": logits}
        if overall_accuracy_per_user is not None:
            self.accuracy(logits, overall_accuracy_per_user, mask)
            output["loss"] = self._loss(tag_logits, overall_accuracy_per_user, mask)
        output["loss"] = self.loss(logits, overall_accuracy_per_user)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}