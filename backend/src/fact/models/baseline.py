from typing import Dict
import numpy as np
import torch
from torch import nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    Seq2VecEncoder, TextFieldEmbedder, 
)
from allennlp.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.token_embedders import Embedding
from allennlp.training.metrics import CategoricalAccuracy

@Model.register('baseline')
class Baseline(Model):
    def __init__(self, 
                vocab: Vocabulary,
                # text_field_embedder: TextFieldEmbedder,
                # seq2vec_encoder: Seq2VecEncoder
):
        super().__init__(vocab)

        EMBEDDING_DIM = 50
        HIDDEN_DIM = 6
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
        dan = BagOfEmbeddingsEncoder(embedding_dim=EMBEDDING_DIM, averaged=True)

        self._text_field_embedder = word_embeddings
        self._seq2vec_encoder = dan
        self._classification_layer = torch.nn.Linear(in_features =self._text_field_embedder.get_output_dim(),
                                            out_features = 1)
        self._loss = nn.BCEWithLogitsLoss()
        self._accuracy = CategoricalAccuracy()

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
        # encoder_out = torch.cat((question_encoder_out.detach, question_features, user_features), 0)
        encoder_out = question_encoder_out
        logits = self._classification_layer(encoder_out)
        output = {"logits": logits}
        if overall_accuracy_per_user is not None:
            print("logits: ", logits)
            print("logits type: ", type(logits))
            print("logits size: ", logits.size())
            print("logits dim: ", logits.dim())
            print("overall_accuracy_per_user: ", overall_accuracy_per_user)
            print("overall_accuracy_per_user type: ", type(overall_accuracy_per_user))
            print("overall_accuracy_per_user size: ", overall_accuracy_per_user.size())
            print("overall_accuracy_per_user dim: ", overall_accuracy_per_user.dim())
            # self._accuracy(logits, overall_accuracy_per_user, mask)
            output["loss"] = self._loss(logits, overall_accuracy_per_user)
        output["loss"] = self._loss(logits, overall_accuracy_per_user)
        return output

    # def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #     return {"accuracy": self.accuracy.get_metric(reset)}    