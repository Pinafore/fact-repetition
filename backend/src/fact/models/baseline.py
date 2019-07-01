from typing import Dict
import numpy as np
import torch
from torch import nn
from allennlp.nn import InitializerApplicator
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    Seq2SeqEncoder, Seq2VecEncoder, 
    TextFieldEmbedder
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
                text_field_embedder: TextFieldEmbedder,
                seq2vec_encoder: Seq2VecEncoder,
                dropout: float,
                num_labels: int = None,
                initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)

        EMBEDDING_DIM = 100
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
        self._text_field_embedder = word_embeddings
        self._seq2vec_encoder = seq2vec_encoder
        self._classifier_input_dim = self._seq2vec_encoder.get_output_dim() + 4
        
        if dropout != 0:
            self._dropout = nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self._num_labels = 2

        classification_layer = [nn.Linear(self._classifier_input_dim, self._num_labels)]
        if dropout != 0:
            classification_layer.append(nn.Dropout(dropout))
        self._classification_layer = nn.Sequential(*classification_layer)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    # Change these to match the text_to_instance argument names
    def forward(self,
                text: Dict[str, torch.Tensor],
                answer: torch.Tensor,
                metadata: Dict,
                question_features: np.ndarray,
                user_features: np.ndarray,
                label: torch.Tensor = None,
                ):
        # This is where all the modeling stuff goes
        # AllenNLP requires that if there are training labels,
        # that a dictionary with key "loss" be returned.
        # You can stuff anything else in the dictionary that might be helpful
        # later.
        embedded_text = self._text_field_embedder(text)
        mask = get_text_field_mask(text).float()
        embedded_text = self._dropout(embedded_text)
        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)
        # print("embedded_text: ", type(embedded_text), 'dim ', embedded_text.dim(), 'size', embedded_text.size())
        # features = torch.cat((question_features, user_features), dim=0)
        # print("features: ", features.size())
        embeddings = torch.cat((embedded_text, question_features, user_features), dim=1)
        # print("embeddings: ", embeddings.size())
        logits = self._classification_layer(embeddings)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {'logits': logits, 'probs': probs}

        if label is not None:
            loss = self._loss(
                # Empirically we have found dropping out after logits works better
                self._dropout(logits),
                label.long().view(-1)
            )
            output_dict['loss'] = loss
            self._accuracy(logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}    