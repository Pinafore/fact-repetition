from typing import Dict
import numpy as np
import torch
from torch import nn
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.regularizers import  L2Regularizer
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
                initializer: InitializerApplicator = InitializerApplicator(),
                # regularizer: L2Regularizer = L2Regularizer(),
                ):
        super().__init__(vocab)

        EMBEDDING_DIM = 100
        # self.uid_num =  vocab.get_vocab_size('uid_tokens')
        # self.qid_num = vocab.get_vocab_size('qid_tokens')
        # self.token_num = vocab.get_vocab_size('tokens')
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
        uid_token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('uid_tokens'),
                            embedding_dim=50)
        qid_token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('qid_tokens'),
                            embedding_dim=50)
        # self._id_embedder = BasicTextFieldEmbedder({"id_tokens": id_token_embedding})
        self._uid_embedder = BasicTextFieldEmbedder({"uid_tokens": uid_token_embedding})
        self._qid_embedder = BasicTextFieldEmbedder({"qid_tokens": qid_token_embedding})
        self._text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
        self._seq2vec_encoder = seq2vec_encoder
        # self._classifier_input_dim =  + 4
        # 64809 + 125419
        # self._classifier_input_dim = self._seq2vec_encoder.get_output_dim() + 768 + 9 + 100
        self._classifier_input_dim = 768 + 9 + 100
        
        if dropout != 0:
            self._dropout = nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self._num_labels = 2

        # classification_layer = [nn.Linear(self._classifier_input_dim, self._num_labels)]
        classification_layer = nn.Linear(self._classifier_input_dim, self._num_labels)
        # if dropout != 0:
        #     classification_layer.append(nn.Dropout(dropout))
        # self._classification_layer = nn.Sequential(*classification_layer)
        self._classification_layer = classification_layer
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    # Change these to match the text_to_instance argument names
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                # answer: torch.Tensor,
                # metadata: Dict,
                uid_tokens: Dict[str, torch.Tensor],
                qid_tokens: Dict[str, torch.Tensor],
                embedding: np.ndarray,
                feature_vec: np.ndarray,
                # user_features: np.ndarray,
                label: torch.Tensor = None,
                ):
        # This is where all the modeling stuff goes
        # AllenNLP requires that if there are training labels,
        # that a dictionary with key "loss" be returned.
        # You can stuff anything else in the dictionary that might be helpful
        # later.
        # print("tokens", type(tokens), tokens)
        # print("uid_tokens", type(uid_tokens), uid_tokens)
        # print("qid_tokens", type(qid_tokens), qid_tokens)
        

        # embedded_text = self._text_field_embedder(tokens)
        # # print("====embedded_text====", embedded_text)
        # mask = get_text_field_mask(tokens).float()
        # embedded_text = self._dropout(embedded_text)
        # embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)


        # print("embedded_text: ", type(embedded_text), 'dim ', embedded_text.dim(), 'size', embedded_text.size())
        # print(embedded_text)

        # print("uid_vocab_size", self.uid_num)
        # print("qid_vocab_size", self.qid_num)
        # print("token_num", self.token_num)
        uid_embedding = self._uid_embedder(uid_tokens).view(-1, 50)
        qid_embedding = self._qid_embedder(qid_tokens).view(-1, 50)
        # print("uid_embedding", uid_embedding.shape, uid_embedding)
        # print("qid_embedding", qid_embedding.shape, qid_embedding)
        # print("DIMEN", tokens['tokens'].shape)
        # print("1", qid_tokens['qid_tokens'].shape)
        # print("2", uid_tokens['uid_tokens'].shape)
        # print(temp1, temp2)
        # print("DIMENSIONS", uid_embedding.shape, qid_embedding.shape, embedding.shape, feature_vec.shape)
        encoding = torch.cat((embedding, feature_vec, uid_embedding, qid_embedding), dim=1)
        # print("encoding", encoding)
        # feature_vec = question_features
        logits = self._classification_layer(encoding)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {'logits': logits, 'probs': probs}
        print("output_dict", output_dict)

        if label is not None:
            self._accuracy(logits, label)
            loss = self._loss(logits, label.long().view(-1))
            output_dict['loss'] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics 