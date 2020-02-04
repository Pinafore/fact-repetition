from typing import Dict, Union, Optional
import os
import numpy as np
import torch
from torch import nn
from overrides import overrides
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.regularizers import  L2Regularizer
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    Seq2SeqEncoder, Seq2VecEncoder, 
    TextFieldEmbedder
)
from allennlp.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.token_embedders import Embedding
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel

from pytorch_pretrained_bert.modeling import BertModel


BERT_DIM = 768


@Model.register('karl_model')
class KarlModel(Model):
    def __init__(self, 
                vocab: Vocabulary,
                qid_embedder: TextFieldEmbedder,
                uid_embedder: TextFieldEmbedder,
                bert_model: Union[str, BertModel],
                bert_train: bool,
                dropout: float,
                use_bert: bool,
                use_rnn: bool,
                contextualizer: Optional[Seq2VecEncoder] = None,
                text_field_embedder: Optional[TextFieldEmbedder] = None,
                ):
        super().__init__(vocab)

        if int(use_bert) + int(use_rnn) != 1:
            raise ValueError('Must use one of bert or rnn')

        self.output_embedding = False
        self._uid_embedder = uid_embedder
        self._qid_embedder = qid_embedder
        self._text_field_embedder = text_field_embedder
        self._use_bert = use_bert
        self._use_rnn = use_rnn
        self._bert_train = bert_train
        if use_bert:
            if isinstance(bert_model, str):
                self._bert_model = PretrainedBertModel.load(bert_model)
            else:
                self._bert_model = bert_model
            for param in self._bert_model.parameters():
                param.requires_grad = bert_train
        else:
            self._bert_model = None
        
        if use_rnn:
            self._we_embed = text_field_embedder
            self._context = contextualizer
        else:
            self._we_embed = None
            self._context = None
        
        if dropout != 0:
            self._dropout = nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        if use_bert:
            classifier_input_dim = BERT_DIM + 100 + 28 # 35 category onehot; 19 difficulty onehot
        else:
            classifier_input_dim = self._context.get_output_dim() + 100 + 28

        self._classifier = nn.Linear(classifier_input_dim, 2)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    # Change these to match the text_to_instance argument names
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                user_id: Dict[str, torch.Tensor],
                question_id: Dict[str, torch.Tensor],
                feature_vec: torch.Tensor,
                label: torch.Tensor = None, 
                ):
        output_dict = {}
        if self._use_bert:
            input_ids = tokens['bert']
            token_type_ids = tokens["bert-type-ids"]
            input_mask = (input_ids != 0).long()
            _, q_rep = self._bert_model(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=input_mask)
        else:
            mask = get_text_field_mask(tokens).float()
            q_rep = self._context(self._we_embed(tokens), mask)

        q_rep = self._dropout(q_rep)
        uid_embedding = self._uid_embedder(user_id)[:, 0, :]
        qid_embedding = self._qid_embedder(question_id)[:, 0, :]

        if self.output_embedding:
            output_dict['q_rep'] = q_rep
            output_dict['feature_vec'] = feature_vec
            output_dict['uid_embedding'] = uid_embedding
            output_dict['qid_embedding'] = qid_embedding

        encoding = torch.cat((q_rep, feature_vec, uid_embedding, qid_embedding), dim=-1)
        logits = self._classifier(encoding)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict['logits'] = logits
        output_dict['probs'] = probs

        if label is not None:
            self._accuracy(logits, label)
            loss = self._loss(logits, label.long().view(-1))
            output_dict['loss'] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]: 
        metrics = {'accuracy': self._accuracy.get_metric(reset=reset)}
        return metrics 