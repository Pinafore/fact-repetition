from typing import Dict
import os
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
from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.token_embedders import Embedding
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from pytorch_transformers import BertPreTrainedModel, BertModel, BertTokenizer, BertConfig

@Model.register('bert_baseline')
class Baseline(Model):
    def __init__(self, 
                vocab: Vocabulary,
                text_field_embedder: TextFieldEmbedder,
                dropout: float,
                num_labels: int = None,
                initializer: InitializerApplicator = InitializerApplicator(),
                ):
        super().__init__(vocab)
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=100)
        uid_token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('uid_tokens'),
                            embedding_dim=50)
        qid_token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('qid_tokens'),
                            embedding_dim=50)
        self._uid_embedder = BasicTextFieldEmbedder({"uid_tokens": uid_token_embedding})
        self._qid_embedder = BasicTextFieldEmbedder({"qid_tokens": qid_token_embedding})
        self._text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
        self._bert = BertModel.from_pretrained('bert-base-uncased')
        # for param in self._bert.parameters():
        #     param.requires_grad = False
        # self._bert.embeddings.word_embeddings = nn.Embedding(vocab.get_vocab_size(), 768, padding_idx=0)
        self._bert.cuda()
        self._bert_pooler = BertPooler('bert-base-uncased')
        # 64809 + 125419
        # self._classifier_input_dim = self._seq2vec_encoder.get_output_dim() + 768 + 9 + 100
        self._classifier_input_dim = 768 + 9 + 100
        
        if dropout != 0:
            self._dropout = nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self._num_labels = 2
        classification_layer = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer = classification_layer
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    # Change these to match the text_to_instance argument names
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                uid_tokens: Dict[str, torch.Tensor],
                qid_tokens: Dict[str, torch.Tensor],
                embedding: np.ndarray,
                feature_vec: np.ndarray,
                label: torch.Tensor = None, 
                ):
        # print("tokens", type(tokens), tokens)
        # print("uid_tokens", type(uid_tokens), uid_tokens)
        # print("qid_tokens", type(qid_tokens), qid_tokens)
        mask = get_text_field_mask(tokens).float()
        # print("tokens", tokens['tokens'].shape)
        outputs = self._bert(tokens['tokens'], attention_mask=mask)
        # print("outputs", outputs[0].shape, outputs[1].shape)
        pooled_output = outputs[1]
        # temp = outputs[0][:, 0]
        # print("temp", temp.shape)
        pooled_output = self._dropout(pooled_output)
        # print("pooled_output", pooled_output.shape)
        # bert_embedding = self._bert_pooler(text_embedding)
        # print("====bert_embedding====", bert_embedding)
        # print("embedded_text: ", type(embedded_text), 'dim ', embedded_text.dim(), 'size', embedded_text.size())
        # print(embedded_text)
        uid_embedding = self._uid_embedder(uid_tokens).view(-1, 50)
        qid_embedding = self._qid_embedder(qid_tokens).view(-1, 50)
        encoding = torch.cat((pooled_output, feature_vec, uid_embedding, qid_embedding), dim=1)
        logits = self._classification_layer(encoding)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {'logits': logits, 'probs': probs}

        if label is not None:
            self._accuracy(logits, label)
            loss = self._loss(logits, label.long().view(-1))
            output_dict['loss'] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]: 
        cwd = os.getcwd()
        dir_temp = os.path.join(cwd + "/fine_tuned_bert_model")
        if not os.path.exists(dir_temp):
            os.mkdir(dir_temp)
        self._bert.save_pretrained("fine_tuned_bert_model")
        exit()
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics 