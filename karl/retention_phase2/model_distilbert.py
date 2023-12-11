#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

from transformers import (
    PretrainedConfig,
    DistilBertModel,
    DistilBertPreTrainedModel,
)


class DistilBertRetentionModelConfig(PretrainedConfig):

    model_type = "distilbert"

    def __init__(
        self,
        vocab_size=30522,
        max_position_embeddings=512,
        sinusoidal_pos_embds=False,
        n_layers=6,
        n_heads=12,
        dim=768,
        hidden_dim=4 * 768,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        initializer_range=0.02,
        qa_dropout=0.1,
        seq_classif_dropout=0.2,
        pad_token_id=0,
        retention_feature_size=0,
        **kwargs,
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.initializer_range = initializer_range
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout
        self.retention_feature_size = retention_feature_size

    @property
    def hidden_size(self):
        return self.dim

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers


class DistilBertRetentionModel(DistilBertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.retention_feature_size = config.retention_feature_size
        self.distilbert = DistilBertModel(config)
        hidden_dim = config.dim + config.retention_feature_size
        self.classifier = nn.Sequential(
            nn.Dropout(config.seq_classif_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(config.seq_classif_dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.loss_fn = nn.BCELoss()
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        retention_features=None,
        output_attentions=None,
        labels=None,
    ):
        bert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        x = hidden_state[:, 0]  # (bs, dim)
        if self.retention_feature_size > 0:
            x = torch.cat((x, retention_features), axis=1)

        x = self.classifier(x)
        x = torch.sigmoid(x)[:, 0]

        outputs = (x,) + bert_output

        if labels is not None:
            loss = self.loss_fn(x, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
