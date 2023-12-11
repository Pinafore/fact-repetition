#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

from transformers import (
    PretrainedConfig,
    BertModel,
    BertPreTrainedModel,
)


class BertRetentionModelConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        position_embedding_type="absolute",
        retention_feature_size=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.retention_feature_size = retention_feature_size


class BertRetentionModel(BertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.retention_feature_size = config.retention_feature_size
        self.bert = BertModel(config)
        hidden_dim = config.hidden_size + config.retention_feature_size
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(config.hidden_dropout_prob),
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
        bert_output = self.bert(
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

        outputs = [x + bert_output]

        if labels is not None:
            loss = self.loss_fn(x, labels)
            outputs = [loss] + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
