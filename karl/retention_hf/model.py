import numpy as np
from collections import Counter
from typing import Dict

import torch
import torch.nn as nn

from transformers import (
    PretrainedConfig,
    DistilBertTokenizerFast,
    DistilBertModel,
    DistilBertPreTrainedModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from transformers.data.metrics import simple_accuracy

from karl.config import settings
from karl.retention_hf.data import RetentionInput, RetentionFeaturesSchema
from karl.retention_hf.data import RetentionDataset, retention_data_collator


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
        retention_feature_size=19,
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
        self.num_labels = config.num_labels
        self.retention_feature_size = config.retention_feature_size
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        if config.retention_feature_size == 0:
            self.pre_classifier = nn.Linear(config.dim, config.dim)
        else:
            self.pre_classifier = nn.Linear(config.dim + config.retention_feature_size, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
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
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        # assuming the whole batch is either all new or all old
        # the first feature is `is_new_fact`
        if self.retention_feature_size > 0:
            pooled_output = torch.cat((pooled_output, retention_features), axis=1)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}


def train_individual(fold='new_card'):
    feature_keys = [
        key for key in RetentionFeaturesSchema.__fields__.keys()
        if key not in ['user_id', 'card_id', 'label']
    ]
    retention_feature_size = 0 if fold == 'new_card' else len(feature_keys)
    config = DistilBertRetentionModelConfig(retention_feature_size=retention_feature_size, num_labels=2)
    model = DistilBertRetentionModel(config=config)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = RetentionDataset(settings.DATA_DIR, f'train_{fold}', tokenizer)
    test_dataset = RetentionDataset(settings.DATA_DIR, f'test_{fold}', tokenizer)
    training_args = TrainingArguments(
        output_dir=f'{settings.CODE_DIR}/output/retention_hf_{fold}',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-05,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=retention_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()


def eval_individual(fold='new_card'):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = RetentionDataset(settings.DATA_DIR, f'train_{fold}', tokenizer)
    test_dataset = RetentionDataset(settings.DATA_DIR, f'test_{fold}', tokenizer)

    training_args = TrainingArguments(
        output_dir=f'{settings.CODE_DIR}/output/retention_hf_{fold}',
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-05,
    )
    model = DistilBertRetentionModel.from_pretrained(training_args.output_dir)
    model.eval()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=retention_data_collator,
        compute_metrics=compute_metrics,
    )

    result = trainer.evaluate(eval_dataset=test_dataset)
    print("***** Eval results retention *****")
    for key, value in result.items():
        print("  %s = %s", key, value)


def test_majority_baseline(fold='new_card'):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = RetentionDataset(settings.DATA_DIR, f'train_{fold}', tokenizer)
    test_dataset = RetentionDataset(settings.DATA_DIR, f'test_{fold}', tokenizer)
    labels = [x.label for x in train_dataset]
    label_majority = Counter(labels).most_common()[0][0]
    print(sum([x.label == label_majority for x in test_dataset]) / len(test_dataset))


if __name__ == '__main__':
    # train_individual(fold='new_card')
    # train_individual(fold='old_card')
    # eval_individual(fold='new_card')
    # eval_individual(fold='old_card')
    test_majority_baseline(fold='new_card')
    test_majority_baseline(fold='old_card')
