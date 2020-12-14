import os
import json
import logging
import dataclasses
from tqdm import tqdm
from overrides import overrides
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    BertModel,
    DistilBertModel,
    PreTrainedTokenizer,
    BertPreTrainedModel,
    DistilBertPreTrainedModel,
    BertConfig,
    DistilBertConfig,
)

from karl.retention.data import get_split_numpy, get_split_dfs, get_questions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionInputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    # token_type_ids: Optional[List[int]] = None
    retention_features: Optional[List[float]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass
class RetentionDataArguments:

    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing numpy arrays and dataframes"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


class RetentionDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            args: RetentionDataArguments,
            fold: str,
            tokenizer: PreTrainedTokenizer,
    ):
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                fold,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            self.features = torch.load(cached_features_file)
            logger.info(f"Loading features from cached file {cached_features_file}")
        else:
            x_train, y_train, x_test, y_test = get_split_numpy(args.data_dir)
            mean = np.mean(x_train, axis=0)
            std = np.std(x_train, axis=0)
            mean[-1] = 0
            std[-1] = 1

            # precompute featurized dataframes
            df_train, df_test = get_split_dfs(args.data_dir)
            questions = get_questions(args.data_dir)

            if fold == 'train':
                xs, ys, df = x_train, y_train, df_train
            elif fold == 'test':
                xs, ys, df = x_test, y_test, df_test
            else:
                raise ValueError("Data fold must be either train or test")

            if args.max_seq_length is None:
                max_length = tokenizer.max_len
            else:
                max_length = args.max_seq_length

            batch_encoding = tokenizer.batch_encode_plus(
                [questions[row.qid] for row in df.itertuples()],
                max_length=max_length,
                pad_to_max_length=True,
                truncation=True,
            )

            features = []
            for i in tqdm(range(len(ys))):
                inputs = {k: v[i] for k, v in batch_encoding.items()}
                x = ((xs[i] - mean) / std).astype(float)
                inputs['retention_features'] = x.tolist()
                inputs['label'] = ys[i].item()
                features.append(RetentionInputFeatures(**inputs))
            self.features = features

            torch.save(self.features, cached_features_file)
            logger.info(f"Saving features into cached file {cached_features_file}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx) -> RetentionInputFeatures:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx]


def retention_data_collator(features: List[RetentionInputFeatures]) -> Dict[str, torch.Tensor]:
    # In this method we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    first = features[0]

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if hasattr(first, "label") and first.label is not None:
        if type(first.label) is int:
            labels = torch.tensor([f.label for f in features], dtype=torch.long)
        else:
            labels = torch.tensor([f.label for f in features], dtype=torch.float)
        batch = {"labels": labels}
    elif hasattr(first, "label_ids") and first.label_ids is not None:
        if type(first.label_ids[0]) is int:
            labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        else:
            labels = torch.tensor([f.label_ids for f in features], dtype=torch.float)
        batch = {"labels": labels}
    else:
        batch = {}

    # Handling of all other possible attributes.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in vars(first).items():
        if (
                k not in ('label', 'label_ids', 'retention_features')
                and v is not None
                and not isinstance(v, str)
        ):
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        elif k == 'retention_features':
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.float)

    return batch


class RetentionBertConfig(BertConfig):

    def __init__(self, retention_feature_size: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.retention_feature_size = retention_feature_size


class RetentionDistilBertConfig(DistilBertConfig):

    def __init__(self, retention_feature_size: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.retention_feature_size = retention_feature_size


class BertRetentionModel(BertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        retention_feature_size = config.retention_feature_size
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size + retention_feature_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @overrides
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        retention_features=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        if retention_features is not None:
            pooled_output = torch.cat((pooled_output, retention_features), axis=1)
            pooled_output = self.dropout(
                F.relu(
                    self.linear1(pooled_output)
                )
            )
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class DistilBertRetentionModel(DistilBertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        retention_feature_size = config.retention_feature_size
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.pre_classifier = nn.Linear(config.dim + retention_feature_size, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)

        self.init_weights()

    @overrides
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
        if retention_features is not None:
            pooled_output = torch.cat((pooled_output, retention_features), axis=1)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
