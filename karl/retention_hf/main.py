#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import Counter
from typing import Dict

import transformers
from transformers import (
    DistilBertTokenizerFast,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

from karl.config import settings
from .data import (  # noqa: F401
    RetentionInput,
    RetentionDataset,
    retention_data_collator,
    feature_fields,
)
from .model_distilbert import DistilBertRetentionModelConfig, DistilBertRetentionModel
from .model_bert import BertRetentionModelConfig, BertRetentionModel

transformers.logging.set_verbosity_info()

model_cls = {
    'distilbert': DistilBertRetentionModel,
    'bert': BertRetentionModel,
}
config_cls = {
    'distilbert': DistilBertRetentionModelConfig,
    'bert': BertRetentionModelConfig,
}
tokenizer_cls = {
    'distilbert': DistilBertTokenizerFast,
    'bert': BertTokenizerFast,
}
full_name = {
    'distilbert': 'distilbert-base-uncased',
    'bert': 'bert-base-uncased',
}


def compute_metrics(p: EvalPrediction) -> Dict:
    print(p.predictions)
    return {"accuracy": np.mean((p.predictions > 0.5) == p.label_ids)}


def train(model_name, output_dir=f'{settings.CODE_DIR}/output', fold='new_card', resume=None):
    retention_feature_size = 0 if fold == 'new_card' else len(feature_fields)
    config = config_cls[model_name](retention_feature_size=retention_feature_size)
    model = model_cls[model_name](config=config)
    tokenizer = tokenizer_cls[model_name].from_pretrained(full_name[model_name])
    train_dataset = RetentionDataset(settings.DATA_DIR, f'train_{fold}', tokenizer)
    test_dataset = RetentionDataset(settings.DATA_DIR, f'test_{fold}', tokenizer)
    training_args = TrainingArguments(
        output_dir=f'{output_dir}/retention_hf_{model_name}_{fold}',
        num_train_epochs=10,
        per_device_train_batch_size=8,
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
    trainer.train(resume)
    trainer.save_model()


def test(model_name, output_dir=f'{settings.CODE_DIR}/output', fold='new_card'):
    tokenizer = tokenizer_cls[model_name].from_pretrained(full_name[model_name])
    train_dataset = RetentionDataset(settings.DATA_DIR, f'train_{fold}', tokenizer)
    test_dataset = RetentionDataset(settings.DATA_DIR, f'test_{fold}', tokenizer)

    training_args = TrainingArguments(
        output_dir=f'{output_dir}/retention_hf_{model_name}_{fold}',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        learning_rate=2e-05,
    )
    model = model_cls[model_name].from_pretrained(training_args.output_dir)
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
    import sys
    train(model_name=sys.argv[1], fold='new_card')
    test(model_name=sys.argv[1], fold='new_card')
    train(model_name=sys.argv[1], fold='old_card')
    test(model_name=sys.argv[1], fold='old_card')
