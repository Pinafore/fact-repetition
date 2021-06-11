#!/usr/bin/env python
# coding: utf-8

import argparse
import random
import numpy as np
from collections import Counter
from typing import Dict

import torch
import transformers
from transformers import (
    DistilBertTokenizerFast,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve

from karl.config import settings
from .data import (  # noqa: F401
    RetentionInput,
    RetentionDataset,
    retention_data_collator,
    feature_fields,
)
from .model_distilbert import DistilBertRetentionModelConfig, DistilBertRetentionModel
from .model_bert import BertRetentionModelConfig, BertRetentionModel
from .model_norep import NorepRetentionModelConfig, NorepRetentionModel

transformers.logging.set_verbosity_info()

model_cls = {
    'distilbert': DistilBertRetentionModel,
    'bert': BertRetentionModel,
    'norep': NorepRetentionModel,
}
config_cls = {
    'distilbert': DistilBertRetentionModelConfig,
    'bert': BertRetentionModelConfig,
    'norep': NorepRetentionModelConfig,
}
tokenizer_cls = {
    'distilbert': DistilBertTokenizerFast,
    'bert': BertTokenizerFast,
    'norep': DistilBertTokenizerFast,
}
full_name = {
    'distilbert': 'distilbert-base-uncased',
    'bert': 'bert-base-uncased',
    'norep': 'distilbert-base-uncased',
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


def compute_metrics(p: EvalPrediction) -> Dict:
    predicted_labels = p.predictions > 0.5
    acc = accuracy_score(p.label_ids, predicted_labels)
    auc = roc_auc_score(p.label_ids, p.predictions)
    prob_true, prob_pred = calibration_curve(p.label_ids, p.predictions, n_bins=10)
    ece = np.mean(np.absolute(prob_true - prob_pred))
    return {
        "acc": acc,
        "auc": auc,
        "ece": ece,
    }


def train(
        model_name,
        output_dir=f'{settings.CODE_DIR}/output',
        fold='new_card',
        resume=None,
        seed=1,
):
    set_seed(seed)
    retention_feature_size = 0 if fold == 'new_card' else len(feature_fields)
    config = config_cls[model_name](retention_feature_size=retention_feature_size)
    model = model_cls[model_name](config=config)
    tokenizer = tokenizer_cls[model_name].from_pretrained(full_name[model_name])
    train_dataset = RetentionDataset(settings.DATA_DIR, f'train_{fold}', tokenizer)
    test_dataset = RetentionDataset(settings.DATA_DIR, f'test_{fold}', tokenizer)
    training_args = TrainingArguments(
        output_dir=f'{output_dir}/retention_hf_{model_name}_{fold}_{seed}',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        learning_rate=2e-05,
        save_steps=2000,
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


def test(model_name, output_dir=f'{settings.CODE_DIR}/output', fold='new_card', seed=1):
    tokenizer = tokenizer_cls[model_name].from_pretrained(full_name[model_name])
    train_dataset = RetentionDataset(settings.DATA_DIR, f'train_{fold}', tokenizer)
    test_dataset = RetentionDataset(settings.DATA_DIR, f'test_{fold}', tokenizer)

    training_args = TrainingArguments(
        output_dir=f'{output_dir}/retention_hf_{model_name}_{fold}_{seed}',
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
    print(f"***** Eval {model_name} {fold} {seed} *****")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=['distilbert', 'bert', 'norep'])
    parser.add_argument('--fold', choices=['new_card', 'old_card'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--resume')
    args = parser.parse_args()

    if args.train:
        train(model_name=args.model_name, fold=args.fold, seed=args.seed, resume=args.resume)
    test(model_name=args.model_name, fold=args.fold, seed=args.seed)
