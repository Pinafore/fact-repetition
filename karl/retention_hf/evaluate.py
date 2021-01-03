#!/usr/bin/env python
# coding: utf-8

import os
import json
import altair as alt
import pandas as pd
from transformers import (
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
)

from karl.retention_hf.model import (
    DistilBertRetentionModel,
    compute_metrics,
)
from karl.retention_hf.data import (  # noqa: F401
    RetentionInput,
    RetentionDataset,
    get_retention_features_df,
    retention_data_collator,
)

from karl.config import settings
from karl.figures import figure_forgetting_curve, figure_recall_rate

alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')


def evaluate(output_dir=f'{settings.CODE_DIR}/output'):
    '''
    This evaluation focuses on *old* cards.
    1. compare the empirical forgetting curve and the predicted one.
    1. compare the empirical recall rate and the predicted one.
    '''
    folds = ['train_new_card', 'train_old_card', 'test_new_card', 'test_old_card']
    predictions = {}
    for fold in folds:
        prediction_path = f'{output_dir}/predictions_{fold}.json'
        if os.path.exists(prediction_path):
            predictions[fold] = json.load(open(prediction_path))
        else:
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            test_dataset = RetentionDataset(settings.DATA_DIR, fold, tokenizer)

            foold = '_'.join(fold.split('_')[1:])
            training_args = TrainingArguments(
                output_dir=f'{output_dir}/retention_hf_{foold}',
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
                train_dataset=test_dataset,
                eval_dataset=test_dataset,
                data_collator=retention_data_collator,
                compute_metrics=compute_metrics,
            )
            p = trainer.predict(test_dataset)
            print(fold, 'accuracy', compute_metrics(p))
            with open(prediction_path, 'w') as f:
                json.dump(p.predictions.tolist(), f)

            predictions[fold] = p.predictions.tolist()

    df_all = get_retention_features_df()
    df_new_card = df_all[df_all.is_new_fact == True]  # noqa: E712
    df_old_card = df_all[df_all.is_new_fact == False]  # noqa: E712
    df_by_fold = {
        'train_new_card': df_new_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[:int(x.user_id.size * 0.75)]).reset_index(),  # .drop(['level_0', 'level_1'], axis=1),
        'test_new_card': df_new_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[int(x.user_id.size * 0.75):]).reset_index(),  # .drop(['level_0', 'level_1'], axis=1),
        'train_old_card': df_old_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[:int(x.user_id.size * 0.75)]).reset_index(),  # .drop(['level_0', 'level_1'], axis=1),
        'test_old_card': df_old_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[int(x.user_id.size * 0.75):]).reset_index(),  # .drop(['level_0', 'level_1'], axis=1),
    }
    for fold in folds:
        df_by_fold[fold]['prediction'] = predictions[fold]
        if 'train' in fold:
            df_by_fold[fold] = df_by_fold[fold].rename(columns={'response': 'train_response'})
            cols = [x for x in df_by_fold[fold].columns if x not in ['prediction', 'train_response', 'test_response']]
            df_by_fold[fold] = df_by_fold[fold].melt(id_vars=cols, value_vars=['prediction', 'train_response'], var_name='type', value_name='value')
        else:
            df_by_fold[fold] = df_by_fold[fold].rename(columns={'response': 'test_response'})
            cols = [x for x in df_by_fold[fold].columns if x not in ['prediction', 'train_response', 'test_response']]
            df_by_fold[fold] = df_by_fold[fold].melt(id_vars=cols, value_vars=['prediction', 'test_response'], var_name='type', value_name='value')

    df_concat = pd.concat(list(df_by_fold.values()))
    # df_concat = df_concat.sample(frac=0.1)
    path = f'{settings.CODE_DIR}/figures_eval'
    figure_forgetting_curve(df_concat, path)
    figure_recall_rate(df_concat, path)
    figure_recall_rate(df_concat, user_id='463', path=path)
    figure_forgetting_curve(df_concat, user_id='463', path=path)


if __name__ == '__main__':
    evaluate(output_dir=f'{settings.CODE_DIR}/output')
