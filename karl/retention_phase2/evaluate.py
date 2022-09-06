#!/usr/bin/env python
# coding: utf-8

import os
import json
import altair as alt
import pandas as pd
from pathlib import Path

from transformers import TrainingArguments, Trainer

from karl.retention_hf.data import (  # noqa: F401
    RetentionInput,
    RetentionDataset,
    get_retention_features_df,
    retention_data_collator,
)

from karl.config import settings
from karl.figures import figure_forgetting_curve, figure_recall_rate
from karl.retention_hf.main import compute_metrics, model_cls, tokenizer_cls, full_name

alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')


def evaluate(model_names=['distilbert', 'bert'], output_dir=f'{settings.CODE_DIR}/output'):
    '''
    This evaluation focuses on *old* cards.
    1. compare the empirical forgetting curve and the predicted one.
    1. compare the empirical recall rate and the predicted one.
    '''
    # figures_dir = f'{settings.CODE_DIR}/figures_eval_all'
    figures_dir = '/fs/www-users/shifeng/figures_eval_all'

    prediction_by_model = {}
    for model_name in model_names:
        prediction_root_dir = f'{output_dir}/retention_hf_{model_name}_predictions'
        Path(prediction_root_dir).mkdir(parents=True, exist_ok=True)
        Path(figures_dir).mkdir(parents=True, exist_ok=True)

        prediction_by_model[model_name] = {}
        for fold in ['train_new_card', 'train_old_card', 'test_new_card', 'test_old_card']:
            foold = '_'.join(fold.split('_')[1:])
            model_output_dir = f'{output_dir}/retention_hf_{model_name}_{foold}'
            prediction_dir = f'{prediction_root_dir}/predictions_{fold}.json'
            if os.path.exists(prediction_dir):
                prediction_by_model[model_name][fold] = json.load(open(prediction_dir))
            else:
                tokenizer = tokenizer_cls[model_name].from_pretrained(full_name[model_name])
                test_dataset = RetentionDataset(settings.DATA_DIR, fold, tokenizer)

                training_args = TrainingArguments(
                    output_dir=model_output_dir,
                    num_train_epochs=10,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=64,
                    learning_rate=2e-05,
                )
                model = model_cls[model_name].from_pretrained(training_args.output_dir)
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
                with open(prediction_dir, 'w') as f:
                    json.dump(p.predictions.tolist(), f)
                prediction_by_model[model_name][fold] = p.predictions.tolist()

    df_all = get_retention_features_df()
    df_new_card = df_all[df_all.is_new_fact == True]  # noqa: E712
    df_old_card = df_all[df_all.is_new_fact == False]  # noqa: E712
    df_by_fold = {
        'train_new_card': df_new_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[:int(x.user_id.size * 0.75)]).reset_index(),
        'test_new_card': df_new_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[int(x.user_id.size * 0.75):]).reset_index(),
        'train_old_card': df_old_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[:int(x.user_id.size * 0.75)]).reset_index(),
        'test_old_card': df_old_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[int(x.user_id.size * 0.75):]).reset_index(),
    }
    for fold in ['train_new_card', 'train_old_card']:
        df_by_fold[fold] = df_by_fold[fold].rename(columns={'response': 'train_response'})
        value_vars = ['train_response']
        id_vars = [x for x in df_by_fold[fold].columns if x not in value_vars]
        df_by_fold[fold] = df_by_fold[fold].melt(id_vars=id_vars, value_vars=value_vars, var_name='type', value_name='value')

    for fold in ['test_new_card', 'test_old_card']:
        df_by_fold[fold] = df_by_fold[fold].rename(columns={'response': 'test_response'})
        value_vars = ['test_response']
        for model_name in model_names:
            df_by_fold[fold][model_name] = prediction_by_model[model_name][fold]
            value_vars.append(model_name)
        id_vars = [x for x in df_by_fold[fold].columns if x not in value_vars]
        df_by_fold[fold] = df_by_fold[fold].melt(id_vars=id_vars, value_vars=value_vars, var_name='type', value_name='value')

    df_concat = pd.concat(list(df_by_fold.values()))
    figure_forgetting_curve(df_concat, figures_dir, max_repetition=2)
    figure_recall_rate(df_concat, figures_dir)
    figure_recall_rate(df_concat, user_id='463', path=figures_dir)
    figure_forgetting_curve(df_concat, user_id='463', path=figures_dir)


if __name__ == '__main__':
    evaluate()
