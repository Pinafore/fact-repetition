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
from karl.figures import save_chart_and_pdf

alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')


def plot(
    source,
    x_axis,
    groupby,
):
    line = alt.Chart().mark_line().encode(
        alt.X(f'{x_axis}:Q', title='Hours'),
        alt.Y('mean(value):Q', title='Recall rate', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('type', title=None),
    )
    band = alt.Chart().mark_errorband(extent='ci').encode(
        alt.X(f'{x_axis}:Q', title='Hours'),
        alt.Y('value:Q', axis=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('type', title=None),
    )
    density = alt.Chart().transform_density(
        x_axis,
        groupby=[groupby, 'repetition_model'],
        as_=[x_axis, 'density'],
    ).mark_area(opacity=0.3, color='pink').encode(
        alt.X(f'{x_axis}:Q'),
        alt.Y('density:Q', axis=alt.Axis(title='Density', titleColor='pink')),
    )

    charts = []
    for val in sorted(source[groupby].unique()):
        charts.append(
            alt.vconcat(*(
                alt.layer(
                    line, band, density,
                    data=source,
                    title=f'{repetition_model}, rep={val}',
                ).transform_filter(
                    (alt.datum[groupby] == val)
                    & (alt.datum.repetition_model == repetition_model)
                ).resolve_scale(
                    y='independent', x='shared'
                ).properties(
                    width=200, height=200,
                )
                for repetition_model in source['repetition_model'].unique()
            ), spacing=30)
        )
    return alt.hconcat(*charts)


def figure_forgetting_curve(
    df: pd.DataFrame,
    path: str = None,
    user_id: str = None,
    groupby: str = 'sm2_repetition',
    max_sm2_repetition: int = 3,
):
    '''
    Recall vs delta broken down by # repetition
    '''
    source = df.copy().drop('utc_date', axis=1)

    if user_id is not None:
        source = source[source.user_id == user_id]
    else:
        source = source[source.repetition_model.isin(['karl100', 'leitner', 'sm2'])]
    source = source[source.usercard_delta != 0]
    source = source[source.sm2_repetition <= max_sm2_repetition]

    source['usercard_delta_binned'] = source.groupby(groupby)['usercard_delta'].transform(lambda x: pd.qcut(x, q=10))
    source['usercard_delta_binned'] = source['usercard_delta_binned'].transform(lambda x: round(x.left / 3600, 2))  # hours

    chart = plot(source, 'usercard_delta_binned', groupby)

    if path is None:
        return chart
    if user_id is None:
        save_chart_and_pdf(chart, f'{path}/eval_forgetting_curve')
    else:
        save_chart_and_pdf(chart, f'{path}/eval_user_{user_id}_forgetting_curve')


def figure_recall_rate(
    df,
    path,
    groupby: str = 'sm2_repetition',
    max_sm2_repetition: int = 3,
):
    '''
    Recall rate broken down by number of repetition (to handle negative
    response, follow either Leitner box or SM2 repetition rules).
    '''
    source = df.copy().drop('utc_date', axis=1)
    source = source[source.repetition_model.isin(['karl100', 'leitner', 'sm2'])]
    source = source[source.sm2_repetition <= max_sm2_repetition]
    source[groupby] = source.apply(lambda x: 'New' if x['is_new_fact'] else str(x[groupby]), axis=1)

    source['n_minutes_spent_binned'] = pd.qcut(source.n_minutes_spent, 20)
    source['n_minutes_spent_binned'] = source['n_minutes_spent_binned'].transform(lambda x: round(x.left / 60, 2))  # hours

    # first bin is very noisy
    # source = source[source['n_minutes_spent_binned'] > 1]
    # source = source.groupby(['n_minutes_spent_binned', 'user_id', 'repetition_model', 'type', groupby])['value'].mean().to_frame('value').reset_index()
    source = source.rename(columns={'n_minutes_spent_binned': 'n_hours_spent_binned'})

    chart = plot(source, 'n_hours_spent_binned', groupby)

    if path is not None:
        save_chart_and_pdf(chart, f'{path}/eval_recall_by_{groupby}')
    else:
        return chart


def evaluate():
    '''
    This evaluation focuses on *old* cards.
    1. compare the empirical forgetting curve and the predicted one.
    1. compare the empirical recall rate and the predicted one.
    '''
    folds = ['train_new_card', 'train_old_card', 'test_new_card', 'test_old_card']
    predictions = {}
    for fold in folds:
        prediction_path = f'{settings.CODE_DIR}/output/predictions_{fold}.json'
        if os.path.exists(prediction_path):
            predictions[fold] = json.load(open(prediction_path))
        else:
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            test_dataset = RetentionDataset(settings.DATA_DIR, fold, tokenizer)

            foold = '_'.join(fold.split('_')[1:])
            training_args = TrainingArguments(
                output_dir=f'{settings.CODE_DIR}/output/retention_hf_{foold}',
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
        'train_new_card': df_new_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[:int(x.user_id.size * 0.75)]).reset_index().drop(['level_0', 'level_1'], axis=1),
        'test_new_card': df_new_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[int(x.user_id.size * 0.75):]).reset_index().drop(['level_0', 'level_1'], axis=1),
        'train_old_card': df_old_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[:int(x.user_id.size * 0.75)]).reset_index().drop(['level_0', 'level_1'], axis=1),
        'test_old_card': df_old_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[int(x.user_id.size * 0.75):]).reset_index().drop(['level_0', 'level_1'], axis=1),
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
    path = f'{settings.CODE_DIR}/figures'
    figure_forgetting_curve(df_concat, path)
    figure_recall_rate(df_concat, path)


if __name__ == '__main__':
    evaluate()
