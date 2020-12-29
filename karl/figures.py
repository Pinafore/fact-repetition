# %%
import os
import numpy as np
import altair as alt
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from karl.retention_hf.data import _get_user_features
from karl.db.session import SessionLocal, engine
from karl.config import settings
from karl.models import User

alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')


def save_chart_and_pdf(chart, path):
    chart.save(f'{path}.json')
    os.system(f'vl2vg {path}.json | vg2pdf > {path}.pdf')


def get_retention_features_df():
    session = SessionLocal()
    # gather features
    futures = []
    executor = ProcessPoolExecutor(
        mp_context=multiprocessing.get_context(settings.MP_CONTEXT),
        initializer=engine.dispose,
    )
    for user in session.query(User):
        if not user.id.isdigit() or len(user.records) == 0:
            continue
        futures.append(executor.submit(_get_user_features, user.id))

    features, labels = [], []
    for future in futures:
        f1, f2 = future.result()
        features.extend(f1)
        labels.extend(f2)

    df = []
    for feature, label in zip(features, labels):
        row = feature.__dict__
        row['response'] = label
        df.append(row)
    df = pd.DataFrame(df)
    return df


def figure_response_and_newness_vs_time(df, path):
    '''
    Break down of [new, old] x [positive, negative] over time.
    '''
    source = df.copy()
    source['n_minutes_spent_binned'] = pd.cut(source.n_minutes_spent, 20, labels=False)
    source['response_and_newness'] = (
        df.is_new_fact.transform(lambda x: 'New, ' if x else 'Old, ')
        + df.response.transform(lambda x: 'Positive' if x else 'Negative')
    )
    source = source[source.repetition_model.isin(['karl100', 'leitner', 'sm2'])]
    source = source.groupby(['response_and_newness', 'repetition_model', 'n_minutes_spent_binned']).size().to_frame('size').reset_index()

    return alt.Chart(source).mark_area().encode(
        alt.X('n_minutes_spent_binned:Q'),
        alt.Y('size:Q', stack='normalize'),
        color='response_and_newness'
    ).facet(
        column='repetition_model'
    )


def figure_recall_by_repetition_vs_time(
    df,
    path,
    facet_by='sm2_repetition',
    color_by='repetition_model',
    max_sm2_repetition=4,
):
    '''
    Recall rate broken down by number of repetition (to handle negative
    response, follow either Leitner box or SM2 repetition rules) over time.
    '''
    source = df.copy()
    source['n_minutes_spent_binned'] = pd.qcut(source.n_minutes_spent, 20, labels=False)
    source = source[source.repetition_model.isin(['karl100', 'leitner', 'sm2'])]
    source = source[source.sm2_repetition <= max_sm2_repetition]
    # first bin is very noisy
    source = source[source.n_minutes_spent_binned > 1]
    source = source.groupby(['n_minutes_spent_binned', 'user_id', facet_by, color_by])['response'].mean().to_frame('response').reset_index()

    line = alt.Chart().mark_line().encode(
        alt.X('n_minutes_spent_binned:Q'),
        alt.Y('mean(response):Q'),
        color=f'{color_by}:N',
    )
    band = alt.Chart().mark_errorband(extent='ci', opacity=0.3).encode(
        alt.X('n_minutes_spent_binned:Q'),
        alt.Y('response:Q'),
        color=f'{color_by}:N',
    )
    return alt.layer(
        line, band, data=source
    ).facet(
        column=facet_by,
    )
    # save_chart_and_pdf(chart, 'test')


# %%
# df = get_retention_features_df()
# df['n_minutes_spent'] = df.groupby('user_id')['elapsed_milliseconds'].cumsum() // 60000
# df.to_hdf(f'{settings.CODE_DIR}/figures.h5', key='df', mode='w')
df = pd.read_hdf(f'{settings.CODE_DIR}/figures.h5', 'df')
# %%
# source = df.loc[np.random.choice(df.index, 3000, replace=False)]
figure_recall_by_repetition_vs_time(df, '', color_by='sm2_repetition', facet_by='repetition_model')
