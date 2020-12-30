# %%
import os
import pytz
import altair as alt
import pandas as pd
import multiprocessing
from datetime import timedelta
from dateutil.parser import parse as parse_date
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


def figure_response_and_newness_over_time(df, path):
    '''
    Break down of [new, old] x [positive, negative] over time.
    '''
    source = df.copy().drop('utc_date', axis=1)
    source['n_minutes_spent_binned'] = source.groupby('repetition_model')['n_minutes_spent'].transform(lambda x: pd.cut(x, 10, labels=False, duplicates='drop'))
    # source['n_minutes_spent_binned'] = source['n_minutes_spent_binned'].transform(lambda x: round(x.left, 2))
    source['response_and_newness'] = (
        df.is_new_fact.transform(lambda x: 'New, ' if x else 'Old, ')
        + df.response.transform(lambda x: 'Positive' if x else 'Negative')
    )
    source = source[source.repetition_model.isin(['karl100', 'leitner', 'sm2'])]
    source = source.groupby(['response_and_newness', 'repetition_model', 'n_minutes_spent_binned']).size().to_frame('size').reset_index()

    selection = alt.selection_multi(fields=['response_and_newness'], bind='legend')
    chart = alt.Chart(source).mark_area().encode(
        alt.X('n_minutes_spent_binned:Q', axis=alt.Axis(title='Minutes')),
        alt.Y('size:Q', stack='normalize', axis=alt.Axis(title='Ratio')),
        color=alt.Color(
            'response_and_newness',
            legend=alt.Legend(title=None, orient='right'),
            scale=alt.Scale(
                domain=['New, Positive', 'New, Negative', 'Old, Positive', 'Old, Negative'],
                range=['#DB4437', '#ED9D97', '#4285F4', '#A0C3FF'],
            )
        ),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    ).add_selection(
        selection
    ).properties(
        width=180,
        height=180,
    ).facet(
        facet=alt.Facet('repetition_model', title=None),
    ).properties(
        title='Response and newness breakdown over time',
    ).configure_legend(
        labelFontSize=15,
    )
    save_chart_and_pdf(chart, f'{path}/response_and_newness_over_time')


def figure_recall_by_repetition_or_model_over_time(
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
    source = df.copy().drop('utc_date', axis=1)
    source['n_minutes_spent_binned'] = pd.qcut(source.n_minutes_spent, 20, labels=False)
    source = source[source.repetition_model.isin(['karl100', 'leitner', 'sm2'])]
    source = source[source.sm2_repetition <= max_sm2_repetition]
    # first bin is very noisy
    source = source[source.n_minutes_spent_binned > 1]
    source = source.groupby(['n_minutes_spent_binned', 'user_id', facet_by, color_by])['response'].mean().to_frame('response').reset_index()

    selection = alt.selection_multi(fields=[color_by], bind='legend')
    line = alt.Chart().mark_line().encode(
        alt.X('n_minutes_spent_binned:Q', title='Minutes spent on app'),
        alt.Y('mean(response):Q', title='Recall rate'),
        color=f'{color_by}:N',
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2))
    )
    band = alt.Chart().mark_errorband(extent='ci').encode(
        alt.X('n_minutes_spent_binned:Q'),
        alt.Y('response:Q'),
        color=f'{color_by}:N',
        opacity=alt.condition(selection, alt.value(0.3), alt.value(0.2))
    )
    chart = alt.layer(
        line, band, data=source
    ).properties(
        width=180,
        height=180,
    ).facet(
        facet=alt.Facet(facet_by, title=None),
    ).add_selection(
        selection
    ).properties(
        title=f'Recall rate broken down by {color_by} over time',
    ).configure_legend(
        labelFontSize=15,
    )
    save_chart_and_pdf(chart, f'{path}/recall_by_{color_by}_over_time')


def figure_user_recall_by_repetition_over_time(
    df: pd.DataFrame,
    user_id: str = None,
    path: str = None,
    color_by='sm2_repetition',
    max_sm2_repetition=4,
):
    '''
    Recall rate broken down by number of repetition (to handle negative
    response, follow either Leitner box or SM2 repetition rules) over time.
    '''
    source = df.copy().drop('utc_date', axis=1)
    source = source[source.user_id == user_id]
    source = source[source.sm2_repetition <= max_sm2_repetition]
    # don't use qcut here??
    source['n_minutes_spent_binned'] = pd.qcut(source.n_minutes_spent, 20, labels=False)

    selection = alt.selection_multi(fields=[color_by], bind='legend')
    line = alt.Chart().mark_line().encode(
        alt.X('n_minutes_spent_binned:Q'),
        alt.Y('mean(response):Q'),
        color=f'{color_by}:N',
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2))
    )
    band = alt.Chart().mark_errorband(extent='ci').encode(
        alt.X('n_minutes_spent_binned:Q'),
        alt.Y('response:Q'),
        color=f'{color_by}:N',
        opacity=alt.condition(selection, alt.value(0.3), alt.value(0.2))
    )
    chart = alt.layer(
        line, band, data=source
    ).add_selection(
        selection
    ).properties(
        width=180,
        height=180,
        title=f'Recall rate of User {user_id} by {color_by} over time',
    ).configure_legend(
        labelFontSize=15,
    )
    if path is not None:
        save_chart_and_pdf(chart, f'{path}/user_{user_id}_recall_by_{color_by}_over_time')
    else:
        return chart


def figure_recall_vs_delta(
    df,
    path,
    groupby='sm2_repetition',
    max_sm2_repetition=4,
):
    '''
    Recall vs delta broken down by # repetition
    '''
    source = df.copy().drop('utc_date', axis=1)
    source = source[source.usercard_delta != 0]
    source = source[source.sm2_repetition <= max_sm2_repetition]

    source['usercard_delta_binned'] = source.groupby(groupby)['usercard_delta'].transform(lambda x: pd.qcut(x, q=10, duplicates='drop'))
    source['usercard_delta_binned'] = source['usercard_delta_binned'].transform(lambda x: round(x.left / 3600, 2))  # hours

    line = alt.Chart().mark_line().encode(
        alt.X('usercard_delta_binned:Q', title='Hours'),
        alt.Y('mean(response):Q', title='Recall rate'),
    )
    band = alt.Chart().mark_errorband(extent='ci').encode(
        alt.X('usercard_delta_binned:Q', title='Hours'),
        alt.Y('response:Q', title='Recall rate'),
    )
    # density = alt.Chart().transform_density(
    #     'usercard_delta_binned:N',
    #     groupby=[groupby],
    #     as_=['usercard_delta_binned', 'density'],
    # ).mark_area(opacity=0.3).encode(
    #     alt.X('usercard_delta_binned:N', title='Hours'),
    #     alt.Y('density:Q', title=None),
    # )
    chart = alt.layer(
        band, line, data=source,
    ).properties(
        width=180,
        height=180,
    ).facet(
        column=groupby,
    ).properties(
        title=f'Recall rate vs Delta broken down by {groupby} over time',
    ).configure_legend(
        labelFontSize=15,
    )
    save_chart_and_pdf(chart, f'{path}/recall_vs_delta_by_{groupby}')


def figure_user_recall_vs_delta(
    df: pd.DataFrame,
    user_id: str,
    path: str = None,
    groupby='sm2_repetition',
):
    '''
    Recall vs delta broken down by # repetition
    '''
    source = df.copy().drop('utc_date', axis=1)
    source = source[source.user_id == user_id]
    source = source[source.usercard_delta != 0]
    # create bins from each group
    source['usercard_delta_binned'] = source.groupby(groupby)['usercard_delta'].transform(lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop'))
    # alternatively, create bins from all examples
    # source['usercard_delta_binned'] = pd.qcut(source.usercard_delta, q=20, labels=False)

    line = alt.Chart().mark_line().encode(
        alt.X('usercard_delta_binned:Q'),
        alt.Y('mean(response):Q'),
    )
    band = alt.Chart().mark_errorband(extent='ci').encode(
        alt.X('usercard_delta_binned:Q'),
        alt.Y('response:Q'),
    )
    chart = alt.layer(
        band, line, data=source
    ).properties(
        width=180,
        height=180,
    ).facet(
        column=groupby,
    ).properties(
        title=f'Recall rate of User {user_id} vs Delta broken down by {groupby} over time',
    ).configure_legend(
        labelFontSize=15,
    )
    if path is not None:
        save_chart_and_pdf(chart, f'{path}/user_{user_id}_recall_vs_delta_by_{groupby}')
    else:
        return chart


def get_user_charts(
    df: pd.DataFrame,
    user_id: str,
    deck_id: str = None,
    date_start: str = '2020-08-23',
    date_end: str = '2024-08-23',
):
    date_start = parse_date(date_start).astimezone(pytz.utc).date()
    date_end = parse_date(date_end).astimezone(pytz.utc).date()
    source = df[df.user_id == user_id]
    if deck_id is not None:
        source = source[source.deck_id == deck_id]
    source = source[source.utc_date >= date_start]
    source = source[source.utc_date <= date_end]
    return {
        'recall_by_repetition_over_time': figure_user_recall_by_repetition_over_time(source, user_id=user_id),
        'recall_vs_delta': figure_user_recall_vs_delta(source, user_id=user_id),
    }


# %%
if __name__ == '__main__':
    # df = get_retention_features_df()
    # df['n_minutes_spent'] = df.groupby('user_id')['elapsed_milliseconds'].cumsum() // 60000
    # df.to_hdf(f'{settings.CODE_DIR}/figures.h5', key='df', mode='w')
    df = pd.read_hdf(f'{settings.CODE_DIR}/figures.h5', 'df')

    path = f'{settings.CODE_DIR}/figures'
    figure_response_and_newness_over_time(df, path)
    # figure_recall_by_repetition_or_model_over_time(df, path=path, facet_by='sm2_repetition', color_by='repetition_model')
    # figure_recall_by_repetition_or_model_over_time(df, path=path, facet_by='repetition_model', color_by='sm2_repetition')
    # figure_recall_vs_delta(df, path=path, groupby='sm2_repetition', max_sm2_repetition=4)
    # figure_user_recall_by_repetition_over_time(df, user_id='463', path=path, max_sm2_repetition=2)
    # figure_user_recall_vs_delta(df, user_id='463', path=path, groupby='sm2_repetition')
