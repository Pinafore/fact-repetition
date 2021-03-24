#!/usr/bin/env python
# coding: utf-8
import pytz
import altair as alt
import pandas as pd
from dateutil.parser import parse as parse_date

from karl.retention_hf.data import get_retention_features_df
from karl.config import settings

alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')


def save_chart_and_pdf(chart, path):
    chart.save(f'{path}.json')
    # os.system(f'vl2vg {path}.json | vg2pdf > {path}.pdf')


def figure_composition(df, path):
    '''
    Break down of [new, old] x [positive, negative] vs amount of time.
    '''
    source = df.copy().drop('utc_date', axis=1)
    source['n_minutes_spent_binned'] = source['n_minutes_spent'].transform(lambda x: pd.qcut(x, 10, duplicates='drop'))
    source['n_minutes_spent_binned'] = source['n_minutes_spent_binned'].transform(lambda x: round(x.left / 60, 2))
    source['response_and_newness'] = (
        df.is_new_fact.transform(lambda x: 'New, ' if x else 'Old, ')
        + df.response.transform(lambda x: 'Positive' if x else 'Negative')
    )
    source = source[source.repetition_model.isin(['karl100', 'leitner', 'sm2'])]
    source = source.groupby(['response_and_newness', 'repetition_model', 'n_minutes_spent_binned']).size().to_frame('size').reset_index()
    source['cumsum'] = source.groupby(['response_and_newness', 'repetition_model'])['size'].cumsum()

    selection = alt.selection_multi(fields=['response_and_newness'], bind='legend')
    chart = alt.Chart(source).mark_area().encode(
        alt.X('n_minutes_spent_binned:Q', axis=alt.Axis(title='Hours')),
        alt.Y('cumsum:Q', stack='normalize', axis=alt.Axis(title='Ratio')),
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
        title='Composition of cards studied by users',
    ).configure_legend(
        labelFontSize=15,
    )
    save_chart_and_pdf(chart, f'{path}/composition')


def plot(
    source,
    x_axis,
    groupby,
):
    if 'type' in source:
        line = alt.Chart().mark_line().encode(
            alt.X(f'{x_axis}:Q', title='Hours'),
            alt.Y('mean(value):Q', title='Recall rate', scale=alt.Scale(domain=[0, 1])),
            color=(alt.Color('type', title=None)),
        )
        band = alt.Chart().mark_errorband(extent='ci').encode(
            alt.X(f'{x_axis}:Q', title='Hours'),
            alt.Y('value:Q', axis=None, scale=alt.Scale(domain=[0, 1])),
            color=(alt.Color('type', title=None)),
        )
    else:
        line = alt.Chart().mark_line().encode(
            alt.X(f'{x_axis}:Q', title='Hours'),
            alt.Y('mean(value):Q', title='Recall rate', scale=alt.Scale(domain=[0, 1])),
        )
        band = alt.Chart().mark_errorband(extent='ci').encode(
            alt.X(f'{x_axis}:Q', title='Hours'),
            alt.Y('value:Q', axis=None, scale=alt.Scale(domain=[0, 1])),
        )
    density = alt.Chart().transform_density(
        x_axis,
        groupby=[groupby, 'repetition_model'],
        as_=[x_axis, 'density'],
    ).mark_area(opacity=0.2, color='gray').encode(
        alt.X(f'{x_axis}:Q'),
        alt.Y('density:Q', axis=alt.Axis(title='Density', titleColor='gray')),
    )

    charts = []
    groupby_vals = sorted(source[groupby].unique())
    if 'New' in groupby_vals:
        groupby_vals = ['New'] + groupby_vals[:-1]  # move new to front
    for val in groupby_vals:
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
                for repetition_model in sorted(source['repetition_model'].unique())
            ), spacing=30)
        )
    return alt.hconcat(*charts)


def figure_forgetting_curve(
    df: pd.DataFrame,
    path: str = None,
    user_id: str = None,
    groupby: str = 'sm2_repetition',
    max_repetition: int = 3,
    repetition_models: list = ['karl100', 'leitner', 'sm2'],
):
    '''
    Recall vs delta broken down by # repetition
    '''
    source = df.copy().drop('utc_date', axis=1)
    if user_id is not None:
        source = source[source.user_id == user_id]
    else:
        source = source[source.repetition_model.isin(repetition_models)]
    source = source[source.usercard_delta != 0]
    source = source[source[groupby] <= max_repetition]

    source['usercard_delta_binned'] = source.groupby(groupby)['usercard_delta'].transform(lambda x: pd.qcut(x, q=10, duplicates='drop'))
    source['usercard_delta_binned'] = source['usercard_delta_binned'].transform(lambda x: round(x.left / 3600, 2))

    if 'response' in source:
        source = source.rename(columns={'response': 'value'})
    chart = plot(source, 'usercard_delta_binned', groupby)

    if path is None:
        return chart
    if user_id is None:
        save_chart_and_pdf(chart, f'{path}/forgetting_curve')
    else:
        save_chart_and_pdf(chart, f'{path}/user_{user_id}_forgetting_curve')


def figure_recall_rate(
    df: pd.DataFrame,
    path: str = None,
    user_id: str = None,
    groupby: str = 'sm2_repetition',
    max_repetition: int = 3,
    repetition_models: list = ['karl100', 'leitner', 'sm2'],
):
    '''
    Recall rate broken down by number of repetition (to handle negative
    response, follow either Leitner box or SM2 repetition rules).
    '''
    source = df.copy().drop('utc_date', axis=1)
    if user_id is not None:
        source = source[source.user_id == user_id]
    else:
        source = source[source.repetition_model.isin(repetition_models)]
    source = source[source[groupby] <= max_repetition]

    source[groupby] = source.apply(lambda x: 'New' if x['is_new_fact'] else str(x[groupby]), axis=1)

    source['n_minutes_spent_binned'] = pd.qcut(source.n_minutes_spent, 20, duplicates='drop')
    source['n_minutes_spent_binned'] = source['n_minutes_spent_binned'].transform(lambda x: round(x.left / 60, 2))
    source = source.rename(columns={'n_minutes_spent_binned': 'n_hours_spent_binned'})

    if 'response' in source:
        source = source.rename(columns={'response': 'value'})

    chart = plot(source, 'n_hours_spent_binned', groupby)

    if path is None:
        return chart
    if user_id is None:
        save_chart_and_pdf(chart, f'{path}/recall_rate')
    else:
        save_chart_and_pdf(chart, f'{path}/user_{user_id}_recall_rate')


def figure_karl100_vs_karl85(
    df: pd.DataFrame,
    path: str = None,
    user_id: str = None,
    groupby: str = 'sm2_repetition',
    max_repetition: int = 3,
):
    source = df.copy().drop('utc_date', axis=1)
    source = source[source.repetition_model.isin(['karl100', 'karl85'])]
    source = source[source[groupby] <= max_repetition]

    source[groupby] = source.apply(lambda x: 'New' if x['is_new_fact'] else str(x[groupby]), axis=1)

    source['n_minutes_spent_binned'] = pd.qcut(source.n_minutes_spent, 20, duplicates='drop')
    source['n_minutes_spent_binned'] = source['n_minutes_spent_binned'].transform(lambda x: round(x.left / 60, 2))
    source = source.rename(columns={'n_minutes_spent_binned': 'n_hours_spent_binned'})

    if 'response' in source:
        source = source.rename(columns={'response': 'value'})

    chart = plot(source, 'n_hours_spent_binned', groupby)

    if path is None:
        return chart
    save_chart_and_pdf(chart, f'{path}/karl100_vs_karl85')


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
        'recall_by_repetition': figure_recall_rate(source, user_id=user_id),
        'recall_vs_delta': figure_forgetting_curve(source, user_id=user_id),
    }


if __name__ == '__main__':
    df = get_retention_features_df()
    path = f'{settings.CODE_DIR}/figures_stat'
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

    # figure_composition(df, path)
    # figure_recall_rate(df, path=path)
    # figure_forgetting_curve(df, path=path)
    # figure_recall_rate(df, user_id='463', path=path)
    # figure_forgetting_curve(df, user_id='463', path=path)

    figure_karl100_vs_karl85(df, path)
