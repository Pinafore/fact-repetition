# %%
import os
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
    os.system(f'vl2vg {path}.json | vg2pdf > {path}.pdf')


def figure_composition(df, path):
    '''
    Break down of [new, old] x [positive, negative] vs amount of time.
    '''
    source = df.copy().drop('utc_date', axis=1)
    source['n_minutes_spent_binned'] = source['n_minutes_spent'].transform(lambda x: pd.qcut(x, 10))
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
    source['n_minutes_spent_binned'] = pd.qcut(source.n_minutes_spent, 20)
    source['n_minutes_spent_binned'] = source['n_minutes_spent_binned'].transform(lambda x: round(x.left / 60, 2))  # hours
    source = source[source.repetition_model.isin(['karl100', 'leitner', 'sm2'])]
    source = source[source.sm2_repetition <= max_sm2_repetition]
    source[groupby] = source.apply(lambda x: 'New' if x['is_new_fact'] else str(x[groupby]), axis=1)
    # first bin is very noisy
    source = source[source.n_minutes_spent_binned > 1]
    source = source.groupby(['n_minutes_spent_binned', 'user_id', 'repetition_model', groupby])['response'].mean().to_frame('response').reset_index()
    source = source.rename(columns={'n_minutes_spent_binned': 'n_hours_spent_binned'})

    selection = alt.selection_multi(fields=['repetition_model'], bind='legend')
    line = alt.Chart().mark_line().encode(
        alt.X('n_hours_spent_binned:Q', title='Hours'),
        alt.Y('mean(response):Q', title='Recall rate'),
        color=alt.Color('repetition_model:N', title='Repetition model'),
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2))
    )
    band = alt.Chart().mark_errorband(extent='ci').encode(
        alt.X('n_hours_spent_binned:Q', title='Hours'),
        alt.Y('mean(response):Q', title='Recall rate'),
        color=alt.Color('repetition_model:N', title='Repetition model'),
        opacity=alt.condition(selection, alt.value(0.3), alt.value(0.2))
    )
    chart = alt.layer(
        line, band, data=source
    ).properties(
        width=180,
        height=180,
    ).facet(
        # force new to appear first
        facet=alt.Facet(groupby, sort=['New']),
    ).add_selection(
        selection
    ).properties(
        title='Recall rate',
    ).configure_legend(
        labelFontSize=15,
    )
    save_chart_and_pdf(chart, f'{path}/recall_by_{groupby}')


def figure_user_recall_by_repetition(
    df: pd.DataFrame,
    user_id: str = None,
    path: str = None,
    groupby: str = 'sm2_repetition',
    max_sm2_repetition: int = 3,
):
    '''
    Recall rate broken down by number of repetition (to handle negative
    response, follow either Leitner box or SM2 repetition rules).
    '''
    source = df.copy().drop('utc_date', axis=1)
    source = source[source.user_id == user_id]
    source = source[source.sm2_repetition <= max_sm2_repetition]
    source[groupby] = source.apply(lambda x: 'New' if x['is_new_fact'] else str(x[groupby]), axis=1)
    # don't use qcut here??
    source['n_minutes_spent_binned'] = pd.qcut(source.n_minutes_spent, 20)
    source['n_minutes_spent_binned'] = source['n_minutes_spent_binned'].transform(lambda x: round(x.left / 60, 2))  # hours
    source = source.rename(columns={'n_minutes_spent_binned': 'n_hours_spent_binned'})

    line = alt.Chart().mark_line().encode(
        alt.X('n_hours_spent_binned:Q', title='Hours'),
        alt.Y('mean(response):Q', title='Recall rate'),
    )
    band = alt.Chart().mark_errorband(extent='ci').encode(
        alt.X('n_hours_spent_binned:Q', title='Hours'),
        alt.Y('response:Q', title='Recall rate'),
    )
    chart = alt.layer(
        line, band, data=source
    ).properties(
        width=180,
        height=180,
    ).facet(
        alt.Facet(f'{groupby}:N', sort=['New']),
    ).configure_legend(
        labelFontSize=15,
    ).properties(
        title=f'Recall rate of User {user_id}',
    )
    if path is not None:
        save_chart_and_pdf(chart, f'{path}/user_{user_id}_recall_by_{groupby}')
    else:
        return chart


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
    source = source[source.usercard_delta != 0]
    source = source[source.sm2_repetition <= max_sm2_repetition]

    source['usercard_delta_binned'] = source.groupby(groupby)['usercard_delta'].transform(lambda x: pd.qcut(x, q=10))
    source['usercard_delta_binned'] = source['usercard_delta_binned'].transform(lambda x: round(x.left / 3600, 2))  # hours

    line = alt.Chart().mark_line().encode(
        alt.X('usercard_delta_binned:Q', title='Hours'),
        alt.Y('mean(response):Q', title='Recall rate'),
    )
    band = alt.Chart().mark_errorband(extent='ci').encode(
        alt.X('usercard_delta_binned:Q', title='Hours'),
        alt.Y('response:Q', title='Recall rate'),
    )
    chart = alt.layer(
        band, line, data=source,
    ).properties(
        width=180, height=180,
    ).facet(
        groupby,
    ).properties(
        title='Forgetting curve' + ('' if user_id is None else f' of User {user_id}'),
    ).configure_legend(
        labelFontSize=15,
    )

    if path is None:
        return chart
    if user_id is None:
        save_chart_and_pdf(chart, f'{path}/forgetting_curve_by_{groupby}')
    else:
        save_chart_and_pdf(chart, f'{path}/user_{user_id}_forgetting_curve_by_{groupby}')


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
        'recall_by_repetition': figure_user_recall_by_repetition(source, user_id=user_id),
        'recall_vs_delta': figure_forgetting_curve(source, user_id=user_id),
    }


# %%
if __name__ == '__main__':
    df = get_retention_features_df()
    path = f'{settings.CODE_DIR}/figures'
    figure_composition(df, path)
    figure_recall_rate(df, path=path)
    figure_forgetting_curve(df, path=path)
    figure_user_recall_by_repetition(df, user_id='463', path=path)
    figure_forgetting_curve(df, user_id='463', path=path)
