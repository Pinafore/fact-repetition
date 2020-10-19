# %%
import os
import json
import bisect
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from pathlib import Path
from pandas.api.types import CategoricalDtype
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict
import altair as alt
from altair.expr import datum
alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')

from karl.new_util import User, Record, parse_date, theme_fs
from karl.web import get_sessions


def save_chart_and_pdf(chart, path):
    chart.save(f'{path}.json')
    os.system(f'vl2vg {path}.json | vg2pdf > {path}.pdf')


def get_record_df(session):
    '''Gather records into a DataFrame'''
    user_start_date = {}  # user_id -> first day of study
    correct_on_first_try = {}  # user_id -> {fact_id -> bool}
    for user in tqdm(session.query(User), total=session.query(User).count()):
        if len(user.records) > 0:
            user_start_date[user.user_id] = user.records[0].date.date()
        correct_on_first_try[user.user_id] = {}
        for record in user.records:
            if record.fact_id in correct_on_first_try[user.user_id]:
                continue
            correct_on_first_try[user.user_id][record.fact_id] = record.response

    date_start = session.query(Record).order_by(Record.date).first().date.date()
    date_end = session.query(Record).order_by(Record.date.desc()).first().date.date()

    rows = []
    for user in tqdm(session.query(User), total=session.query(User).count()):
        if len(user.records) == 0:
            continue

        last_record = session.query(Record).\
            filter(Record.user_id == user.user_id).\
            filter(Record.date <= date_end).\
            order_by(Record.date.desc()).first()
        if last_record is None:
            continue

        for record in session.query(Record).\
            filter(Record.user_id == user.user_id).\
            filter(Record.date >= date_start).\
            filter(Record.date <= date_end):
            elapsed_seconds = record.elapsed_milliseconds_text / 1000
            elapsed_seconds += record.elapsed_milliseconds_answer / 1000
            elapsed_minutes = elapsed_seconds / 60
            leitner_box = json.loads(record.user_snapshot)['leitner_box']
            rows.append({
                'record_id': record.record_id,
                'user_id': user.user_id,
                'fact_id': record.fact_id,
                'repetition_model': json.loads(record.scheduler_snapshot)['repetition_model'],
                'is_new_fact': record.is_new_fact,
                'result': record.response,
                'datetime': record.date,
                'elapsed_minutes': elapsed_minutes,
                'user_start_date': user_start_date[user.user_id],
                'is_known_fact': correct_on_first_try[user.user_id][record.fact_id],
                'leitner_box': leitner_box.get(record.fact_id, 0),
            })
    return pd.DataFrame(rows).sort_values('datetime', axis=0)


def get_processed_df(session):
    '''Computer varoius x-axis and metrics'''

    date_start = session.query(Record).order_by(Record.date).first().date.date()
    date_end = session.query(Record).order_by(Record.date.desc()).first().date.date()

    df = get_record_df(session)
    df['date'] = df['datetime'].apply(lambda x: x.date())
    '''Compute x-axis'''
    # number of total facts shown since start
    df['n_facts_shown'] = df.groupby('user_id').cumcount() + 1
    # number of days since start
    df['n_days_since_start'] = (df.date - df.user_start_date).dt.days
    # number of total minutes
    df['n_minutes_spent'] = df.groupby('user_id')['elapsed_minutes'].cumsum()

    def func(bins):
        def find_bin(n):
            idx = bisect.bisect(bins, n)
            return bins[min(len(bins) - 1, idx)]
        return find_bin

    # bin n_facts_shown
    n_facts_bin_size = 10  # facts
    n_bins = (df['n_facts_shown'].max()) // n_facts_bin_size + 1
    n_facts_bins = [i * n_facts_bin_size for i in range(n_bins)]
    df['n_facts_shown_binned'] = df.n_facts_shown.apply(func(n_facts_bins))

    # bin n_days_since_start
    n_days_bin_size = 3  # days
    n_bins = (df.n_days_since_start.max()) // n_days_bin_size + 1
    n_days_bins = [i * n_days_bin_size for i in range(n_bins)]
    df['n_days_since_start_binned'] = df.n_days_since_start.apply(func(n_days_bins))

    # bin date
    date_bin_size = 3  # days
    n_bins = (date_end - date_start).days // date_bin_size + 1
    date_bins = [date_start + i * timedelta(days=date_bin_size) for i in range(n_bins)]
    df['date_binned'] = df.date.apply(func(date_bins))

    # bin n_minutes_spent
    n_minutes_bin_size = 60  # minutes
    n_bins = int((df.n_minutes_spent.max()) // n_minutes_bin_size + 1)
    n_minutes_bins = [i * n_minutes_bin_size for i in range(n_bins)]
    df['n_minutes_spent_binned'] = df.n_facts_shown.apply(func(n_minutes_bins))

    df.date = df.date.astype(np.datetime64)
    df.datetime = df.datetime.astype(np.datetime64)
    df.date_binned = df.date_binned.astype(np.datetime64)

    '''Compute derivative metrics'''
    df['n_new_facts_correct'] = df.groupby('user_id', group_keys=False).apply(lambda x: x.is_new_fact & x.result)
    df['n_new_facts_wrong'] = df.groupby('user_id', group_keys=False).apply(lambda x: x.is_new_fact & ~x.result)
    df['n_old_facts_correct'] = df.groupby('user_id', group_keys=False).apply(lambda x: ~x.is_new_fact & x.result)
    df['n_old_facts_wrong'] = df.groupby('user_id', group_keys=False).apply(lambda x: ~x.is_new_fact & ~x.result)
    df['n_new_facts_correct_csum'] = df.groupby('user_id')['n_new_facts_correct'].cumsum()
    df['n_new_facts_wrong_csum'] = df.groupby('user_id')['n_new_facts_wrong'].cumsum()
    df['n_old_facts_correct_csum'] = df.groupby('user_id')['n_old_facts_correct'].cumsum()
    df['n_old_facts_wrong_csum'] = df.groupby('user_id')['n_old_facts_wrong'].cumsum()
    df['ratio_new_correct_vs_all'] = df.n_new_facts_correct_csum / df.n_facts_shown
    df['ratio_new_wrong_vs_all'] = df.n_new_facts_wrong_csum / df.n_facts_shown
    df['ratio_old_correct_vs_all'] = df.n_old_facts_correct_csum / df.n_facts_shown
    df['ratio_old_wrong_vs_all'] = df.n_old_facts_wrong_csum / df.n_facts_shown

    df[f'initial_O_'] = df.apply(lambda x: (x.leitner_box == 0) & x.result, axis=1)
    df[f'initial_X_'] = df.apply(lambda x: (x.leitner_box == 0) & (~x.result), axis=1)
    df[f'initial_O'] = df.groupby('user_id')[f'initial_O_'].cumsum()
    df[f'initial_X'] = df.groupby('user_id')[f'initial_X_'].cumsum()
    progress_names = [f'initial_O', f'initial_X']
    for i in df.leitner_box.unique():
        if i > 0:
            df[f'level_{i}_O_'] = df.apply(lambda x: (x.leitner_box == i) & x.result & (~x.is_known_fact), axis=1)
            df[f'level_{i}_X_'] = df.apply(lambda x: (x.leitner_box == i) & (~x.result) & (~x.is_known_fact), axis=1)
            df[f'level_{i}_O'] = df.groupby('user_id')[f'level_{i}_O_'].cumsum()
            df[f'level_{i}_X'] = df.groupby('user_id')[f'level_{i}_X_'].cumsum()
            progress_names += [f'level_{i}_O', f'level_{i}_X']
    return df

def figure_new_old_successful_failed(
    df: pd.DataFrame,
    output_path: str,
):
    ''' breakdown by [new, old] x [successful, failed] '''
    x_axis_name = 'n_minutes_spent_binned'
    metrics = [
        'ratio_new_correct_vs_all',
        'ratio_new_wrong_vs_all',
        'ratio_old_wrong_vs_all',
        'ratio_old_correct_vs_all',
    ]
    source = df.groupby(['repetition_model', x_axis_name]).mean().reset_index()
    source = pd.melt(
        source ,
        id_vars=['repetition_model', x_axis_name],
        value_vars=metrics,
        var_name='name',
        value_name='value',
    )
    source = source[source.repetition_model.isin(['karl100', 'leitner', 'sm2'])]
    source = source.replace({
        'name': {
            'ratio_new_correct_vs_all': 'New, Successful',
            'ratio_old_correct_vs_all': 'Old, Successful',
            'ratio_new_wrong_vs_all': 'New, Failed',
            'ratio_old_wrong_vs_all': 'Old, Failed',
        },
        'repetition_model': {
            'karl100': 'KARL',
            'leitner': 'Leitner',
            'sm2': 'SM-2',
        }
    })

    selection = alt.selection_multi(fields=['name'], bind='legend')

    chart = alt.Chart(source).mark_area().encode(
        alt.X('n_minutes_spent_binned', axis=alt.Axis(title='Minutes spent on app')),
        alt.Y('sum(value)', axis=alt.Axis(title='Ratio')),
        color=alt.Color(
            'name',
            legend=alt.Legend(title=None, orient='right'),
            # scale=alt.Scale(scheme='dark2'),
            scale=alt.Scale(
                domain=['New, Successful', 'New, Failed', 'Old, Successful', 'Old, Failed'],
                range=['#DB4437', '#ED9D97', '#4285F4', '#A0C3FF'],
            )
        ),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    ).add_selection(
        selection
    ).properties(
        width=180,
        height=180
    ).facet(
        facet=alt.Facet('repetition_model', title=None),
        # columns=2
    ).configure_legend(
        labelFontSize=15,
    )

    # save_chart_and_pdf(chart, f'figures/new_old_correct_wrong')
    chart.save(f'{output_path}/new_old_correct_wrong.json')
    # chart.save('test.json')


def figure_model_level_vs_effort(
    df: pd.DataFrame,
    output_path: str,
):
    '''repetition model level vs effort'''
    x_axis_name = 'n_minutes_spent_binned'

    progress_names = [f'initial_O', f'initial_X']
    for i in df.leitner_box.unique():
        if i > 0:
            progress_names += [f'level_{i}_O', f'level_{i}_X']

    source = pd.melt(
        df,
        id_vars=[x_axis_name, 'repetition_model', 'user_id'],
        value_vars=progress_names,
        var_name='name',
        value_name='value',
    )

    source = source[source.repetition_model.isin(['karl100', 'leitner', 'sm2'])]

    source['type'] = source.name.apply(lambda x: 'Successful' if x[-1] == 'O' else 'Failed')
    source['level'] = source.name.apply(lambda x: x[:-2])
    source = source.groupby([x_axis_name, 'repetition_model', 'name', 'type', 'level', 'user_id']).mean().reset_index()
    source = source.groupby([x_axis_name, 'repetition_model', 'name', 'type', 'level']).agg(['mean', 'std']).reset_index()
    source.columns = [l1 if not l2 else l2 for l1, l2 in source.columns]
    source['min'] = source['mean'] - source['std'] / 2
    source['max'] = source['mean'] + source['std'] / 2

    source = source.replace({
        'level': {
            'initial': 'Initial',
            'level_1': 'Level.0',
            'level_2': 'Level.1',
            'level_3': 'Level.2',
            'level_4': 'Level.3',
            'level_5': 'Level.4',
            'level_6': 'Level.5',
            'level_7': 'Level.6',
            'level_8': 'Level.7',
            'level_9': 'Level.8',
        },
        'repetition_model': {
            'karl100': 'KARL',
            'leitner': 'Leitner',
            'sm2': 'SM-2',
        }
    })

    selection = alt.selection_multi(fields=['level'], bind='legend')

    line = alt.Chart().mark_line().encode(
        alt.X(x_axis_name, title='Minutes spent on app'),
        alt.Y('mean', title='Number of flashcards'),
        strokeDash=alt.StrokeDash('type', title='Result'),
        color=alt.Color('level', title='Level'),
        size=alt.condition(selection, alt.value(3), alt.value(1)),
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2))
    )
    band = alt.Chart().mark_area().encode(
        x=x_axis_name,
        y='min',
        y2='max',
        stroke=alt.StrokeDash('type', title='Result'),
        color=alt.Color('level', title='Level'),
        opacity=alt.condition(selection, alt.value(0.5), alt.value(0.2))
    )
    chart = alt.layer(
        band,
        line,
        data=source
    ).properties(
        width=180,
        height=180
    ).facet(
        facet=alt.Facet('repetition_model', title=None),
    ).add_selection(
        selection
    ).configure_legend(
        labelFontSize=15,
    )

    # save_chart_and_pdf(chart, 'figures/repetition_model_study_reports_all')
    chart.save(f'{output_path}/repetition_model_level_vs_effort.json')
    # chart.save('test.json')


def figure_model_level_ratio(
    df: pd.DataFrame,
    output_path: str,
):
    '''repetition model level ratio'''
    x_axis_name = 'n_minutes_spent_binned'

    progress_names = [f'initial_O', f'initial_X']
    for i in df.leitner_box.unique():
        if i > 0:
            progress_names += [f'level_{i}_O', f'level_{i}_X']

    source = pd.melt(
        df,
        id_vars=[x_axis_name, 'datetime', 'repetition_model', 'user_id'],
        value_vars=progress_names,
        var_name='name',
        value_name='value',
    )

    source = source[source.repetition_model.isin(['karl100', 'leitner', 'sm2'])]

    source['type'] = source.name.apply(lambda x: 'Successful' if x[-1] == 'O' else 'Failed')
    source['level'] = source.name.apply(lambda x: x[:-2])

    df_left = source[source.type == 'Successful'].drop(['name'], axis=1)
    df_right = source[source.type == 'Failed'].drop(['name'], axis=1)
    source = pd.merge(df_left, df_right, how='left', on=[
        x_axis_name, 'user_id', 'repetition_model', 'datetime', 'level',
    ])
    source['ratio'] = source.value_x / (source.value_x + source.value_y)
    source = source.drop(['value_x', 'type_x', 'value_y', 'type_y'], axis=1)

    source = source.groupby([x_axis_name, 'repetition_model', 'level', 'user_id']).mean().reset_index()
    source = source.groupby([x_axis_name, 'repetition_model', 'level']).agg(['mean', 'std']).reset_index()
    source.columns = [l1 if not l2 else l2 for l1, l2 in source.columns]
    source['min'] = source['mean'] - source['std'] / 2
    source['max'] = source['mean'] + source['std'] / 2
    source.dropna()
    # source.fillna(0, inplace=True)

    source = source.replace({
        'level': {
            'initial': 'Initial',
            'level_1': 'Level.0',
            'level_2': 'Level.1',
            'level_3': 'Level.2',
            'level_4': 'Level.3',
            'level_5': 'Level.4',
            'level_6': 'Level.5',
            'level_7': 'Level.6',
            'level_8': 'Level.7',
            'level_9': 'Level.8',
        },
        'repetition_model': {
            'karl100': 'KARL',
            'leitner': 'Leitner',
            'sm2': 'SM-2',
        }
    })

    selection = alt.selection_multi(fields=['level'], bind='legend')

    line = alt.Chart().mark_line().encode(
        alt.X(x_axis_name, title='Minutes spent on app'),
        alt.Y('mean', title='Recall rate'),
        color=alt.Color('level', title='Level'),
        size=alt.condition(selection, alt.value(3), alt.value(1)),
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2))
    )
    band = alt.Chart().mark_area().encode(
        x=x_axis_name,
        y='min',
        y2='max',
        color=alt.Color('level', title='Level'),
        opacity=alt.condition(selection, alt.value(0.5), alt.value(0.2))
    )
    chart = alt.layer(
        band,
        line,
        data=source
    ).properties(
        width=180,
        height=180
    ).facet(
        facet=alt.Facet('repetition_model', title=None),
    ).add_selection(
        selection
    ).configure_legend(
        labelFontSize=15,
    )

    # save_chart_and_pdf(chart, 'figures/repetition_model_ratio')
    chart.save(f'{output_path}/repetition_model_level_ratio.json')
    # chart.save('test.json')


def figure_karl100_vs_karl85_level_ratio(
    df: pd.DataFrame,
    output_path: str,
):
    '''karl100 vs karl85 level ratio'''
    x_axis_name = 'n_minutes_spent_binned'

    progress_names = [f'initial_O', f'initial_X']
    for i in df.leitner_box.unique():
        if i > 0:
            progress_names += [f'level_{i}_O', f'level_{i}_X']

    source = pd.melt(
        df,
        id_vars=[x_axis_name, 'datetime', 'repetition_model', 'user_id'],
        value_vars=progress_names,
        var_name='name',
        value_name='value',
    )

    source = source[source.repetition_model.isin(['karl100', 'karl85'])]

    source['type'] = source.name.apply(lambda x: 'Successful' if x[-1] == 'O' else 'Failed')
    source['level'] = source.name.apply(lambda x: x[:-2])

    df_left = source[source.type == 'Successful'].drop(['name'], axis=1)
    df_right = source[source.type == 'Failed'].drop(['name'], axis=1)
    source = pd.merge(df_left, df_right, how='left', on=[
        x_axis_name, 'user_id', 'repetition_model', 'datetime', 'level',
    ])
    source['ratio'] = source.value_x / (source.value_x + source.value_y)
    source = source.drop(['value_x', 'type_x', 'value_y', 'type_y'], axis=1)

    source = source.groupby([x_axis_name, 'repetition_model', 'level', 'user_id']).mean().reset_index()
    source = source.groupby([x_axis_name, 'repetition_model', 'level']).agg(['mean', 'std']).reset_index()
    source.columns = [l1 if not l2 else l2 for l1, l2 in source.columns]
    source['min'] = source['mean'] - source['std'] / 2
    source['max'] = source['mean'] + source['std'] / 2
    source.dropna()
    # source.fillna(0, inplace=True)

    source = source.replace({
        'level': {
            'initial': 'Initial',
            'level_1': 'Level.0',
            'level_2': 'Level.1',
            'level_3': 'Level.2',
            'level_4': 'Level.3',
            'level_5': 'Level.4',
            'level_6': 'Level.5',
            'level_7': 'Level.6',
            'level_8': 'Level.7',
            'level_9': 'Level.8',
        },
        'repetition_model': {
            'karl100': 'KARL Target=100%',
            'karl85': 'KARL Target=85%',
        }
    })

    selection = alt.selection_multi(fields=['level'], bind='legend')

    line = alt.Chart().mark_line().encode(
        alt.X(x_axis_name, title='Minutes spent on app'),
        alt.Y('mean', title='Recall rate'),
        color=alt.Color('level', title='Level'),
        size=alt.condition(selection, alt.value(3), alt.value(1)),
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2))
    )
    band = alt.Chart().mark_area().encode(
        x=x_axis_name,
        y='min',
        y2='max',
        color=alt.Color('level', title='Level'),
        opacity=alt.condition(selection, alt.value(0.5), alt.value(0.2))
    )
    chart = alt.layer(
        band,
        line,
        data=source
    ).properties(
        width=300,
        height=300
    ).facet(
        facet=alt.Facet('repetition_model', title=None),
    ).add_selection(
        selection
    ).configure_legend(
        labelFontSize=15,
    )

    # save_chart_and_pdf(chart, 'figures/repetition_model_ratio')
    chart.save(f'{output_path}/100vs85_level_ratio.json')
    # chart.save('test.json')


def get_user_charts(user: User):
    '''Gather records into a single dataframe'''
    correct_on_first_try = {}
    rows = []
    for record in user.records:
        if record.fact_id not in correct_on_first_try:
            correct_on_first_try[record.fact_id] = record.response
        elapsed_seconds = record.elapsed_milliseconds_text / 1000
        elapsed_seconds += record.elapsed_milliseconds_answer / 1000
        elapsed_minutes = elapsed_seconds / 60

        leitner_box = json.loads(record.user_snapshot)['leitner_box']

        rows.append({
            'record_id': record.record_id,
            'user_id': user.user_id,
            'fact_id': record.fact_id,
            'repetition_model': json.loads(record.scheduler_snapshot)['repetition_model'],
            'is_new_fact': record.is_new_fact,
            'result': record.response,
            'datetime': record.date,
            'elapsed_minutes': elapsed_minutes,
            'is_known_fact': correct_on_first_try[record.fact_id],
            'leitner_box': leitner_box.get(record.fact_id, 0),
        })
    df = pd.DataFrame(rows).sort_values('datetime', axis=0)

    df[f'initial_O_'] = df.apply(lambda x: (x.leitner_box == 0) & x.result, axis=1)
    df[f'initial_X_'] = df.apply(lambda x: (x.leitner_box == 0) & ~x.result, axis=1)
    df[f'initial_O'] = df[f'initial_O_'].cumsum()
    df[f'initial_X'] = df[f'initial_X_'].cumsum()
    progress_names = [f'initial_O', f'initial_X']
    for i in df.leitner_box.unique():
        if i == 0:
            continue
        df[f'level_{i}_O_'] = df.apply(lambda x: (x.leitner_box == i) & x.result & (~x.is_known_fact), axis=1)
        df[f'level_{i}_X_'] = df.apply(lambda x: (x.leitner_box == i) & (~x.result) & (~x.is_known_fact), axis=1)
        df[f'level_{i}_O'] = df[f'level_{i}_O_'].cumsum()
        df[f'level_{i}_X'] = df[f'level_{i}_X_'].cumsum()
        progress_names += [f'level_{i}_O', f'level_{i}_X']

    charts = {}  # chart name -> chart

    '''Progress (count on each level) vs datetime + bars for effort'''
    source = pd.melt(
        df,
        id_vars='datetime',
        value_vars=progress_names,
        var_name='name',
        value_name='value',
    ).reset_index()
    source['type'] = source.name.apply(lambda x: 'Successful' if x[-1] == 'O' else 'Failed')
    source['level'] = source.name.apply(lambda x: x[:-2])

    df_right = df[df.user_id == user.user_id][[
        'datetime',
        'elapsed_minutes',
    ]]
    source = pd.merge(source, df_right, how='left', on='datetime')

    source['date'] = source['datetime'].apply(lambda x: x.date())
    source.date = pd.to_datetime(source.date)
    source.datetime = pd.to_datetime(source.datetime)

    source = source.replace({
        'level': {
            'initial': 'Initial',
            'level_1': 'Level.0',
            'level_2': 'Level.1',
            'level_3': 'Level.2',
            'level_4': 'Level.3',
            'level_5': 'Level.4',
            'level_6': 'Level.5',
            'level_7': 'Level.6',
            'level_8': 'Level.7',
            'level_9': 'Level.8',
        },
    })

    selection = alt.selection_multi(fields=['level'], bind='legend')
    base = alt.Chart(source).encode(
        alt.X('date', axis=alt.Axis(title='Date'))
    )
    bar = base.mark_bar(opacity=0.3, color='#57A44C').encode(
        alt.Y(
            'sum(elapsed_minutes)',
            axis=alt.Axis(title='Minutes spent on app', titleColor='#57A44C')
        )
    )
    line = base.mark_line().encode(
        alt.Y('value', axis=alt.Axis(title='Number of flashcards')),
        color=alt.Color('level', title='Level'),
        strokeDash=alt.StrokeDash('type', title='Result'),
        size=alt.condition(selection, alt.value(3), alt.value(1))
    ).add_selection(
        selection
    )
    repetition_model = json.loads(user.records[-1].scheduler_snapshot)['repetition_model']
    chart = alt.layer(
        bar,
        line
    ).resolve_scale(
        y='independent'
    ).configure_legend(
        labelFontSize=15,
    )

    # .properties(
    #     title=f'user: {user.user_id} {repetition_model}'
    # )

    charts['user_level_vs_effort'] = chart

    df_left = source[source.type == 'Successful'].drop(['name', 'date'], axis=1)
    df_right = source[source.type == 'Failed'].drop(['name', 'date'], axis=1)
    source = pd.merge(df_left, df_right, how='left', on=[
        'datetime', 'level', 'elapsed_minutes',
    ]).drop(['index_x', 'index_y'], axis=1)
    source['ratio'] = source.value_x / (source.value_x + source.value_y)
    source['date'] = source['datetime'].apply(lambda x: x.date())
    source.date = pd.to_datetime(source.date)
    source.datetime = pd.to_datetime(source.datetime)
    chart = alt.Chart(source).mark_line().encode(
        alt.X('date', title='Date'),
        alt.Y('mean(ratio)', title='Recall rate'),
        color=alt.Color('level', title='Level'),
        size=alt.condition(selection, alt.value(3), alt.value(1))
    ).add_selection(
        selection
    ).configure_legend(
        labelFontSize=15,
    )

    # .properties(
    #     title=f'user: {user.user_id} {repetition_model}'
    # )

    charts['user_level_ratio'] = chart

    return charts


def figures():
    output_path = '/fs/clip-quiz/shifeng/ihsgnef.github.io/images'
    session = get_sessions()['prod']
    df = get_processed_df(session)
    figure_new_old_successful_failed(df, output_path)
    figure_model_level_vs_effort(df, output_path)
    figure_model_level_ratio(df, output_path)
    figure_karl100_vs_karl85_level_ratio(df, output_path)
    user = session.query(User).get('463')
    charts = get_user_charts(user)
    charts['user_level_vs_effort'].save(f'{output_path}/{user.user_id}_user_level_vs_effort.json')
    charts['user_level_ratio'].save(f'{output_path}/{user.user_id}_user_level_ratio.json')


if __name__ == '__main__':
    figures()


"""
n_bins = 10
n_facts_bin_size = (df['n_facts_shown'].max()) // (n_bins - 1)
n_facts_bins = [i * n_facts_bin_size for i in range(n_bins)]
df_users = df.groupby('user_id')['record_id'].count().reset_index(name='count')
df_users['count_binned'] = df_users['count'].apply(func(n_facts_bins))
user_binned = df_users[['user_id', 'count_binned']].to_dict()
user_binned_dict = {
    v: user_binned['count_binned'][k] for k, v in user_binned['user_id'].items()
}

df_sub = df.copy()
df_sub['user_records_binned'] = df_sub['user_id'].apply(lambda x: user_binned_dict[x])
for user_bin in df_sub['user_records_binned'].unique():
    df_plot = pd.melt(
        df_sub[df_sub.user_records_binned == user_bin],
        id_vars=['user_id', 'repetition_model', x_axis_name],
        value_vars=leitner_boxes,
        var_name='name',
        value_name='value',
    )
    df_plot = df_plot.groupby(['repetition_model', 'name', x_axis_name]).agg(['mean', 'std']).reset_index()
    df_plot.columns = [l1 if not l2 else l2 for l1, l2 in df_plot.columns]
    df_plot['min'] = df_plot['mean'] - df_plot['std'] / 2
    df_plot['max'] = df_plot['mean'] + df_plot['std'] / 2
    df_plot.name = df_plot.name.astype(CategoricalDtype(categories=leitner_boxes,ordered=True))

    Path('figures/repetition_model_reports').mkdir(parents=True, exist_ok=True)

    line = alt.Chart().mark_line().encode(
        x=x_axis_name,
        y='mean',
        color='name',
    )
    band = alt.Chart().mark_area(opacity=0.5, color='gray').encode(
        x=x_axis_name,
        y='min',
        y2='max',
        color='name',
    )
    chart = alt.layer(
        band, line, data=df_plot
    ).facet(
        'repetition_model',
        columns=2
    ).properties(
        title=f'users records bin: {user_bin}'
    )
    save_chart_and_pdf(chart, f'figures/repetition_model_reports/{user_bin}')
# %%
# '''System report'''
# # daily active users 
# df_plot = df.groupby(['user_id', 'repetition_model', 'date_binned']).mean().reset_index()
# df_plot = df_plot.groupby(['repetition_model', 'date_binned'])['user_id'].count().reset_index(name='n_active_users')
# total_daily_count = df_plot.groupby('date_binned').sum().to_dict()['n_active_users']
# df_plot['ratio'] = df_plot.apply(lambda x: x['n_active_users'] / total_daily_count[x['date_binned']], axis=1)
# 
# chart = alt.Chart(df_plot).mark_line().encode(
#     x='date_binned',
#     y='n_active_users',
#     color='repetition_model',
# )
# save_chart_and_pdf(chart, 'figures/system_activity')
# %%
# # scatter plot of number of records vs number of minutes colored by repetition model
# df_left = df.groupby(['user_id', 'repetition_model'])['user_id'].count().reset_index(name='n_records')
# df_right = df.groupby(['user_id', 'repetition_model'])['elapsed_minutes'].sum().reset_index(name='total_minutes')
# df_plot = pd.merge(df_left, df_right, how='left', on=['user_id', 'repetition_model'])
# 
# alt.Chart(df_plot).mark_point().encode(
#     alt.X('n_records'),
#     alt.Y('total_minutes'),
#     color='repetition_model',
# )
# %%
alt.chart(df).mark_point().encode(
    alt.X('date'),
    alt.Y('sum(n_facts_shown)'),
    color='repetition_model',
)
"""