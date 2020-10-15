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
alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')

from karl.new_util import User, Record, parse_date, theme_fs

def save_chart_and_pdf(chart, path):
    chart.save(f'{path}.json')
    os.system(f'vl2vg {path}.json | vg2pdf > {path}.pdf')


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
    chart = alt.layer(bar, line).resolve_scale(
        y='independent')
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
        x='date',
        y='mean(ratio)',
        color='level',
        size=alt.condition(selection, alt.value(3), alt.value(1))
    ).add_selection(
        selection
    ).properties(
        title=f'user: {user.user_id} {repetition_model}'
    )

    charts['user_level_ratio'] = chart

    return charts


def get_sessions():
    import socket
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    hostname = socket.gethostname()
    if hostname.startswith('newspeak'):
        db_host = '/fs/clip-quiz/shifeng/postgres/run'
    elif hostname.startswith('lapine'):
        db_host = '/fs/clip-scratch/shifeng/postgres/run'
    else:
        print('unrecognized hostname')
        exit()
    engines = {
        'prod': create_engine(f'postgresql+psycopg2://shifeng@localhost:5433/karl-prod?host={db_host}'),
        'dev': create_engine(f'postgresql+psycopg2://shifeng@localhost:5433/karl-dev?host={db_host}'),
    }
    return {
        env: sessionmaker(bind=engine, autoflush=False)()
        for env, engine in engines.items()
    }


if __name__ == '__main__':
    session = get_sessions()['prod']
    user = session.query(User).get('463')
    charts = get_user_charts(user)
    output_path = '/fs/clip-quiz/shifeng/ihsgnef.github.io/images'
    charts['user_level_vs_effort'].save(f'{output_path}/{user.user_id}_user_level_vs_effort.json')
    charts['user_level_ratio'].save(f'{output_path}/{user.user_id}_user_level_ratio.json')