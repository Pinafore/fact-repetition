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
alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')

from karl.new_util import User, Record, parse_date, theme_fs
from karl.web import get_sessions

def save_chart_and_pdf(chart, path):
    chart.save(f'{path}.json')
    os.system(f'vl2vg {path}.json | vg2pdf > {path}.pdf')

session = get_sessions()['prod']

for user in tqdm(session.query(User), total=session.query(User).count()):
    if len(user.records) < 100:
        continue


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
            'leitner_box': leitner_box.get(record.fact_id, 1),
        })
    df = pd.DataFrame(rows).sort_values('datetime', axis=0)

    names = []
    for i in df.leitner_box.unique():
        df[f'level_{i - 1}_O_'] = df.apply(lambda x: (x.leitner_box == i) & (x.result), axis=1)
        df[f'level_{i - 1}_X_'] = df.apply(lambda x: (x.leitner_box == i) & (~x.result), axis=1)
        df[f'level_{i - 1}_O'] = df[f'level_{i - 1}_O_'].cumsum()
        df[f'level_{i - 1}_X'] = df[f'level_{i - 1}_X_'].cumsum()
        names += [f'level_{i - 1}_O', f'level_{i - 1}_X']

    df_plot = pd.melt(
        df,
        id_vars='datetime',
        value_vars=names,
        var_name='name',
        value_name='value',
    ).reset_index()
    df_plot['type'] = df_plot.name.apply(lambda x: 'Correct' if x[-1] == 'O' else 'Wrong')
    df_plot['level'] = df_plot.name.apply(lambda x: x[:-2])

    df_right = df[df.user_id == user.user_id][[
        'datetime',
        'elapsed_minutes',
    ]]
    df_plot = pd.merge(df_plot, df_right, how='left', on='datetime')

    df_plot['date'] = df_plot['datetime'].apply(lambda x: x.date())
    df_plot.date = pd.to_datetime(df_plot.date)
    df_plot.datetime = pd.to_datetime(df_plot.datetime)

    base = alt.Chart(df_plot).encode(
        alt.X('date', axis=alt.Axis(title='Date'))
    )
    bar = base.mark_bar(opacity=0.4, color='#57A44C').encode(
        alt.Y(
            'sum(elapsed_minutes)',
            axis=alt.Axis(title='Minutes spent', titleColor='#57A44C'),
        )
    )
    line = base.mark_line().encode(
        alt.Y(
            'value',
            axis=alt.Axis(title='Familiarity'),
        ),
        color='level',
        strokeDash='type',
    )
    repetition_model = json.loads(user.records[-1].scheduler_snapshot)['repetition_model']
    chart = alt.layer(bar, line).resolve_scale(
        y='independent',
    ).properties(
        title=f'user: {user.user_id} {repetition_model}'
    )
    Path('figures/user_progress_effort').mkdir(parents=True, exist_ok=True)
    save_chart_and_pdf(chart, f'figures/user_progress_effort/{repetition_model}_{user.user_id}')

    df_left = df_plot[df_plot.type == 'Correct'].drop(['name', 'date'], axis=1)
    df_right = df_plot[df_plot.type == 'Wrong'].drop(['name', 'date'], axis=1)
    df1 = pd.merge(df_left, df_right, how='left', on=[
        'datetime', 'level', 'elapsed_minutes',
    ]).drop(['index_x', 'index_y'], axis=1)
    df1['ratio'] = df1.value_x / (df1.value_x + df1.value_y)
    df1['date'] = df1['datetime'].apply(lambda x: x.date())
    df1.date = pd.to_datetime(df1.date)
    df1.datetime = pd.to_datetime(df1.datetime)
    chart = alt.Chart(df1).mark_line().encode(
        x='date',
        y='mean(ratio)',
        color='level',
    ).properties(
        title=f'user: {user.user_id} {repetition_model}'
    )
    Path('figures/user_level_ratio').mkdir(parents=True, exist_ok=True)
    save_chart_and_pdf(chart, f'figures/user_level_ratio/{repetition_model}_{user.user_id}')
    # %%