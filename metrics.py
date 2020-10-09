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

# %%
'''Gather records into a DataFrame'''
session = get_sessions()['prod']

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
raw_df = pd.DataFrame(rows).sort_values('datetime', axis=0)
# %%
df = raw_df.copy()
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
# %%
# [new, old] x [correct, wrong]
x_axis_name = 'n_minutes_spent_binned'
metrics = [
    'ratio_new_correct_vs_all',
    'ratio_new_wrong_vs_all',
    'ratio_old_wrong_vs_all',
    'ratio_old_correct_vs_all',
]
df_plot = df.groupby(['repetition_model', x_axis_name]).mean().reset_index()
df_plot = pd.melt(
    df_plot,
    id_vars=['repetition_model', x_axis_name],
    value_vars=metrics,
    var_name='name',
    value_name='value',
)
df_plot.name = df_plot.name.astype(CategoricalDtype(categories=metrics, ordered=True))

chart = alt.Chart(df_plot).mark_area().encode(
    x='n_minutes_spent_binned',
    y='sum(value)',
    color='name',
).facet(
    facet='repetition_model:N',
    columns=2
)
save_chart_and_pdf(chart, f'figures/new_old_correct_wrong')
# %%
x_axis_name = 'n_minutes_spent_binned'

df_plot = pd.melt(
    df,
    id_vars=[x_axis_name, 'repetition_model', 'user_id'],
    value_vars=progress_names,
    var_name='name',
    value_name='value',
)

df_plot['type'] = df_plot.name.apply(lambda x: 'Correct' if x[-1] == 'O' else 'Wrong')
df_plot['level'] = df_plot.name.apply(lambda x: x[:-2])
df_plot = df_plot.groupby([x_axis_name, 'repetition_model', 'name', 'type', 'level', 'user_id']).mean().reset_index()
df_plot = df_plot.groupby([x_axis_name, 'repetition_model', 'name', 'type', 'level']).agg(['mean', 'std']).reset_index()
df_plot.columns = [l1 if not l2 else l2 for l1, l2 in df_plot.columns]
df_plot['min'] = df_plot['mean'] - df_plot['std'] / 2
df_plot['max'] = df_plot['mean'] + df_plot['std'] / 2

line = alt.Chart().mark_line().encode(
    x=x_axis_name,
    y='mean',
    color='level',
    strokeDash='type',
)
band = alt.Chart().mark_area(opacity=0.5, color='gray').encode(
    x=x_axis_name,
    y='min',
    y2='max',
    color='name',
)
chart = alt.layer(band, line, data=df_plot).facet('repetition_model', columns=2)
save_chart_and_pdf(chart, 'figures/repetition_model_study_reports_all')
# %%
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