# %%
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
from plotnine import ggplot, ggtitle, aes, theme,\
    geom_point, geom_line, geom_area, geom_ribbon, geom_bar, geom_density, \
    facet_grid, facet_wrap,\
    element_text, scale_fill_brewer, scale_fill_manual,\
    scale_x_log10, scale_y_log10

from karl.new_util import User, Record, parse_date, theme_fs
from karl.web import get_sessions


def infer_repetition_model(scheduler_snapshot) -> str:
    params = json.loads(scheduler_snapshot)
    if params['qrep'] == 0:
        if params['leitner'] > 0:
            return 'leitner'
        elif params['sm2'] > 0:
            return 'sm2'
        else:
            return 'unknown'
    else:
        if 'recall_target' in params:
            return 'karl-' + str(int(params['recall_target'] * 100))
        else:
            return 'karl-100'


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
    repetition_model = infer_repetition_model(last_record.scheduler_snapshot)

    for record in session.query(Record).\
        filter(Record.user_id == user.user_id).\
        filter(Record.date >= date_start).\
        filter(Record.date <= date_end):
        seconds = record.elapsed_seconds_text if record.elapsed_seconds_text else record.elapsed_milliseconds_text / 1000
        seconds += record.elapsed_seconds_answer if record.elapsed_seconds_answer else record.elapsed_milliseconds_answer / 1000

        rows.append({
            'record_id': record.record_id,
            'user_id': user.user_id,
            'fact_id': record.fact_id,
            'repetition_model': repetition_model,
            'is_new_fact': record.is_new_fact,
            'result': record.response,
            'date': record.date.date(),
            'datetime': record.date,
            'elapsed_minutes': seconds / 60,
            'user_start_date': user_start_date[user.user_id],
            'is_known_fact': correct_on_first_try[user.user_id][record.fact_id],
            'leitner_box': json.loads(record.user_snapshot)['leitner_box'][record.fact_id],
        })
raw_df = pd.DataFrame(rows).sort_values('datetime', axis=0)

# %%
df = raw_df.copy()
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
for i in df.leitner_box.unique():
    if i > 1:
        # initial box means first occurrence; it doesn't count
        # not previously-known facts that got improved to level-i familiarity 
        df[f'n_learned_{i}'] = df.groupby('user_id', group_keys=False).apply(lambda x: (x.leitner_box == i) & x.result & ~x.is_known_fact)

df['n_new_facts_correct_csum'] = df.groupby('user_id')['n_new_facts_correct'].cumsum()
df['n_new_facts_wrong_csum'] = df.groupby('user_id')['n_new_facts_wrong'].cumsum()
df['n_old_facts_correct_csum'] = df.groupby('user_id')['n_old_facts_correct'].cumsum()
df['n_old_facts_wrong_csum'] = df.groupby('user_id')['n_old_facts_wrong'].cumsum()
for i in df.leitner_box.unique():
    if i > 1:
        # initial box means first occurrence; it doesn't count
        df[f'n_learned_{i}_csum'] = df.groupby('user_id')[f'n_learned_{i}'].cumsum()

# [new, old] x [correct, wrong] vs all
df['ratio_new_correct_vs_all'] = df.n_new_facts_correct_csum / df.n_facts_shown
df['ratio_new_wrong_vs_all'] = df.n_new_facts_wrong_csum / df.n_facts_shown
df['ratio_old_correct_vs_all'] = df.n_old_facts_correct_csum / df.n_facts_shown
df['ratio_old_wrong_vs_all'] = df.n_old_facts_wrong_csum / df.n_facts_shown
# %%
# [new, old] x [correct, wrong]
x_axis_name = 'n_minutes_spent_binned'
metrics = [
    'ratio_new_correct_vs_all',
    'ratio_new_wrong_vs_all',
    'ratio_old_correct_vs_all',
    'ratio_old_wrong_vs_all',
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
p = (
    ggplot(
        df_plot,
        aes(x=x_axis_name, y='value', color='name', fill='name'),
    )
    + geom_area(alpha=0.75)
    + facet_wrap('repetition_model')
    + theme_fs()
    + theme(
        axis_text_x=element_text(rotation=30)
    )
    + scale_fill_brewer(type='div', palette=4)
    # + scale_fill_manual(
    #     values=[
    #         '#ff1400',  # dark red
    #         '#ffaca5',  # light red
    #         '#595bff',  # dark blue
    #         '#b2b3ff',  # light blue
    #     ]
    # )
)
p.draw()
p.save(f'figures/new_old_correct_wrong.pdf')
# %%
# study progress
n_bins = 10
n_facts_bin_size = (df['n_facts_shown'].max()) // (n_bins - 1)
n_facts_bins = [i * n_facts_bin_size for i in range(n_bins)]
df_users = df.groupby('user_id')['record_id'].count().reset_index(name='count')
df_users['count_binned'] = df_users['count'].apply(func(n_facts_bins))
user_binned = df_users[['user_id', 'count_binned']].to_dict()
user_binned_dict = {
    v: user_binned['count_binned'][k] for k, v in user_binned['user_id'].items()
}

x_axis_name = 'n_minutes_spent_binned'
leitner_boxes = sorted([i for i in df.leitner_box.unique() if i > 1])
# reverse because lower sequential values usually corresponds to darker colors
leitner_boxes = [f'n_learned_{i}_csum' for i in leitner_boxes][::-1]
extended_metrics = (
    [f'{metric}_mean' for metric in leitner_boxes]
    + [f'{metric}_min' for metric in leitner_boxes]
    + [f'{metric}_max' for metric in leitner_boxes]
)

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
    
    p = (
        ggplot(
            df_plot,
            aes(x=x_axis_name, y='mean', ymin='min', ymax='max', color='name', fill='name'),
        )
        + geom_line(alpha=0.75, size=1)
        + geom_ribbon(alpha=0.5)
        + facet_wrap('repetition_model')
        + theme_fs()
        + theme(
            axis_text_x=element_text(rotation=30),
            aspect_ratio=1,
        )
        + ggtitle(f'users records bin: {user_bin}')
    )
    p.draw()
    p.save(f'figures/repetition_model_study_reports_{user_bin}.pdf')
# %%
'''System report'''
# daily active users 
df_plot = df.groupby(['user_id', 'repetition_model', 'date_binned']).mean().reset_index()
df_plot = df_plot.groupby(['repetition_model', 'date_binned'])['user_id'].count().reset_index(name='n_active_users')
total_daily_count = df_plot.groupby('date_binned').sum().to_dict()['n_active_users']
df_plot['ratio'] = df_plot.apply(lambda x: x['n_active_users'] / total_daily_count[x['date_binned']], axis=1)

p = (
    ggplot(
        df_plot,
        aes(x='date_binned', y='n_active_users', color='repetition_model', fill='repetition_model')
    )
    + geom_line(alpha=0.75, size=1)
    + theme_fs()
    + theme(
        axis_text_x=element_text(rotation=30)
    )
)
p.draw()
p.save(f'figures/system_activity.pdf')
# %%
# scatter plot of number of records vs number of minutes colored by repetition model
df_left = df.groupby(['user_id', 'repetition_model'])['user_id'].count().reset_index(name='n_records')
df_right = df.groupby(['user_id', 'repetition_model'])['elapsed_minutes'].sum().reset_index(name='total_minutes')
df_plot = pd.merge(df_left, df_right, how='left', on=['user_id', 'repetition_model'])
p = (
    ggplot(
        df_plot,
        aes(x='n_records', y='total_minutes', color='repetition_model', fill='repetition_model')
    )
    + geom_point(alpha=0.6, size=3)
    + theme_fs()
    + scale_fill_brewer(type='qual')
    + scale_x_log10()
    + scale_y_log10()
)
p.draw()
p.save(f'figures/users.pdf')
# %%
users = session.query(User).all()
users = sorted(users, key=lambda x: -len(x.records))
# %%
for i, user in enumerate(users[:30]):
    x_axis_name = 'datetime'
    repetition_model = infer_repetition_model(user.records[-1].scheduler_snapshot)
    leitner_boxes = sorted([i for i in df.leitner_box.unique() if i > 1])
    # reverse because lower sequential values usually corresponds to darker colors
    leitner_boxes = [f'n_learned_{i}_csum' for i in leitner_boxes][::-1]
    leitner_box_type = CategoricalDtype(categories=leitner_boxes, ordered=True)
    df_plot = pd.melt(
        df[df.user_id == user.user_id],
        id_vars=x_axis_name,
        value_vars=leitner_boxes,
        var_name='name',
        value_name='value',
    ).reset_index()
    df_plot.name = df_plot.name.astype(leitner_box_type)

    df_plot['date'] = df_plot['datetime'].apply(lambda x: x.date())
    df_plot['count'] = 0.1

    df_minutes = df[df.user_id == user.user_id][['datetime', 'elapsed_minutes']]
    df_plot = pd.merge(df_plot, df_minutes, how='left', on='datetime')
    # df_1 = df_plot.groupby('date')['index'].count().reset_index(name='count')
    # df_plot = pd.merge(df_plot, df_1, how='left', on='date')
    # df_plot['count'] /= 100

    p = (
        ggplot(df_plot)
        + geom_line(
            aes(x=x_axis_name, y='value', color='name', fill='name'),
            alpha=0.75, size=1.5,
        )
        + geom_bar(aes(x='date', y='elapsed_minutes'), stat='identity', alpha=0.5)
        + theme_fs()
        + theme(
            axis_text_x=element_text(rotation=30)
        )
        + scale_fill_brewer(type='seq', palette=3)
        + ggtitle(f'user_id: {user.user_id} repetition_model: {repetition_model}')
    )
    p.draw()
    Path('figures/user_study_reports').mkdir(parents=True, exist_ok=True)
    p.save(f'figures/user_study_reports/{user.user_id}.pdf')
    
# %%