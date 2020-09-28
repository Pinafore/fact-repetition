# %%
import json
import bisect
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict
from plotnine import ggplot, aes, theme,\
    geom_point, geom_line, geom_area,\
    facet_grid, facet_wrap,\
    element_text

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

# the question we are trying to answer is: how much did the user manage
 # to learn during this time period?
        # The first choice is to only consider facts that are first shown
        # within the time window.
        # Among these cards, some are known to the user a priori, some are
        # unfimiliar but eventually learned, some never got learned during the
        # time window.
        # The goal is for the user to be able to learn the maximium number of
        # cards with the minimium possible effort.


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
            'elapsed_minutes': seconds / 60,
            'user_start_date': user_start_date[user.user_id],
            'is_known_fact': correct_on_first_try[user.user_id][record.fact_id],
            'leitner_box': json.loads(record.user_snapshot)['leitner_box'][record.fact_id],
        })
raw_df = pd.DataFrame(rows)

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
print(n_minutes_bins)
df['n_minutes_spent_binned'] = df.n_facts_shown.apply(func(n_minutes_bins))

'''Compute derivative metrics'''
df['n_new_facts_correct'] = df.groupby('user_id', group_keys=False).apply(lambda x: x.is_new_fact & x.result)
df['n_new_facts_wrong'] = df.groupby('user_id', group_keys=False).apply(lambda x: x.is_new_fact & ~x.result)
df['n_old_facts_correct'] = df.groupby('user_id', group_keys=False).apply(lambda x: ~x.is_new_fact & x.result)
df['n_old_facts_wrong'] = df.groupby('user_id', group_keys=False).apply(lambda x: ~x.is_new_fact & ~x.result)
for i in df.leitner_box.unique():
    # not previously-known facts that got improved to level-i familiarity 
    df[f'n_learned_{i}'] = df.groupby('user_id', group_keys=False).apply(lambda x: (x.leitner_box == i) & x.result & ~x.is_known_fact)

# choose an x-axis and average across users.
# choices (binned version optional):
# n_minutes_spent, n_facts_shown, n_days_since_start, date
df['n_new_facts_correct_csum'] = df.groupby('user_id')['n_new_facts_correct'].cumsum()
df['n_new_facts_wrong_csum'] = df.groupby('user_id')['n_new_facts_wrong'].cumsum()
df['n_old_facts_correct_csum'] = df.groupby('user_id')['n_old_facts_correct'].cumsum()
df['n_old_facts_wrong_csum'] = df.groupby('user_id')['n_old_facts_wrong'].cumsum()

# [new, old] x [correct, wrong] vs all
df['ratio_new_correct_vs_all'] = df.n_new_facts_correct_csum / df.n_facts_shown
df['ratio_new_wrong_vs_all'] = df.n_new_facts_wrong_csum / df.n_facts_shown
df['ratio_old_correct_vs_all'] = df.n_old_facts_correct_csum / df.n_facts_shown
df['ratio_old_wrong_vs_all'] = df.n_old_facts_wrong_csum / df.n_facts_shown

names = [
    'ratio_new_correct_vs_all',
    'ratio_new_wrong_vs_all',
    'ratio_old_correct_vs_all',
    'ratio_old_wrong_vs_all',
]

x_axis_name = 'n_facts_shown_binned'
df_plot = pd.melt(
    df.groupby(['repetition_model', x_axis_name]).sum().reset_index(),
    id_vars=['repetition_model', x_axis_name],
    value_vars=names,
    var_name='name',
    value_name='value',
)

p = (
    ggplot(df_plot)
    + geom_area(aes(x=x_axis_name, y='value', fill='name'), alpha=0.75)
    + facet_wrap('repetition_model')
    + theme_fs()
    + theme(
        axis_text_x=element_text(rotation=30)
    )
)
p.draw()

# %%
# turn metrics into columns so it's easier to compute derivative metrics
df = df.pivot_table(index=['repetition_model', 'n_days_since_start'], columns='name', values='value')
df['n_facts_shown_csum'] = df.groupby('repetition_model')['n_facts_shown'].cumsum()
df['n_new_facts_correct_csum'] = df.groupby('repetition_model')['n_new_facts_correct'].cumsum()
df['n_new_facts_wrong_csum'] = df.groupby('repetition_model')['n_new_facts_wrong'].cumsum()
df['n_old_facts_correct_csum'] = df.groupby('repetition_model')['n_old_facts_correct'].cumsum()
df['n_old_facts_wrong_csum'] = df.groupby('repetition_model')['n_old_facts_wrong'].cumsum()
df['n_learned_csum'] = df.groupby('repetition_model')['n_learned'].cumsum()
df['n_learned_but_forgotten_csum'] = df.groupby('repetition_model')['n_learned_but_forgotten'].cumsum()
# [new, old] x [correct, wrong] vs all
df['ratio_new_correct_vs_all'] = df.n_new_facts_correct_csum / df.n_facts_shown_csum
df['ratio_new_wrong_vs_all'] = df.n_new_facts_wrong_csum / df.n_facts_shown_csum
df['ratio_old_correct_vs_all'] = df.n_old_facts_correct_csum / df.n_facts_shown_csum
df['ratio_old_wrong_vs_all'] = df.n_old_facts_wrong_csum / df.n_facts_shown_csum
# [learned, learned_but_forgotten] vs all
df['ratio_learned_vs_all'] = df.n_learned_csum / df.n_facts_shown_csum
df['ratio_learned_but_forgotten_vs_all'] = df.n_learned_but_forgotten_csum / df.n_facts_shown_csum
# df['ratio_known_vs_all'] = df.n_known_old_facts_shown / df.n_facts_shown
# df['ratio_known_vs_old'] = df.n_known_old_facts_shown / df.n_old_facts_shown
# df['ratio_known_correct_vs_known'] = df.n_known_old_facts_correct / df.n_known_old_facts_shown

# %%
# done with derivatives, convert metrics from columns to rows
df = df.stack()
df.name = 'value'
df = df.reset_index()

# %%
# so that plotnine recognize dates as sequential data
df.date = df.date.astype(np.datetime64)
df.date_binned = df.date_binned.astype(np.datetime64)

'''Break down of [new, old] x [correct, wrong]'''
names = [
    'ratio_new_correct_vs_all',
    'ratio_new_wrong_vs_all',
    'ratio_old_correct_vs_all',
    'ratio_old_wrong_vs_all',
]
p = (
    ggplot(df[df.name.isin(names)])
    + geom_area(aes(x='n_days_since_start', y='value', fill='name'), alpha=0.75)
    + facet_wrap('repetition_model')
    + theme_fs()
    + theme(
        axis_text_x=element_text(rotation=30)
    )
)
p.draw()

# %%
names = [
    'ratio_learned_vs_all',
    'ratio_learned_but_forgotten_vs_all',
]
p = (
    ggplot(df[df.name.isin(names)])
    + geom_line(aes(x='n_facts_shown_csum', y='value', color='name'), alpha=0.75)
    + facet_wrap('repetition_model')
    + theme_fs()
    + theme(
        axis_text_x=element_text(rotation=30)
    )
)
p.draw()
# %%