# %%
"""each metric has a name, a description, and a scalar value"""
# think of the metrics from the perspectives of the scheduler
# if the scheduler recommended this fact and the response is X, what does it say about the scheduler?
# e.g. is it too aggressively showing difficult new facts? is it repeating easy old facts too much?
# for `learned` metric, everything is limited to the given datetime span
# it captures the number of previous unknown facts that was successfully learned using the system

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

# %%
class DatedMetric:

    name = None
    description = None
    values = defaultdict(lambda: 0) # date -> scalar value

    def __init__(self, **kwargs):
        pass
    
    def update(self, record):
        pass

    def __getitem__(self, date: datetime):
        return self.values[date]

    def items(self):
        return self.values.items()


class n_facts_shown(DatedMetric):

    name = 'n_facts_shown'
    description = 'Number of facts shown (including repetitions).'

    def __init__(self, **kwargs):
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        self.values[record.date.date()] += 1


class n_new_facts_shown(DatedMetric):

    name = 'n_new_facts_shown'
    description = 'Number of new facts shown.'

    def __init__(self, **kwargs):
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        self.values[record.date.date()] += record.is_new_fact



class n_new_facts_correct(DatedMetric):

    name = 'n_new_facts_correct'
    description = 'Number of new facts answered correctly.'

    def __init__(self, **kwargs):
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        self.values[record.date.date()] += (record.is_new_fact and record.response)


class n_new_facts_wrong(DatedMetric):

    name = 'n_new_facts_wrong'
    description = 'Number of new facts answered incorrectly.'

    def __init__(self, **kwargs):
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        self.values[record.date.date()]+= (record.is_new_fact and not record.response)


class n_old_facts_shown(DatedMetric):

    name = 'n_old_facts_shown'
    description = 'Number of old facts reviewed.'

    def __init__(self, **kwargs):
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        self.values[record.date.date()] += not record.is_new_fact


class n_old_facts_correct(DatedMetric):

    name = 'n_old_facts_correct'
    description = 'Number of old facts answered correctly.'

    def __init__(self, **kwargs):
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        self.values[record.date.date()] += (not record.is_new_fact and record.response)


class n_old_facts_wrong(DatedMetric):

    name = 'n_old_facts_wrong'
    description = 'Number of old facts answered incorrectly.'

    def __init__(self, **kwargs):
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        self.values[record.date.date()] += (not record.is_new_fact and not record.response)


class n_known_old_facts_shown(DatedMetric):

    name = 'n_known_old_facts_shown'
    description = '''Number of already-known old facts shown. These are the
        facts that the user got correct the first try (potentially before the
        datetime span). Thess cards are probably too easy.'''

    def __init__(self, **kwargs):
        self.correct_on_first_try = kwargs.pop('correct_on_first_try')
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        self.values[record.date.date()] += not record.is_new_fact and \
            self.correct_on_first_try[record.user_id][record.fact_id]


class n_known_old_facts_correct(DatedMetric):

    name = 'n_known_old_facts_correct'
    description = 'Number of already-known old facts answered correctly (which is expected). These are the facts that the user got correct the first try (potentially before the datetime span). Thess cards are probably too easy.'

    def __init__(self, **kwargs):
        self.correct_on_first_try = kwargs.pop('correct_on_first_try')
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        self.values[record.date.date()] += not record.is_new_fact and \
            self.correct_on_first_try[record.user_id][record.fact_id] and \
            record.response


class n_known_old_facts_wrong(DatedMetric):

    name = 'n_known_old_facts_wrong'
    description = 'Number of already-known old facts answered incorrectly (which is unexpected). These are the facts that the user got correct the first try (potentially before the datetime span). This means the user might have got it correct by being lucky.'

    def __init__(self, **kwargs):
        self.correct_on_first_try = kwargs.pop('correct_on_first_try')
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        self.values[record.date.date()] += not record.is_new_fact and \
            self.correct_on_first_try[record.user_id][record.fact_id] and \
            not record.response


class n_learned(DatedMetric):

    name = 'n_learned'
    description = 'Number of not known facts that the user saw for the first time, but correctly answered multiple times afterwards. Specifically, we consider facts that the user got the correct answer twice consecutively.'

    def __init__(self, **kwargs):
        self.counter = {}  # fact_id -> number of consecutive correct guesses
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        if record.is_new_fact and not record.response:
            # only consider facts not known before
            self.counter[record.fact_id] = 0
        if record.fact_id not in self.counter:
            return

        if record.response:
            self.counter[record.fact_id] += 1
            if self.counter[record.fact_id] == 2:
                self.values[record.date.date()] += 1
        else:
            self.counter[record.fact_id] = 0


class n_learned_but_forgotten(DatedMetric):

    name = 'n_learned_but_fogotten'
    description = 'Number of learned cards answered incorrectly afterwards. See `n_learned` for definition of what is a learned card.'

    def __init__(self, **kwargs):
        self.counter = {}  # fact_id -> number of consecutive correct guesses
        self.values = defaultdict(lambda: 0)

    def update(self, record):
        if record.is_new_fact and not record.response:
            # only consider facts not known before
            self.counter[record.fact_id] = 0
        if record.fact_id not in self.counter:
            return

        if record.response:
            self.counter[record.fact_id] += 1
        else:
            if self.counter[record.fact_id] >= 2:
                self.values[record.date.date()] += 1
            self.counter[record.fact_id] = 0


def infer_repetition_model(session, user: User, record: Record = None, date_end: datetime = None) -> str:
    # find out the last repetition model that the user used before date_end
    # given a dictionary of params, infer what repetition model is used
    if record is None:
        records = session.query(Record).\
            filter(Record.user_id == user.user_id)
        if date_end is not None:
            records = records.filter(Record.date <= date_end)
        record = records.order_by(Record.date.desc()).first()
    if record is None:
        return None
    
    params = json.loads(record.scheduler_snapshot)
    if params['qrep'] == 0:
        if params['leitner'] > 0:
            return 'leitner'
        elif params['sm2'] > 0:
            return 'sm2'
        else:
            return 'unknown'
    else:
        if 'recall_target' in params:
            return 'karl-' + str(params['recall_target'] * 100)
        else:
            return 'karl-100'


def compute_accumulative_metrics(
    session,
    metric_class_list,
    date_start: datetime = None,
    date_end: datetime = None,
    date_bin_size: int = 3,
):
    '''
    User-specific metrics.
    '''
    n_users = session.query(User).count()

    correct_on_first_try = {}  # user_id -> {fact_id -> bool}
    for user in tqdm(session.query(User), total=n_users):
        correct_on_first_try[user.user_id] = {}
        for record in user.records:
            if record.fact_id in correct_on_first_try[user.user_id]:
                continue
            correct_on_first_try[user.user_id][record.fact_id] = record.response

    if date_start is None:
        date_start = session.query(Record).order_by(Record.date).first().date.date()
    if date_end is None:
        date_end = session.query(Record).order_by(Record.date.desc()).first().date.date()

    rows = []
    for user in tqdm(session.query(User), total=n_users):
        if len(user.records) == 0:
            continue

        # initialize user specific metrics
        metrics = [metric_class(correct_on_first_try=correct_on_first_try) for metric_class in metric_class_list]

        for record in session.query(Record).\
            filter(Record.user_id == user.user_id).\
            filter(Record.date >= date_start).\
            filter(Record.date <= date_end):
            for m in metrics:
                m.update(record)

        repetition_model = infer_repetition_model(session, user=user, date_end=date_end)
        for m in metrics:
            for date, value in m.items():
                rows.append({
                    'user_id': user.user_id,
                    'name': m.name,
                    'value': value,
                    'date': date,
                    'repetition_model': repetition_model,
                })

    df = pd.DataFrame(rows)

    # bin dates and aggregate numbers
    n_bins = (date_end - date_start).days // date_bin_size + 1
    end_dates = [date_start + i * timedelta(days=date_bin_size) for i in range(n_bins)]

    def find_date_bucket(date):
        idx = bisect.bisect(end_dates, date)
        return end_dates[min(len(end_dates) - 1, idx)]

    df['date_window_end'] = df.date.apply(find_date_bucket)
    df.date= df.date.astype(np.datetime64)
    df.date_window_end = df.date_window_end.astype(np.datetime64)
    return df

session = get_sessions()['prod']

accumulative_metric_class_list = [
    n_facts_shown,
    n_new_facts_shown,
    n_new_facts_correct,
    n_new_facts_wrong,
    n_old_facts_shown,
    n_old_facts_correct,
    n_old_facts_wrong,
    n_known_old_facts_shown,
    n_known_old_facts_correct,
    n_known_old_facts_wrong,
    n_learned,
    n_learned_but_forgotten,
]

raw_df = compute_accumulative_metrics(session, accumulative_metric_class_list)

# %%
# when we compute derivative metrics (e.g. ratios), we can either
# 1) take the sum of raw metrics first, then compute derivatives
# of compute derivatives for each user, then take the average

threshold = 20
Path(f'figures/{threshold}_plus/').mkdir(parents=True, exist_ok=True)
user_ids = [u.user_id for u in session.query(User) if len(u.previous_study) >= threshold]
df = raw_df[raw_df.user_id.isin(user_ids)]

# take the sum of users within each group
df = df.groupby(['repetition_model', 'name', 'date_window_end']).sum().reset_index()
# turn metrics into columns so it's easier to compute derivative metrics
df = df.pivot_table(index=['repetition_model', 'date_window_end'], columns='name', values='value')
df['ratio_new_correct_vs_all'] = df.n_new_facts_correct / df.n_facts_shown
df['ratio_new_wrong_vs_all'] = df.n_new_facts_wrong / df.n_facts_shown
df['ratio_old_correct_vs_all'] = df.n_old_facts_correct / df.n_facts_shown
df['ratio_old_wrong_vs_all'] = df.n_old_facts_correct / df.n_facts_shown
# df['ratio_known_vs_all'] = df.n_known_old_facts_shown / df.n_facts_shown
# df['ratio_known_vs_old'] = df.n_known_old_facts_shown / df.n_old_facts_shown
# df['ratio_known_correct_vs_known'] = df.n_known_old_facts_correct / df.n_known_old_facts_shown

# done with derivatives, convert metrics from columns to rows
df = df.stack()
df.name = 'value'
df = df.reset_index()

# for name in df.name.unique():
names = [
    'ratio_new_correct_vs_all',
    'ratio_new_wrong_vs_all',
    'ratio_old_correct_vs_all',
    'ratio_old_wrong_vs_all',
]
p = (
    ggplot(df[df.name.isin(names)])
    + geom_area(aes(x='date_window_end', y='value', fill='name'), alpha=0.75)
    + facet_wrap('repetition_model')
    + theme_fs()
    + theme(
        axis_text_x=element_text(rotation=30)
    )
)
p.draw()