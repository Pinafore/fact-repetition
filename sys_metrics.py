# %%
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict
from operator import methodcaller
from plotnine import *

from karl.web import get_sessions
from karl.new_util import Record, User, theme_fs


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


class n_daily_users(DatedMetric):
    
    name = 'n_daily_users'
    description = 'Number of users that studied at least one fact each day.'
    
    def __init__(self, **kwargs):
        self.daily_active_user_ids = defaultdict(set)  # date -> set of active users on that date
        self.values = {}
    
    def update(self, record):
        date = record.date.date()
        if date not in self.values:
            self.values[date] = 0
        if record.user.user_id not in self.daily_active_user_ids[date]:
            self.values[date] += 1
            self.daily_active_user_ids[date].add(record.user.user_id)


class n_daily_users_10_plus(DatedMetric):
    
    name = 'n_daily_users_10_plus'
    description = 'Number of users that studied at least 10 fact each day.'
    
    def __init__(self, **kwargs):
        self.daily_user_count = defaultdict(dict)  # date -> number of facts studied by each user
        self.threshold = 10
        self.values = {}
    
    def update(self, record):
        date = record.date.date()
        if date not in self.values:
            self.values[date] = 0
        if record.user_id not in self.daily_user_count[date]:
            self.daily_user_count[date][record.user_id] = 0
        self.daily_user_count[date][record.user_id] += 1
        if self.daily_user_count[date][record.user_id] == self.threshold:
            self.values[date] += 1
            

class n_daily_users_50_plus(DatedMetric):
    
    name = 'n_daily_users_50_plus'
    description = 'Number of users that studied at least 50 fact each day.'
    
    def __init__(self, **kwargs):
        self.daily_user_count = defaultdict(dict)  # date -> number of facts studied by each user
        self.threshold = 50
        self.values = {}
    
    def update(self, record):
        date = record.date.date()
        if date not in self.values:
            self.values[date] = 0
        if record.user_id not in self.daily_user_count[date]:
            self.daily_user_count[date][record.user_id] = 0
        self.daily_user_count[date][record.user_id] += 1
        if self.daily_user_count[date][record.user_id] == self.threshold:
            self.values[date] += 1
            
            
class n_daily_users_100_plus(DatedMetric):
    
    name = 'n_daily_users_100_plus'
    description = 'Number of users that studied at least 100 fact each day.'
    
    def __init__(self, **kwargs):
        self.daily_user_count = defaultdict(dict)  # date -> number of facts studied by each user
        self.threshold = 100
        self.values = {}
    
    def update(self, record):
        date = record.date.date()
        if date not in self.values:
            self.values[date] = 0
        if record.user_id not in self.daily_user_count[date]:
            self.daily_user_count[date][record.user_id] = 0
        self.daily_user_count[date][record.user_id] += 1
        if self.daily_user_count[date][record.user_id] == self.threshold:
            self.values[date] += 1

            
class n_total_facts_shown(DatedMetric):
    
    name = 'n_total_facts_shown'
    description = 'Total number of facts studied on each day.'
    
    def __init__(self, **kwargs):
        self.values = defaultdict(lambda: 0)
    
    def update(self, record):
        self.values[record.date.date()] += 1
        

def infer_repetition_model(record: Record) -> str:
    # find out the last repetition model that the user used before date_end
    # given a dictionary of params, infer what repetition model is used
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
            return 'karl-' + str(int(params['recall_target'] * 100))
        else:
            return 'karl-100'


class n_leitner_users(DatedMetric):

    name = 'n_leitner_users'
    description = 'Number of daily active Leitner users.'

    def __init__(self, **kwargs):
        self.daily_users = defaultdict(set)
        self.values = defaultdict(lambda: 0)
    
    def update(self, record):
        repetition_model = infer_repetition_model(record)
        date = record.date.date()
        if repetition_model == 'leitner' and record.user_id not in self.daily_users[date]:
            self.daily_users[date].add(record.user_id)
            self.values[date] += 1


class n_sm2_users(DatedMetric):

    name = 'n_sm2_users'
    description = 'Number of daily active SM2 users.'

    def __init__(self, **kwargs):
        self.daily_users = defaultdict(set)
        self.values = defaultdict(lambda: 0)
    
    def update(self, record):
        repetition_model = infer_repetition_model(record)
        date = record.date.date()
        if repetition_model == 'sm2' and record.user_id not in self.daily_users[date]:
            self.daily_users[date].add(record.user_id)
            self.values[date] += 1


class n_karl100_users(DatedMetric):

    name = 'n_karl100_users'
    description = 'Number of daily active karl100 users.'

    def __init__(self, **kwargs):
        self.daily_users = defaultdict(set)
        self.values = defaultdict(lambda: 0)
    
    def update(self, record):
        repetition_model = infer_repetition_model(record)
        date = record.date.date()
        if repetition_model == 'karl-100' and record.user_id not in self.daily_users[date]:
            self.daily_users[date].add(record.user_id)
            self.values[date] += 1


class n_karl85_users(DatedMetric):

    name = 'n_karl85_users'
    description = 'Number of daily active karl85 users.'

    def __init__(self, **kwargs):
        self.daily_users = defaultdict(set)
        self.values = defaultdict(lambda: 0)
    
    def update(self, record):
        repetition_model = infer_repetition_model(record)
        date = record.date.date()
        if repetition_model == 'karl-85' and record.user_id not in self.daily_users[date]:
            self.daily_users[date].add(record.user_id)
            self.values[date] += 1


class n_karl50_users(DatedMetric):

    name = 'n_karl50_users'
    description = 'Number of daily active karl85 users.'

    def __init__(self, **kwargs):
        self.daily_users = defaultdict(set)
        self.values = defaultdict(lambda: 0)
    
    def update(self, record):
        repetition_model = infer_repetition_model(record)
        date = record.date.date()
        if repetition_model == 'karl-50' and record.user_id not in self.daily_users[date]:
            self.daily_users[date].add(record.user_id)
            self.values[date] += 1


def get_sys_metrics(
    session,
    metric_class_list,
    date_start: datetime = None,
    date_end: datetime = None,
):
        
    records = session.query(Record)
    if date_start is not None:
        records = records.filter(Record.date >= date_start)
    if date_end is not None:
        records = records.filter(Record.date <= date_end)
    n_records = records.count()
    
    metrics = [m() for m in metric_class_list]
    for record in tqdm(records, total=n_records):
        for m in metrics:
            m.update(record)
            
    rows = []
    for m in metrics:
        for date, value in m.values.items():
            rows.append({
                'date': date,
                'name': m.name,
                'value': value,
            })
    df = pd.DataFrame(rows)
    df.date = df.date.astype(np.datetime64)
    return df

# %%
session = get_sessions()['prod']

metric_class_list = [
    n_daily_users,
    n_daily_users_10_plus,
    n_daily_users_50_plus,
    n_daily_users_100_plus,
    n_leitner_users,
    n_sm2_users,
    n_karl100_users,
    n_karl85_users,
    n_karl50_users,
]

df = get_sys_metrics(session, metric_class_list)

# %%
names = [m.name for m in [
    n_leitner_users,
    n_sm2_users,
    n_karl100_users,
    n_karl85_users,
    n_karl50_users,
]]
p = (
    ggplot(df[df.name.isin(names)])
    + aes(x='date', y='value', color='name')
    + geom_point()
    + geom_line()
    + theme_fs()
    + theme(
        axis_text_x=element_text(rotation=30)
    )
)
p.draw()
# p.save('output/sys_metrics.pdf')
# %%
p = (
    ggplot(df[df.name.isin(names)])
    + aes(x='date', y='value', color='name')
    + geom_area()
    + theme_fs()
    + theme(
        axis_text_x=element_text(rotation=30)
    )
)
p.draw()
# %%
