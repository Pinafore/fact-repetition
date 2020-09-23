# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict
from operator import methodcaller
from plotnine import *

from karl.web import get_sessions
from karl.new_util import Record, User, theme_fs
from metrics import Metric


# %%
class DatedMetric:

    name = None
    description = None
    values = None  # date -> scalar value

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
        
        
class n_average_facts_shown(DatedMetric):
    
    name = 'n_average_facts_shown'
    description = 'Total number of facts studied on each day.'
    
    def __init__(self, **kwargs):
        self.n_total_facts_shown = n_total_facts_shown(**kwargs)
        self.n_daily_users = n_daily_users(**kwargs)
        self.values = defaultdict(lambda: 0)
    
    def update(self, record):
        self.n_total_facts_shown.update(record)
        self.n_daily_users.update(record)
        date = record.date.date()
        self.values[date] = self.n_total_facts_shown.values[date] / self.n_daily_users.values[date]


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
    # n_total_facts_shown,
    n_average_facts_shown,
]

df = get_sys_metrics(session, metric_class_list)

# %%
p = (
    ggplot(df)
    + aes(x='date', y='value', color='name')
    + geom_point()
    + geom_line()
    + theme_fs()
    + theme(
        axis_text_x=element_text(rotation=30)
    )
)
p.save('output/sys_metrics.pdf')
# %%
