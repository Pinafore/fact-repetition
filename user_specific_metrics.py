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
from plotnine import ggplot, ggtitle, aes, theme,\
    geom_point, geom_line, geom_area, geom_ribbon, geom_bar, geom_density, \
    facet_grid, facet_wrap,\
    element_text, scale_fill_brewer, scale_fill_manual,\
    scale_x_log10, scale_y_log10

from karl.new_util import User, Record, parse_date, theme_fs
from karl.web import get_sessions

session = get_sessions()['prod']
users = session.query(User).all()
user = users[3]
# %%
correct_on_first_try = {}
rows = []
for record in user.records:
    if record.fact_id not in correct_on_first_try:
        correct_on_first_try[record.fact_id] = record.response
    elapsed_seconds = record.elapsed_milliseconds_text / 1000
    elapsed_seconds += record.elapsed_milliseconds_answer / 1000
    elapsed_minutes = elapsed_seconds / 60

    rows.append({
        'record_id': record.record_id,
        'user_id': user.user_id,
        'fact_id': record.fact_id,
        'repetition_model': json.loads(record.scheduler_snapshot)['repetition_model'],
        'is_new_fact': record.is_new_fact,
        'result': record.response,
        'date': record.date.date(),
        'datetime': record.date,
        'elapsed_minutes': elapsed_minutes,
        'is_known_fact': correct_on_first_try[record.fact_id],
        'leitner_box': json.loads(record.user_snapshot)['leitner_box'][record.fact_id],
    })
df = pd.DataFrame(rows).sort_values('datetime', axis=0)

for i in df.leitner_box.unique():
    if i > 1:
        # initial box means first occurrence; it doesn't count
        # not previously-known facts that got improved to level-i familiarity 
        df[f'n_learned_{i}'] = df.apply(lambda x: (x.leitner_box == i) & x.result & ~x.is_known_fact, axis=1)
        df[f'n_learned_{i}_csum'] = df.groupby('user_id')[f'n_learned_{i}'].cumsum()

x_axis_name = 'datetime'
repetition_model = json.loads(user.records[-1].scheduler_snapshot)['repetition_model']
leitner_boxes = sorted([i for i in df.leitner_box.unique() if i > 1])
# reverse because lower sequential values usually corresponds to darker colors
leitner_boxes = [f'n_learned_{i}_csum' for i in leitner_boxes][::-1]
# leitner_box_type = CategoricalDtype(categories=leitner_boxes, ordered=True)
df_plot = pd.melt(
    df,
    id_vars=x_axis_name,
    value_vars=leitner_boxes,
    var_name='name',
    value_name='value',
).reset_index()
# df_plot.name = df_plot.name.astype(leitner_box_type)

df_plot['date'] = df_plot['datetime'].apply(lambda x: x.date())
df_minutes = df[df.user_id == user.user_id][['datetime', 'elapsed_minutes']]
df_plot = pd.merge(df_plot, df_minutes, how='left', on='datetime')

df_plot.date = pd.to_datetime(df_plot.date)
df_plot.datetime = pd.to_datetime(df_plot.datetime)

base = alt.Chart().encode(
    alt.X('date', axis=alt.Axis(title='Date'))
)
bar = base.mark_bar(opacity=0.4, color='#57A44C').encode(
    alt.Y(
        'sum(elapsed_minutes)',
        axis=alt.Axis(title='Minutes spent', titleColor='#57A44C'),
    )
)
line = base.mark_line().encode(
    alt.Y('value', axis=alt.Axis(title='Familiarity')),
    color='name',
)
chart = alt.layer(bar, line, data=df_plot).resolve_scale(
    y='independent',
)

file_name = f'figures/user_study_reports/{user.user_id}.json'
print(file_name)
# pdf_name = f'figures/user_study_reports/{user.user_id}.pdf'
chart.save(file_name)
# os.system(f'vl2vg {file_name} | vg2pdf > {pdf_name}')
# chart
# %%
import altair as alt
from vega_datasets import data

source = data.stocks()

highlight = alt.selection(
    type='single', on='mouseover', fields=['symbol'], nearest=True
)

base = alt.Chart(source).encode(
    x='date:T',
    y='price:Q',
    color='symbol:N'
)

points = base.mark_circle().encode(
    opacity=alt.value(0)
).add_selection(
    highlight
)

lines = base.mark_line().encode(
    size=alt.condition(~highlight, alt.value(1), alt.value(3))
)

chart = points + lines
chart.save(file_name)
# %%
