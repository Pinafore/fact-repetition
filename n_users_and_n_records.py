# %%
import pandas as pd
import altair as alt
from karl.new_util import User, Record, parse_date
from karl.scheduler import MovingAvgScheduler
from karl.web import get_sessions

session = get_sessions()['prod']
# %%
rows = []
user_ids = set()  # to check if its new user
date_start = parse_date('2020-08-23')
records = session.query(Record).\
        order_by(Record.date).\
        filter(Record.date >= date_start)
n_records = 0
curr_date = records.first().date.date()
for record in records:
    date = record.date.date()
    user_ids.add(record.user_id)
    if date != curr_date:
        rows.append({
            'n_users': len(user_ids),
            'n_records': n_records,
            'date': curr_date,
        })
        n_records = 0
        curr_date = date
    else:
        n_records += 1
if n_records > 0:
    rows.append({
        'n_users': len(user_ids),
        'n_records': n_records,
        'date': curr_date,
    })
df = pd.DataFrame(rows)
df.date = pd.to_datetime(df.date)
df.n_records = df.n_records.cumsum()

df.set_index('date', inplace=True)
# %%
source = df.reset_index().melt('date')

lines = alt.Chart().mark_line().encode(
    x='date',
    y='value:Q',
    color='variable:N'
).properties(
    height=200,
    width=400,
)

nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['date'], empty='none')
selectors = alt.Chart().mark_point().encode(
    x='date',
    opacity=alt.value(0),
).add_selection(
    nearest
)

points = lines.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
)

text = lines.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, 'value:Q', alt.value(' '))
)

rules = alt.Chart().mark_rule(color='gray').encode(
    x='date',
).transform_filter(
    nearest
)

chart = alt.layer(lines, selectors, points, text, rules).facet(
    row='variable:N',
    data=source,
).resolve_scale(
    y='independent'
)

output_path = '/fs/clip-quiz/shifeng/ihsgnef.github.io/images'
chart.save(f'{output_path}/n_users_and_n_records.json')
# %%
