# %%
import pytz
import altair as alt
from karl.retention_hf.data import get_retention_features_df
from karl.figures import save_chart_and_pdf
from dateutil.parser import parse as parse_date

alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')
# %%
def figure_fret_user(
    source,
    user_id,
):
    source = source[source.user_id == user_id].reset_index(drop=True)
    card_ids = {}  # card_id -> order of appearance
    for x in source.card_id:
        if x in card_ids:
            continue
        else:
            card_ids[x] = len(card_ids)
    source['converted_card_id'] = source.card_id.apply(lambda x: int(card_ids[x]))
    source = source.reset_index()
    source = source.drop('utc_date', axis=1).reset_index()
    chart = alt.Chart(source).mark_circle(size=60).encode(
        x='converted_card_id:Q',
        y='utc_datetime:T',
        color='response',
    ).properties(
        width=800,
        height=2000,
        title=source.user_id[0] + ' ' + source.repetition_model[0]
    )    
    save_chart_and_pdf(chart, f'/fs/www-users/shifeng/files/fret')


# %%
date_start = '2020-12-20'
date_start = parse_date(date_start).astimezone(pytz.utc).date()

df = get_retention_features_df()
# df = df[df.utc_date >= date_start]
# print(df.groupby('user_id').size().sort_values(ascending=False)[:20])
# %%
figure_fret_user(source=df, user_id='617')
# %%
# user_id
# 463    9173
# 38     7441
# 496    5990
# 123    4765
# 85     4733
# 413    3527
# 45     2209
# 46     2167
# 564    2123
# 288    2018
# 105    1742
# 68     1523
# 229    1368
# 136    1318
# 343    1308
# 328    1297
# 294    1146
# 93     1129
# 138    1058
# 270    1033