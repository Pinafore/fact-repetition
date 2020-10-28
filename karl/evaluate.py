# %%
'''Use collected data (stored in PostgreSQL) to evaluate retention model'''
import pandas as pd
import altair as alt
from tqdm import tqdm
import sqlalchemy as sa

from karl.web import get_sessions
from karl.models import User
from karl.retention.baseline import RetentionModel


session = get_sessions()['prod']
retention_model = RetentionModel()
# %%
rows = []
users = session.query(User)
for user in tqdm(users, total=users.count()):
    user_records = user.records
    if not sa.inspect(user).detached:
        session.expunge(user)
    user.records = user_records
    for record in user.records:
        fact = record.fact
        if not sa.inspect(fact).detached:
            session.expunge(fact)

        fact.results = record.fact_snapshot.results.get(fact.fact_id, [])

        user.previous_study = record.user_snapshot.previous_study
        user.results = record.user_snapshot.results
        user.count_correct_before = record.user_snapshot.count_correct_before
        user.count_wrong_before = record.user_snapshot.count_wrong_before
        user.params = record.user_snapshot.params

        # predict
        prob = retention_model.predict_one(user, fact, record.date)
        features, features_dict = retention_model.compute_features(
            user, fact, record.date
        )

        row = {
            'user_id': user.user_id,
            'fact_id': fact.fact_id,
            'record_id': record.record_id,
            'result': record.response,
            'prediction': prob,
        }
        row.update(features_dict)
        rows.append(row)
# %%
df = pd.DataFrame(rows)
df['prediction_binary'] = df.prediction.apply(lambda x: x > 0.5)
df['accuracy'] = (df.prediction_binary == df.result)
# %%
line = alt.Chart().mark_line().encode(
    alt.X('prediction:Q', bin=alt.Bin(maxbins=20)),
    alt.Y('mean(result):Q')
)
diag = alt.Chart().mark_line(strokeDash=[1, 1]).encode(
    alt.X('prediction:Q', bin=alt.Bin(maxbins=20)),
    alt.Y('prediction:Q', bin=alt.Bin(maxbins=20), title=None, axis=None),
)
bar = alt.Chart().mark_bar(opacity=0.3).encode(
    alt.X('prediction:Q', bin=alt.Bin(maxbins=20)),
    alt.Y('count()')
)

source = df[df.user_gap_from_previous < 1500]

chart = alt.layer(
    line, diag, bar, data=source,
).resolve_scale(
    y='independent'
).properties(
    height=180, width=300,
).facet(
    alt.Facet('user_gap_from_previous', bin=alt.Bin(maxbins=10)),
    columns=2,
).resolve_scale(
    y='shared'
).resolve_axis(
    y='independent'
)

chart.save('test.json')
# %%
