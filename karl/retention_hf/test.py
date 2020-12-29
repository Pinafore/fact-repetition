# %%
import os
import numpy as np
import altair as alt
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from karl.retention_hf.data import _get_user_features
from karl.db.session import SessionLocal, engine
from karl.config import settings
from karl.models import User

alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')


def save_chart_and_pdf(chart, path):
    chart.save(f'{path}.json')
    os.system(f'vl2vg {path}.json | vg2pdf > {path}.pdf')


def get_retention_features_df():
    session = SessionLocal()
    # gather features
    futures = []
    executor = ProcessPoolExecutor(
        mp_context=multiprocessing.get_context(settings.MP_CONTEXT),
        initializer=engine.dispose,
    )
    for user in session.query(User):
        if not user.id.isdigit() or len(user.records) == 0:
            continue
        futures.append(executor.submit(_get_user_features, user.id))

    features, labels = [], []
    for future in futures:
        f1, f2 = future.result()
        features.extend(f1)
        labels.extend(f2)

    df = []
    for feature, label in zip(features, labels):
        row = feature.__dict__
        row['label'] = label
        df.append(row)
    df = pd.DataFrame(df)
    return df


def figure_recall_vs_delta(df, facet_by='sm2_repetition'):
    # source = df.loc[np.random.choice(df.index, 3000, replace=False)]
    source = df[df.usercard_delta != 0]
    source['usercard_delta_binned'] = pd.qcut(source.usercard_delta, q=20, labels=False)
    chart = alt.Chart(source).mark_line().encode(
        alt.X('usercard_delta_binned:Q'),
        alt.Y('mean(label):Q'),
        # color='leitner_box:N'
    ).facet(
        facet_by,
        rows=1,
    )
    save_chart_and_pdf(chart, f'recall_vs_delta_by_{facet_by}')

# %%
if __name__ == '__main__':
    df = get_retention_features_df()
    figure_recall_vs_delta(df, facet_by='sm2_repetition')
    figure_recall_vs_delta(df, facet_by='leitner_box')

# %%
df = get_retention_features_df()
# %%
# source = df.loc[np.random.choice(df.index, 3000, replace=False)]
# source = source[source.usercard_delta != 0]
source = df[df.usercard_delta != 0]
source['usercard_delta_binned'] = pd.qcut(source.usercard_delta, q=20, labels=False)

facet_by = 'sm2_repetition'

line = alt.Chart().mark_line().encode(
    alt.X('usercard_delta_binned:Q'),
    alt.Y('mean(label):Q'),
)
band = alt.Chart().mark_errorband(extent='ci').encode(
    alt.X('usercard_delta_binned:Q'),
    alt.Y('label:Q'),
)
density = alt.Chart().transform_density(
    'usercard_delta_binned',
    groupby=[facet_by],
    as_=['usercard_delta_binned', 'density'],
    extent=[0, 20],
).mark_area().encode(
    alt.X('usercard_delta_binned:Q'),
    alt.Y('density:Q')
)
chart = alt.layer(
    band, line, density, data=source
).facet(
    facet_by
)
save_chart_and_pdf(chart, 'test')
