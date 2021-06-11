import os
import json
import requests
import pandas as pd
import altair as alt
from tqdm import tqdm
from copy import deepcopy
from typing import Optional
from datetime import datetime, timedelta
from karl.retention_hf.data import get_retention_features_df, RetentionFeaturesSchema
from karl.figures import save_chart_and_pdf

alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype')


df = get_retention_features_df()
df = df.groupby('leitner_box').sample(frac=0.01)
print(df.groupby('leitner_box').count())

def predict(x):
    x = deepcopy(x.__dict__)
    x['utc_date'] = str(x['utc_date'])
    x['utc_datetime'] = str(x['utc_datetime'])
    return json.loads(
        requests.get(
            'http://127.0.0.1:8001/api/karl/predict_one',
            data=json.dumps(x)
        ).text
    )[0]


def predict_batch(xs):
    xs = deepcopy([x.__dict__ for x in xs])
    for x in xs:
        x['utc_date'] = str(x['utc_date'])
        x['utc_datetime'] = str(x['utc_datetime'])

    return json.loads(
        requests.get(
            'http://127.0.0.1:8001/api/karl/predict',
            data=json.dumps(xs)
        ).text
    )


def update_leitner(
    v_usercard: RetentionFeaturesSchema,
    response: bool,
    date: datetime,
) -> None:
    # leitner boxes 1~10
    # days[0] = None as placeholder since we don't have box 0
    # days[9] and days[10] = 999 to make it never repeat
    days = [0, 0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 999, 999]
    increment_days = {i: x for i, x in enumerate(days)}

    if v_usercard.leitner_box is None:
        # boxes: 1 ~ 10
        v_usercard.leitner_box = 1

    v_usercard.leitner_box += (1 if response else -1)
    v_usercard.leitner_box = max(min(v_usercard.leitner_box, 10), 1)
    interval = timedelta(days=increment_days[v_usercard.leitner_box])
    return date + interval


def update_sm2(
    v_usercard: RetentionFeaturesSchema,
    response: bool,
    date: datetime,
) -> None:
    def get_quality_from_response(response: bool) -> int:
        return 4 if response else 1

    if v_usercard.sm2_repetition is None:
        v_usercard.sm2_repetition = 0
        v_usercard.sm2_efactor = 2.5
        v_usercard.sm2_interval = 1
        v_usercard.sm2_repetition = 0

    q = get_quality_from_response(response)
    v_usercard.sm2_repetition += 1
    v_usercard.sm2_efactor = max(1.3, v_usercard.sm2_efactor + 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02))

    if not response:
        v_usercard.sm2_interval = 0
        v_usercard.sm2_repetition = 0
    else:
        if v_usercard.sm2_repetition == 1:
            v_usercard.sm2_interval = 1
        elif v_usercard.sm2_repetition == 2:
            v_usercard.sm2_interval = 6
        else:
            v_usercard.sm2_interval *= v_usercard.sm2_efactor

    v_usercard.sm2_interval = min(500, v_usercard.sm2_interval)
    return date + timedelta(days=v_usercard.sm2_interval)


def update_feature_vector(
    v: RetentionFeaturesSchema,
    response: bool,
    date: datetime,
    previous_study_date: Optional[datetime],
):
    v = deepcopy(v)

    usercard_delta = None
    if previous_study_date is not None:
        usercard_delta = (date - previous_study_date).total_seconds()
    v.usercard_n_study_positive += response
    v.usercard_n_study_negative += (not response)
    v.usercard_n_study_total += 1
    v.usercard_delta_previous = usercard_delta
    v.usercard_previous_study_response = response
    if v.correct_on_first_try is None:
        v.correct_on_first_try = response

    # update leitner
    leitner_scheduled_date = update_leitner(v, response, date)
    # update sm2
    sm2_scheduled_date = update_sm2(v, response, date)

    v.user_n_study_positive += response
    v.user_n_study_negative += (not response)
    v.user_n_study_total += 1

    v.card_n_study_positive += response
    v.card_n_study_negative += (not response)
    v.card_n_study_total += 1

    return v, leitner_scheduled_date, sm2_scheduled_date


if os.path.exists('sensitivity.h5'):
    source = pd.read_hdf('sensitivity.h5', 'df')
else:
    rows = []
    for row_dict in tqdm(df.to_dict(orient="records")):
        x = RetentionFeaturesSchema(**row_dict)

        if x.is_new_fact:
            continue

        n_intervals = 10
        previous_study_date = x.utc_datetime - timedelta(seconds=x.usercard_delta * 3600)
        interval = timedelta(days=5)

        leitner_scheduled_date = x.utc_datetime + timedelta(seconds=x.delta_to_leitner_scheduled_date * 3600)
        sm2_scheduled_date = x.utc_datetime + timedelta(seconds=x.delta_to_sm2_scheduled_date * 3600)

        original_p_recall = predict(x)

        # NOTE
        if original_p_recall > 0.7:
            continue

        # before the study
        xs = []
        for i in range(n_intervals + 1):
            date = previous_study_date + i * interval
            x.usercard_delta = (date - previous_study_date).total_seconds() // 3600
            x.delta_to_leitner_scheduled_date = (leitner_scheduled_date - date).total_seconds() // 3600
            x.delta_to_sm2_scheduled_date = (sm2_scheduled_date - date).total_seconds() // 3600
            xs.append(deepcopy(x))

        ys = predict_batch(xs)
        for i, y in enumerate(ys):
            rows.append({
                'index': i - n_intervals,
                'user_id': x.user_id,
                'card_id': x.card_id,
                'record_id': row_dict['record_id'],
                'leitner_box': row_dict['leitner_box'],
                'p_recall': y,
                'original_p_recall': original_p_recall,
                'type': 'before',
            })

        # study
        true_previous_study_date = previous_study_date
        previous_study_date = x.utc_datetime

        # assume study is correct
        xs = []
        v, leitner_scheduled_date, sm2_scheduled_date = update_feature_vector(x, True, x.utc_datetime, true_previous_study_date)
        for i in range(n_intervals + 1):
            date = previous_study_date + i * interval
            v.usercard_delta = (date - previous_study_date).total_seconds() // 3600
            v.delta_to_leitner_scheduled_date = (leitner_scheduled_date - date).total_seconds() // 3600
            v.delta_to_sm2_scheduled_date = (sm2_scheduled_date - date).total_seconds() // 3600
            xs.append(deepcopy(v))

        ys = predict_batch(xs)
        for i, y in enumerate(ys):
            rows.append({
                'index': i,
                'user_id': x.user_id,
                'card_id': x.card_id,
                'record_id': row_dict['record_id'],
                'leitner_box': row_dict['leitner_box'],
                'p_recall': y,
                'original_p_recall': original_p_recall,
                'type': 'correct',
            })

        # assume study is wrong
        xs = []
        v, leitner_scheduled_date, sm2_scheduled_date = update_feature_vector(x, False, x.utc_datetime, true_previous_study_date)
        for i in range(n_intervals + 1):
            date = previous_study_date + i * interval
            v.usercard_delta = (date - previous_study_date).total_seconds() // 3600
            v.delta_to_leitner_scheduled_date = (leitner_scheduled_date - date).total_seconds() // 3600
            v.delta_to_sm2_scheduled_date = (sm2_scheduled_date - date).total_seconds() // 3600
            xs.append(deepcopy(v))

        ys = predict_batch(xs)
        for i, y in enumerate(ys):
            rows.append({
                'index': i,
                'user_id': x.user_id,
                'card_id': x.card_id,
                'record_id': row_dict['record_id'],
                'leitner_box': row_dict['leitner_box'],
                'p_recall': y,
                'original_p_recall': original_p_recall,
                'type': 'wrong',
            })

    source = pd.DataFrame(rows)
    source['original_p_recall_binned'] = pd.qcut(source.original_p_recall, 4, duplicates='drop')
    source['original_p_recall_binned'] = source['original_p_recall_binned'].transform(lambda x: x.left)
    # source.to_hdf('sensitivity.h5', key='df', mode='w')

print(source.groupby('leitner_box').count())

line = alt.Chart().mark_line().encode(
    alt.X('index', title='Days till study'),
    alt.Y(
        'mean(p_recall)',
        title='Predicted recall probability',
        scale=alt.Scale(domain=[0.5, 0.75]),
    ),
    color=alt.Color('type'),
)
band = alt.Chart().mark_errorband(extent='ci', opacity=0.2).encode(
    alt.X('index', title='Days till study'),
    alt.Y(
        'p_recall',
        title='Predicted recall probability',
        scale=alt.Scale(domain=[0.5, 0.75]),
    ),
    color=alt.Color('type'),
)
chart = alt.layer(
    line, band,
    data=source,
).properties(
    width=160,
    height=160,
).facet(
    facet=alt.Facet(
        'original_p_recall_binned',
        title='Predicted recall probability at the time of the study',
    ),
    columns=4,
)
save_chart_and_pdf(chart, '/fs/www-users/shifeng/files/sensitivity', to_pdf=True)
