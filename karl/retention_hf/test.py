# %%
import json
import requests

from karl.db.session import SessionLocal
from karl.retention_hf.data import vectors_to_features
from karl.models import User, UserFeatureVector, CardFeatureVector, UserCardFeatureVector


# %%
session = SessionLocal()
new_records = [x for x in session.query(User).get('463').records if x.is_new_fact]
old_records = [x for x in session.query(User).get('463').records if not x.is_new_fact]
feature_vectors = []
for record in new_records[:10] + old_records[:10]:
    v_user = session.query(UserFeatureVector).get(record.id)
    v_card = session.query(CardFeatureVector).get(record.id)
    v_usercard = session.query(UserCardFeatureVector).get(record.id)
    elapsed_milliseconds = record.elapsed_milliseconds_text + record.elapsed_milliseconds_answer
    feature_vectors.append(vectors_to_features(
        v_usercard, v_user, v_card, record.date, record.card.text, elapsed_milliseconds
    ))

feature_vectors = [x.__dict__ for x in feature_vectors]
for x in feature_vectors:
    x['utc_date'] = str(x['utc_date'])

scores = json.loads(
    requests.get(
        'http://127.0.0.1:8001/api/karl/predict',
        data=json.dumps(feature_vectors)
    ).text
)

print(scores)
