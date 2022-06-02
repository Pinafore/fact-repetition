import json
import pickle
import random
import requests
import numpy as np
from datetime import timedelta
from dateutil.parser import parse as parse_date

from karl.schemas import ScheduleRequestSchema, UpdateRequestSchema, ParametersSchema
from karl.schemas import KarlFactV2, ScheduleRequestV2, UpdateRequestV2
from karl.config import settings


with open(f'{settings.DATA_DIR}/diagnostic_questions.pkl', 'rb') as f:
    diagnostic_facts = pickle.load(f)

URL = f'{settings.API_URL}/api/karl'
n_days = 1
n_facts_per_day = 10
n_facts_per_query = 300
user_id = 'dummy'
start_date = parse_date('2028-06-01 08:00:00.000001 -0400')

requests.get(f'{URL}/reset_user?user_id={user_id}')
requests.put(f'{URL}/set_params?user_id={user_id}', data=json.dumps(ParametersSchema().dict()))

profile = {}  # key -> [values]

for nth_day in range(n_days):
    print(f'day {nth_day}')
    seconds_offset = 0  # offset from the the first fact of today, in seconds
    for nth_fact in range(n_facts_per_day):
        current_date = start_date + timedelta(days=nth_day) + timedelta(seconds=seconds_offset)
        print(current_date.strftime('%Y-%m-%dT%H:%M:%S%z'))

        facts = [
            KarlFactV2(
                fact_id=fact['fact_id'] + 1000000,
                text=fact['text'],
                answer=fact['answer'],
                deck_name='dummy',
                deck_id=1000000,
                category=fact['category'],
            )
            for fact in random.sample(diagnostic_facts, n_facts_per_query)
        ]
        schedule_request = ScheduleRequestV2(
            facts=facts,
            repetition_model='karl',
            user_id=user_id,
        )
        schedule_response = json.loads(
            requests.post(
                f'{URL}/schedule_v2',
                data=json.dumps(schedule_request.dict())
            ).text
        )

        for key, value in schedule_response.get('profile', {}).items():
            if key not in profile:
                profile[key] = []
            profile[key].append(value)

        index = schedule_response['order'][0]
        fact_id = schedule_request.facts[index].fact_id
        debug_id = schedule_response['debug_id']
        print(fact_id, debug_id)

        response = bool(np.random.binomial(1, 0.5))

        update_request = UpdateRequestV2(
            user_id=user_id,
            fact_id=schedule_request.facts[index].fact_id,
            label=response,
            deck_name='dummy',
            deck_id=1000000,
            elapsed_milliseconds_text=10000,
            elapsed_milliseconds_answer=10000,
            debug_id=debug_id,
            history_id=f'sim_history_{nth_day}_{nth_fact}',
            studyset_id='dummy_studyset_' + debug_id,
            test_mode=False,
        )
        update_response = json.loads(
            requests.post(
                f'{URL}/update_v2',
                data=json.dumps(update_request.dict()),
            ).text
        )

        for key, value in update_response.get('profile', {}).items():
            if key not in profile:
                profile[key] = []
            profile[key].append(value)

        seconds_offset += 20

print()
print()
for key, values in profile.items():
    print(key, np.mean(values))
