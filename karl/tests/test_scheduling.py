# %%
import json
import pickle
import random
import requests
import numpy as np
from datetime import timedelta
from dateutil.parser import parse as parse_date

from karl.schemas import ScheduleRequestSchema, UpdateRequestSchema, ParametersSchema
from karl.config import settings


with open(f'{settings.DATA_DIR}/diagnostic_questions.pkl', 'rb') as f:
    diagnostic_facts = pickle.load(f)

URL = 'http://127.0.0.1:8000/api/karl'
n_days = 1
n_facts_per_day = 10
n_facts_per_query = 300
user_id = 'dummy'
start_date = parse_date('2028-06-01 08:00:00.000001 -0400')

# requests.get(f'{URL}/reset_user?env=prod&user_id={user_id}')
requests.put(f'{URL}/set_params?env=prod&user_id={user_id}', data=json.dumps(ParametersSchema().__dict__))

for nth_day in range(n_days):
    print(f'day {nth_day}')
    seconds_offset = 0  # offset from the the first fact of today, in seconds
    for nth_fact in range(n_facts_per_day):
        current_date = start_date + timedelta(days=nth_day) + timedelta(seconds=seconds_offset)
        print(current_date.strftime('%Y-%m-%dT%H:%M:%S%z'))

        schedule_requests = [
            ScheduleRequestSchema(
                text=fact['text'],
                date=current_date.strftime('%Y-%m-%dT%H:%M:%S%z'),
                answer=fact['answer'],
                category=fact['category'],
                user_id=user_id,
                fact_id='sim_' + str(fact['fact_id']),
            )
            for fact in random.sample(diagnostic_facts, n_facts_per_query)
        ]
        schedule_response = json.loads(
            requests.post(
                f'{URL}/schedule?env=prod',
                data=json.dumps([r.__dict__ for r in schedule_requests])
            ).text
        )

        index = schedule_response['order'][0]
        fact_id = schedule_requests[index].fact_id
        debug_id = schedule_response['debug_id']
        print(fact_id, debug_id)

        response = bool(np.random.binomial(1, 0.5))

        update_request = UpdateRequestSchema(
            text=schedule_requests[index].text,
            date=current_date.strftime('%Y-%m-%dT%H:%M:%S%z'),
            answer=schedule_requests[index].answer,
            category=schedule_requests[index].category,
            user_id=user_id,
            fact_id=schedule_requests[index].fact_id,
            label=response,
            history_id=f'sim_history_{nth_day}_{nth_fact}',
            elapsed_milliseconds_text=10000,
            elapsed_milliseconds_answer=10000,
            debug_id=debug_id,
        )
        update_response = json.loads(
            requests.post(
                f'{URL}/update?env=prod',
                data=json.dumps([update_request.__dict__]),
            ).text
        )

        seconds_offset += 20

# %%
