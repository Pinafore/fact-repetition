'''
Test the following:
1. schedule request with empty list of facts
2. schedule once, study multiple cards
3. test mode updates
'''
import json
import pickle
import random
import requests
import numpy as np
from datetime import timedelta
from dateutil.parser import parse as parse_date

from karl.schemas import ParametersSchema, RecallTarget
from karl.schemas import KarlFactSchema, ScheduleRequestSchema, UpdateRequestSchema
from karl.config import settings


with open(f'{settings.DATA_DIR}/diagnostic_questions.pkl', 'rb') as f:
    diagnostic_facts = pickle.load(f)

URL = f'{settings.API_URL}/api/karl'
n_days = 1
n_facts_per_day = 20
n_facts_per_query = 1000
user_id = 'dummy'
start_date = parse_date('2028-06-01 08:00:00.000001 -0400')

requests.get(f'{URL}/reset_user?user_id={user_id}')
requests.put(f'{URL}/set_params?user_id={user_id}', data=json.dumps(ParametersSchema().dict()))

# schedule requests with empty list of facts
schedule_request = ScheduleRequestSchema(
    facts=[],
    repetition_model='fsrs',
    user_id=user_id,
    recall_target=RecallTarget(target=0.8, target_window_lowest=0, target_window_highest=1.0)
)
schedule_response = json.loads(
    requests.post(
        f'{URL}/schedule_v3',
        data=json.dumps(schedule_request.dict())
    ).text
)
print(schedule_response)
print()
print()


# each day, the dummy student issues one schedule request with n_facts_per_query cards
# the student then studies n_facts_per_day cards in the order returned by the scheduler
profile = {}  # key -> [values]
for nth_day in range(n_days):
    print(f'day {nth_day}')

    # randomly sample n_facts_per_query
    facts = [
        KarlFactSchema(
            fact_id=fact['fact_id'] + 1000000,
            text=fact['text'],
            answer=fact['answer'],
            deck_name='dummy',
            deck_id=1000000,
            category=fact['category'],
        )
        for fact in random.sample(diagnostic_facts, n_facts_per_query)
    ]

    # schedule request for the day
    schedule_request = ScheduleRequestSchema(
        facts=facts,
        repetition_model='fsrs',
        user_id=user_id,
        recall_target=RecallTarget(target=0.8, target_window_lowest=0, target_window_highest=1.0)
    )
    schedule_response = json.loads(
        requests.post(
            f'{URL}/schedule_v3',
            data=json.dumps(schedule_request.dict())
        ).text
    )

    for key, value in schedule_response.get('profile', {}).items():
        if key not in profile:
            profile[key] = []
        profile[key].append(value)

    seconds_offset = 0  # offset from the the first fact of today, in seconds
    for nth_fact in range(n_facts_per_day):
        current_date = start_date + timedelta(days=nth_day) + timedelta(seconds=seconds_offset)
        print(current_date.strftime('%Y-%m-%dT%H:%M:%S%z'))

        index = schedule_response['order'][nth_fact]
        fact_id = schedule_request.facts[index].fact_id
        debug_id = schedule_response['debug_id']
        print(fact_id, debug_id)

        # randomly sample user response
        response = bool(np.random.binomial(1, 0.5))

        update_request = UpdateRequestSchema(
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
            fact=schedule_request.facts[index],
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


#### test mode

for nth_fact in range(n_facts_per_day):
    index = schedule_response['order'][nth_fact]
    fact_id = schedule_request.facts[index].fact_id

    # randomly sample user response
    response = bool(np.random.binomial(1, 0.5))

    update_request = UpdateRequestSchema(
        user_id=user_id,
        fact_id=schedule_request.facts[index].fact_id,
        label=response,
        deck_name='dummy',
        deck_id=1000000,
        elapsed_milliseconds_text=10000,
        elapsed_milliseconds_answer=10000,
        debug_id=debug_id,
        history_id=f'sim_history_test_{nth_fact}',
        studyset_id='dummy_studyset_test',
        test_mode=True,
    )
    update_response = json.loads(
        requests.post(
            f'{URL}/update_v2',
            data=json.dumps(update_request.dict()),
        ).text
    )


# see if scheduler correctly raises an exception due to duplicate study record ID
update_request = UpdateRequestSchema(
    user_id=user_id,
    fact_id=schedule_request.facts[index].fact_id,
    label=response,
    deck_name='dummy',
    deck_id=1000000,
    elapsed_milliseconds_text=10000,
    elapsed_milliseconds_answer=10000,
    debug_id=debug_id,
    history_id='295584',
    studyset_id='dummy_studyset_' + debug_id,
    test_mode=False,
    fact=schedule_request.facts[index],
)

update_response = json.loads(
    requests.post(
        f'{URL}/update_v2',
        data=json.dumps(update_request.dict()),
    ).text
)
print(update_response)

print()
print()
for key, values in profile.items():
    print(key, np.mean(values))
