import json
import time
import pickle
import random
import asyncio
import requests
import numpy as np
from typing import List
from datetime import datetime
from dateutil.parser import parse as parse_date
from concurrent.futures import ThreadPoolExecutor

from karl.config import settings
from karl.schemas import ScheduleRequestSchema, UpdateRequestSchema, ScheduleResponseSchema

URL = 'http://127.0.0.1:8000/api/karl'


with open(f'{settings.DATA_DIR}/diagnostic_questions.pkl', 'rb') as f:
    diagnostic_facts = pickle.load(f)


def schedule(
    user_id: str,
    date: datetime,
):
    print('schedule starts at'.ljust(20), datetime.now().strftime('%H:%M:%S.%f'))

    schedule_requests = [
        ScheduleRequestSchema(
            text=fact['text'],
            date=date.strftime('%Y-%m-%dT%H:%M:%S%z'),
            answer=fact['answer'],
            category=fact['category'],
            user_id=user_id,
            fact_id='sim_' + str(fact['fact_id']),
        )
        for fact in random.sample(diagnostic_facts, 300)
    ]
    schedule_response = json.loads(
        requests.post(
            f'{URL}/schedule?env=prod',
            data=json.dumps([r.__dict__ for r in schedule_requests])
        ).text
    )

    print('schedule finishes at'.ljust(20), datetime.now().strftime('%H:%M:%S.%f'))
    return schedule_requests, schedule_response

def update(
    schedule_requests: List[ScheduleRequestSchema],
    schedule_response: ScheduleResponseSchema,
    date: datetime,
):
    print('update starts at'.ljust(20), datetime.now().strftime('%H:%M:%S.%f'))

    index = schedule_response['order'][0]
    debug_id = schedule_response['debug_id']
    user_id = schedule_requests[0].user_id
    response = bool(np.random.binomial(1, 0.5))

    update_request = UpdateRequestSchema(
        text=schedule_requests[index].text,
        date=date.strftime('%Y-%m-%dT%H:%M:%S%z'),
        answer=schedule_requests[index].answer,
        category=schedule_requests[index].category,
        user_id=user_id,
        fact_id=schedule_requests[index].fact_id,
        label=response,
        history_id=f'sim_history_{1}_{1}',
        elapsed_milliseconds_text=10000,
        elapsed_milliseconds_answer=10000,
        debug_id=debug_id,
    )
    requests.post(
        f'{URL}/update?env=prod',
        data=json.dumps([update_request.__dict__]),
    )

    print('update finishes at'.ljust(20), datetime.now().strftime('%H:%M:%S.%f'))

def user_stats(user_id: str):
    print('stats starts at'.ljust(20), datetime.now().strftime('%H:%M:%S %f'))
    requests.get(f'{URL}/get_user_stats?env=prod&user_id={user_id}')
    print('stats finishes at'.ljust(20), datetime.now().strftime('%H:%M:%S %f'))

def leaderboard(user_id: str):
    print('leaderboard starts at'.ljust(20), datetime.now().strftime('%H:%M:%S %f'))
    requests.get(f'{URL}/leaderboard?rank_type=total_seen&limit=10&min_studied=10&env=43&user_id={user_id}')
    print('leaderboard finishes at'.ljust(20), datetime.now().strftime('%H:%M:%S %f'))

def wait_and_print(wait: int, what: str):
    requests.get(f'{URL}/wait_and_print?wait={wait}&what={what}')

async def test0():
    await asyncio.gather(
        asyncio.to_thread(wait_and_print, 5, 'first'),
        asyncio.to_thread(wait_and_print, 5, 'second'),
    )

async def test1():
    user_id = 'dummy'

    await asyncio.gather(
        asyncio.to_thread(user_stats, user_id),
        asyncio.to_thread(leaderboard, user_id),
        asyncio.to_thread(user_stats, user_id),
    )

async def test2():
    user_id = 'dummy'
    date = parse_date('2028-06-01 08:00:00.000001 -0400')

    with ThreadPoolExecutor() as executor:
        future = executor.submit(schedule, user_id, date)
        executor.submit(user_stats, user_id)
        executor.submit(leaderboard, user_id)
    schedule_requests, schedule_response = future.result()

    time.sleep(2)

    await asyncio.gather(
        asyncio.to_thread(update, schedule_requests, schedule_response, date),
        asyncio.to_thread(user_stats, user_id),
        asyncio.to_thread(user_stats, user_id),
        asyncio.to_thread(leaderboard, user_id),
    )

if __name__ == '__main__':
    # asyncio.run(test0())
    # asyncio.run(test1())
    asyncio.run(test2())
