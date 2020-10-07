# %%
import os
import json
import copy
import pickle
import unittest
import requests
import numpy as np
from pprint import pprint
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

from karl.db import SchedulerDB
from karl.util import User, Fact, History, Params, ScheduleRequest
from karl.scheduler import MovingAvgScheduler

# %%
URL = 'http://127.0.0.1:8000/api/karl'

env = 'dev'
user_id = '2'

# get the simulated user ready
user = json.loads(json.loads(requests.get(f'{URL}/get_user?user_id={user_id}&env={env}').text))


# %%
class TestWeb(unittest.TestCase):

    def test_user_stats(self):
        URL = 'http://127.0.0.1:8000/api/karl'

        env = 'prod'
        user_id = 'unit_test_user'
        deck_id = 'unit_test_deck'

        # get the simulated user ready
        requests.get(f'{URL}/reset_user?user_id={user_id}&env={env}')
        # requests.put(
        #     f'{URL}/set_params?user_id={user_id}&env={env}',
        #     data=json.dumps(Params().__dict__),
        # )

        requests.get(f'{URL}/set_repetition_model?user_id={user_id}&env={env}&repetition_model=karl76')
        return

        # get some facts ready
        with open('data/diagnostic_questions.pkl', 'rb') as f:
            diagnostic_facts = pickle.load(f)
        facts = copy.deepcopy(diagnostic_facts[:10])
        for fact in facts:
            fact.update({
                'env': env,
                'deck_id': deck_id,
                'user_id': user_id,
            })

        date_start = '2028-06-03T03:41:14.779779-0400'
        for day in range(10):
            for i in range(10):
                date = parse_date(date_start) + timedelta(days=day, seconds=i)

                fact = facts[i]
                fact['label'] = True
                fact['date'] = date.strftime('%Y-%m-%dT%H:%M:%S%z')

                # update scheduler with binary outcome
                fact['history_id'] = f'{user_id}_{fact["fact_id"]}_{fact["date"]}'
                fact['elapsed_milliseconds_text'] = 2000
                fact['elapsed_milliseconds_answer'] = 2000
                requests.post(f'{URL}/update', data=json.dumps([fact]))

        date_end = parse_date(date_start) + timedelta(days=12)
        date_end = date_end.strftime('%Y-%m-%dT%H:%M:%S%z')
        req = f'{URL}/get_user_stats?user_id={user_id}&env={env}&date_start={date_start}&date_end={date_end}'
        leaderboard = json.loads(requests.get(req).text)
        pprint(leaderboard)


if __name__ == '__main__':
    unittest.main()
