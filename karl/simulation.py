#!/usr/bin/env python
# coding: utf-8

import json
import copy
import pickle
import requests
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

from karl.util import ScheduleRequest, SetParams, Fact, User, Params
from karl.retention.baseline import RetentionModel
# from karl.new_retention import HFRetentionModel as RetentionModel

model = RetentionModel()

CORRECT = True
WRONG = False
URL = 'http://127.0.0.1:8000/api/karl'

fret_file = open('output/sim_fret.txt', 'w')
detail_file = open('output/sim_detail.txt', 'w')


def get_result(user_id: str, fact: dict, date: datetime) -> bool:
    '''the core of simulated user that determines the binary outcome '''
    request = ScheduleRequest(
        user_id=user_id,
        env='dev',
        fact_id=fact['fact_id'],
        text=fact['text'],
        deck_name='simulation',
        date=str(date),
        answer=fact['answer'],
        category=fact['category'],
        repetition_model='leitner',
    )
    # retrieve fact
    r = requests.post(f'{URL}/get_fact', data=json.dumps(request.__dict__))
    fact = Fact.unpack(json.loads(r.text))

    # retrieve user
    r = requests.get(f'{URL}/get_user?user_id={user_id}&env=dev')
    user = User.unpack(json.loads(r.text))

    if False:
        # use retention model to predict a binary outcome
        result = model.predict_one(user, fact)
        return CORRECT if result > 0.5 else WRONG
    else:
        # use heuristic to predict a binary outcome
        prob = 0.5  # default
        if fact.fact_id in user.previous_study:
            prev_date, prev_response = user.previous_study[fact.fact_id]
            h = (
                user.count_correct_before.get(fact.fact_id, 0)
                + user.count_wrong_before.get(fact.fact_id, 0)
            )
            delta = (date - prev_date).days
            prob = np.exp2(- delta / h)

        result = np.random.binomial(1, prob)
        return CORRECT if result else WRONG


def schedule_and_update(user_id: str, facts: List[dict], date: datetime, params: Params) -> Tuple[dict, dict]:
    '''
    schedule facts
    use simulated user to obtain binary outcome of the top card
    update scheduler with binary outcome
    return details from both schedule and update
    '''
    # construct schedule requests
    for fact in facts:
        fact.update({
            'date': str(date),
            'repetition_model': 'leitner',
            'deck_name': 'simulation',
            'env': 'dev',
        })

    # schedule, get the ordering of facts
    r = requests.post(f'{URL}/schedule', data=json.dumps(facts))
    schedule_outputs = json.loads(r.text)

    # get the top card, get a binary outcome from the simulated user
    fact = facts[schedule_outputs['order'][0]]
    fact['label'] = get_result(user_id, fact, date)

    # update scheduler with binary outcome
    fact['history_id'] = f'{user_id}_{fact["fact_id"]}_{str(date)}'
    fact['elapsed_seconds_text'] = 2
    fact['elapsed_seconds_answer'] = 2
    r = requests.post(f'{URL}/update', data=json.dumps([fact]))
    update_outputs = json.loads(r.text)

    # record scheduling details
    print(date.strftime('%Y-%m-%d-%H-%M'), file=detail_file)
    print('    {: <16} : {}'.format('fact_id', fact['fact_id']), file=detail_file)
    print('    {: <16} : {}'.format('result', fact['label']), file=detail_file)
    print('    {: <16} : {}'.format('text', fact['text']), file=detail_file)
    print('    {: <16} : {}'.format('answer', fact['answer']), file=detail_file)
    print('    {: <16} : {}'.format('category', fact['category']), file=detail_file)
    print('    {: <16} : {}'.format('current date', str(date)), file=detail_file)

    print('', file=detail_file)
    print(' ' * 3, 'update details', file=detail_file)
    for key, value in update_outputs.items():
        if isinstance(value, float):
            print(' ' * 7, '{: <16} : {:.4f}'.format(key, value), file=detail_file)
        elif isinstance(value, int) or isinstance(value, str):
            print(' ' * 7, '{: <16} : {}'.format(key, value), file=detail_file)

    print('', file=detail_file)
    for i, fact_info in enumerate(schedule_outputs['facts_info']):
        print(' ' * 3, 'fact', i, file=detail_file)
        for key, value in fact_info.items():
            if isinstance(value, float):
                print(' ' * 7, '{: <16} : {:.4f}'.format(key, value), file=detail_file)
            elif isinstance(value, int) or isinstance(value, str):
                print(' ' * 7, '{: <16} : {}'.format(key, value), file=detail_file)
        print(' ' * 7, 'scores', file=detail_file)
        for k, v in fact_info['scores'].items():
            print(' ' * 11,
                  '{: <16} : {:.4f} x {:.2f}'.format(k, v, params.__dict__.get(k, 0)),
                  file=detail_file)
        print('', file=detail_file)

    return schedule_outputs, update_outputs


def test_scheduling(
        user_id: str = 'dummy',
        n_days: int = 10,
        n_total_facts: int = 100,
        max_facts_per_day: int = 100,
):

    params = SetParams(
        user_id=user_id,
        env='dev',
        qrep=0,
        skill=0,
        recall=1,  # NOTE make sure this is turned on so the scheduler is consistent with simulation
        category=0,
        leitner=0,
        sm2=0,
        decay_qrep=0.9,
        decay_skill=0.9,
        cool_down=0,
        cool_down_time_correct=20,
        cool_down_time_wrong=4,
        max_recent_facts=10,
    )

    TURN_AROUND = 0.5 * 60  # how long it takes to study a fact, in seconds
    # MAX_REVIEW_WINDOW = 2 * 60 * 60  # seconds

    with open('data/diagnostic_questions.pkl', 'rb') as f:
        diagnostic_facts = pickle.load(f)
    facts = copy.deepcopy(diagnostic_facts[:n_total_facts])

    for i, fact in enumerate(facts):
        fact['user_id'] = user_id

    # prepare scheduler
    requests.post(f'{URL}/set_params', data=json.dumps(params.__dict__))
    requests.get(f'{URL}/reset_user?user_id={user_id}&env=dev')

    fact_to_column = dict()  # fact -> ith column in the fret plot
    start_date = parse_date('2028-06-01 08:00:00.000001 -0400')
    profile = defaultdict(list)  # performance profiling

    for days in range(n_days):
        print(f'day {days}', file=fret_file)
        print(f'day {days}')
        time_offset = 0  # offset from the the first fact of today, in seconds
        for ith_fact in tqdm(range(max_facts_per_day)):
            current_date = start_date + timedelta(days=days) + timedelta(seconds=time_offset)

            # NOTE exhaust MAX_FACTS_PER_DAY
            # # stop if both True
            # #   1) no reivew scheduled within MAX_REVIEW_WINDOW by Leitner
            # #   2) already studied 10 new facts
            # r = requests.get(f'{URL}/get_user?user_id={usdr_id}&env=dev')
            # user = User.unpack(json.loads(r.text))
            # leitner_scheduled_dates = list(user.leitner_scheduled_date.values())
            # if len(leitner_scheduled_dates) == 0:
            #     delta = 0
            # else:
            #     next_review_date = min(leitner_scheduled_dates)
            #     delta = (next_review_date - current_date).total_seconds()
            # if delta >= MAX_REVIEW_WINDOW:
            #     break

            # schedule, study, update
            schedule_outputs, update_outputs = schedule_and_update(user_id, facts, current_date, params)

            # find the proper column in the fret file for the scheduled fact
            # new column if new fact, otherwise look it up
            fact_id = facts[schedule_outputs['order'][0]]['fact_id']
            if fact_id not in fact_to_column:
                fact_to_column[fact_id] = len(fact_to_column)

            # update fret plot
            print('{} {: <6} {: <3} {}{}'.format(
                current_date.strftime('%Y-%m-%d-%H-%M'),
                fact_id,
                ith_fact,
                ' ' * fact_to_column[fact_id],
                'o' if update_outputs['response'] == CORRECT else 'x'
            ), file=fret_file)

            # update performance profile
            for key, value in schedule_outputs['profile'].items():
                profile[key].append(value)

            # move current offset forward
            time_offset += TURN_AROUND

        print('', file=fret_file)

    # summarize performance profile
    print('===== performance profile =====')
    for key, value in profile.items():
        count, time = list(zip(*value))
        print(f'{key} {np.mean(time)} per call with on average {np.mean(count)} facts')


if __name__ == '__main__':
    test_scheduling(user_id='dummy_1', n_days=3, n_total_facts=30, max_facts_per_day=30)
    test_scheduling(user_id='dummy_2', n_days=6, n_total_facts=30, max_facts_per_day=40)
    test_scheduling(user_id='dummy_3', n_days=4, n_total_facts=50, max_facts_per_day=20)
    # test_for_leaderboard()
