#!/usr/bin/env python
# coding: utf-8

import json
import copy
import pickle
import requests
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from pprint import pprint
from collections import defaultdict
from datetime import datetime, timedelta

from karl.util import parse_date, ScheduleRequest, Fact, User, Params
# from karl.retention.baseline import RetentionModel
from karl.new_retention import HFRetentionModel as RetentionModel

model = RetentionModel()

CORRECT = True
WRONG = False
USER_ID = 'test_web_dummy'

fret_file = open('output/sim_fret.txt', 'w')
detail_file = open('output/sim_detail.txt', 'w')

params = Params(
    user_id=USER_ID,
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


def get_result(fact: dict, date: datetime) -> bool:
    '''the core of simulated user that determines the binary outcome '''
    request = ScheduleRequest(
        text=fact['text'],
        date=str(date),
        answer=fact['answer'],
        category=fact['category'],
        user_id=USER_ID,
        fact_id=fact['fact_id'],
        repetition_model='leitner',
        deck_name='simulation',
        env='simulation',
    )
    # retrieve fact
    r = requests.post('http://127.0.0.1:8000/api/karl/get_fact', data=json.dumps(request.__dict__))
    fact = Fact.unpack(json.loads(r.text))

    # retrieve user
    r = requests.get('http://127.0.0.1:8000/api/karl/get_user/{}'.format(USER_ID))
    user = User.unpack(json.loads(r.text))

    if True:
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


def schedule_and_update(facts: List[dict], date: datetime) -> Tuple[dict, dict]:
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
            'env': 'simulation',
        })

    # schedule, get the ordering of facts
    r = requests.post('http://127.0.0.1:8000/api/karl/schedule', data=json.dumps(facts))
    schedule_outputs = json.loads(r.text)

    # get the top card, get a binary outcome from the simulated user
    fact = facts[schedule_outputs['order'][0]]
    fact['label'] = get_result(fact, date)

    # update scheduler with binary outcome
    fact['history_id'] = 'dummy_{}_{}'.format(fact['fact_id'], str(date))
    r = requests.post('http://127.0.0.1:8000/api/karl/update', data=json.dumps([fact]))
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


def test_scheduling():
    N_DAYS = 10
    N_TOTAL_FACTS = 100
    MAX_FACTS_PER_DAY = 100
    TURN_AROUND = 0.5 * 60  # how long it takes to study a fact, in seconds
    MAX_REVIEW_WINDOW = 2 * 60 * 60  # seconds

    with open('data/diagnostic_questions.pkl', 'rb') as f:
        diagnostic_facts = pickle.load(f)
    facts = copy.deepcopy(diagnostic_facts[:N_TOTAL_FACTS])

    for i, fact in enumerate(facts):
        fact['user_id'] = USER_ID

    # prepare scheduler
    requests.post('http://127.0.0.1:8000/api/karl/set_params', data=json.dumps(params.__dict__))
    requests.get('http://127.0.0.1:8000/api/karl/reset_user/{}'.format(USER_ID))

    fact_to_column = dict()  # fact -> ith column in the fret plot
    start_date = parse_date('2028-06-1 08:00:00.000001')
    profile = defaultdict(list)  # performance profiling

    for days in range(N_DAYS):
        print('day {}'.format(days), file=fret_file)
        print('day {}'.format(days))
        time_offset = 0  # offset from the the first fact of today, in seconds
        for ith_fact in tqdm(range(MAX_FACTS_PER_DAY)):
            current_date = start_date + timedelta(days=days) + timedelta(seconds=time_offset)

            # NOTE exhaust MAX_FACTS_PER_DAY
            # # stop if both True
            # #   1) no reivew scheduled within MAX_REVIEW_WINDOW by Leitner
            # #   2) already studied 10 new facts
            # r = requests.get('http://127.0.0.1:8000/api/karl/get_user/{}'.format(USER_ID))
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
            schedule_outputs, update_outputs = schedule_and_update(facts, current_date)

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
    for key, value in profile.items():
        count, time = list(zip(*value))
        print(key, np.mean(count), np.mean(time))


def test_for_leaderboard():
    # TODO this should go to unit tests
    # r = requests.get('http://127.0.0.1:8000/api/karl/get_user_stats/{}'.format(USER_ID))
    # print(json.loads(r.text))
    # print()

    r = requests.get('http://127.0.0.1:8000/api/karl/get_all_users')
    users = [User.unpack(s) for s in json.loads(r.text)]
    if len(users) == 0:
        return

    stats = users[0].user_stats.__dict__
    pprint({
        'new_facts': stats['new_facts'],
        'reviewed_facts': stats['reviewed_facts'],
        'total_seen': stats['total_seen'],
        'total_seconds': stats['total_seconds'],
        'last_week_seen': stats['last_week_seen'],
        'last_week_new_facts': stats['last_week_new_facts'],
        'new_known_rate': stats['new_known_rate'],
        'review_known_rate': stats['review_known_rate'],
    })


if __name__ == '__main__':
    test_scheduling()
    # test_for_leaderboard()
