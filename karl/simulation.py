#!/usr/bin/env python
# coding: utf-8

import json
import copy
import pickle
import requests
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta

from karl.util import parse_date, ScheduleRequest, Fact, User, Params


params = Params()
USER_ID = 'test_web_dummy'
# from retention import RetentionModel
# model = RetentionModel()

fret_file = open('sim_fret.txt', 'w')
detail_file = open('sim_detail.txt', 'w')


def get_result(fact: dict, date: datetime):
    request = ScheduleRequest(
        text=fact['text'],
        date=str(date),
        answer=fact['answer'],
        category=fact['category'],
        user_id=USER_ID,
        fact_id=fact['fact_id'],
    )
    r = requests.post('http://127.0.0.1:8000/api/karl/get_fact',
                      data=json.dumps(request.__dict__))
    fact = Fact.unpack(json.loads(r.text))

    r = requests.get('http://127.0.0.1:8000/api/karl/get_user/{}'.format(USER_ID))
    user = User.unpack(json.loads(r.text))

    '''
    result = model.predict_one(user, fact)
    return 'correct' if result > 0.5 else 'wrong'
    '''

    prob = 0.5  # default
    if fact.fact_id in user.previous_study:
        prev_date, prev_response = user.previous_study[fact.fact_id]
        h = user.count_correct_before.get(fact.fact_id, 0) + user.count_wrong_before.get(fact.fact_id, 0)
        delta = (date - prev_date).days
        prob = np.exp2(- delta / h)

    result = np.random.binomial(1, prob)
    return 'correct' if result else 'wrong'

def schedule_and_update(facts, date):
    for fact in facts:
        fact['date'] = str(date)

    r = requests.post('http://127.0.0.1:8000/api/karl/schedule', data=json.dumps(facts))
    schedule_outputs = json.loads(r.text)

    fact = facts[schedule_outputs['order'][0]]
    fact['label'] = get_result(fact, date)

    fact['history_id'] = 'dummy_{}_{}'.format(fact['fact_id'], str(date))
    r = requests.post('http://127.0.0.1:8000/api/karl/update', data=json.dumps([fact]))
    update_outputs = json.loads(r.text)

    print(current_date.strftime('%Y-%m-%d-%H-%M'), file=detail_file)
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

    return schedule_outputs, update_outputs,


if __name__ == '__main__':
    N_DAYS = 1
    N_TOTAL_FACTS = 100
    MAX_FACTS_PER_DAY = 100
    TURN_AROUND = 0.5 * 60  # seconds
    MAX_REVIEW_WINDOW = 2 * 60 * 60  # seconds

    with open('data/diagnostic_questions.pkl', 'rb') as f:
        diagnostic_facts = pickle.load(f)
    facts = copy.deepcopy(diagnostic_facts[:N_TOTAL_FACTS])
    for i, fact in enumerate(facts):
        fact['user_id'] = USER_ID

    # requests.post('http://127.0.0.1:8000/api/karl/set_params', data=json.dumps(params))
    requests.get('http://127.0.0.1:8000/api/karl/reset_user/{}'.format(USER_ID))

    fact_to_column = dict()
    start_date = parse_date('2028-06-1 08:00:00.000001')
    profile = defaultdict(list)
    for days in range(N_DAYS):
        print('day {}'.format(days), file=fret_file)
        print('day {}'.format(days))
        time_offset = 0
        for ith_fact in tqdm(range(MAX_FACTS_PER_DAY)):
            current_date = start_date + timedelta(days=days) + timedelta(seconds=time_offset)

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

            schedule_outputs, update_outputs = schedule_and_update(facts, current_date)
            fact_id = facts[schedule_outputs['order'][0]]['fact_id']
            for key, value in schedule_outputs['profile'].items():
                profile[key].append(value)

            if fact_id not in fact_to_column:
                fact_to_column[fact_id] = len(fact_to_column)

            print('{} {: <6} {: <3} {}{}'.format(
                current_date.strftime('%Y-%m-%d-%H-%M'),
                fact_id,
                ith_fact,
                ' ' * fact_to_column[fact_id],
                'o' if update_outputs['response'] == 'correct' else 'x'
            ), file=fret_file)

            time_offset += TURN_AROUND
        print('', file=fret_file)

    for key, value in profile.items():
        count, time = list(zip(*value))
        print(key, np.mean(count), np.mean(time))

    r = requests.get('http://127.0.0.1:8000/api/karl/get_user_stats/{}'.format(USER_ID))
    print(json.loads(r.text))
