#!/usr/bin/env python
# coding: utf-8

import sys
import json
import copy
import pickle
import requests
import numpy as np
from datetime import datetime, timedelta

from util import parse_date, ScheduleRequest, Card, User, Params


params = Params()
USER_ID = 'test_web_dummy'
# from retention import RetentionModel
# model = RetentionModel()

fret_file = open('sim_fret.txt', 'w')
# fret_file = sys.stdout
detail_file = open('sim_detail.txt', 'w')


def get_result(card: dict, date: datetime):
    request = ScheduleRequest(
        text=card['text'],
        date=str(date),
        answer=card['answer'],
        category=card['category'],
        user_id=USER_ID,
        question_id=card['question_id'],
    )
    r = requests.post('http://127.0.0.1:8000/api/karl/get_card', data=json.dumps(request.__dict__))
    card = Card.unpack(json.loads(r.text))

    r = requests.post('http://127.0.0.1:8000/api/karl/get_user', data=json.dumps({'user_id': USER_ID}))
    user = User.unpack(json.loads(r.text))

    '''
    result = model.predict_one(user, card)
    return 'correct' if result > 0.5 else 'wrong'
    '''

    prob = 0.5  # default
    if card.card_id in user.previous_study:
        prev_date, prev_response = user.previous_study[card.card_id]
        h = user.count_correct_before.get(card.card_id, 0) + user.count_wrong_before.get(card.card_id, 0)
        delta = (date - prev_date).days
        prob = np.exp2(- delta / h)

    result = np.random.binomial(1, prob)
    return 'correct' if result else 'wrong'

def schedule_and_update(cards, date):
    for card in cards:
        card['date'] = str(date)

    r = requests.post('http://127.0.0.1:8000/api/karl/schedule', data=json.dumps(cards))
    schedule_outputs = json.loads(r.text)

    card = cards[schedule_outputs['order'][0]]
    card['label'] = get_result(card, date)

    card['history_id'] = 'dummy_{}_{}'.format(card['question_id'], str(date))
    r = requests.post('http://127.0.0.1:8000/api/karl/update', data=json.dumps([card]))
    update_outputs = json.loads(r.text)

    print(current_date.strftime('%Y-%m-%d-%H-%M'), file=detail_file)
    print('    {: <16} : {}'.format('card_id', card['question_id']), file=detail_file)
    print('    {: <16} : {}'.format('result', card['label']), file=detail_file)
    print('    {: <16} : {}'.format('text', card['text']), file=detail_file)
    print('    {: <16} : {}'.format('answer', card['answer']), file=detail_file)
    print('    {: <16} : {}'.format('category', card['category']), file=detail_file)
    print('    {: <16} : {}'.format('current date', str(date)), file=detail_file)

    print('', file=detail_file)
    print(' ' * 3, 'update details', file=detail_file)
    for key, value in update_outputs.items():
        if isinstance(value, float):
            print(' ' * 7, '{: <16} : {:.4f}'.format(key, value), file=detail_file)
        elif isinstance(value, int) or isinstance(value, str):
            print(' ' * 7, '{: <16} : {}'.format(key, value), file=detail_file)

    print('', file=detail_file)
    for i, card_info in enumerate(schedule_outputs['cards_info']):
        print(' ' * 3, 'card', i, file=detail_file)
        for key, value in card_info.items():
            if isinstance(value, float):
                print(' ' * 7, '{: <16} : {:.4f}'.format(key, value), file=detail_file)
            elif isinstance(value, int) or isinstance(value, str):
                print(' ' * 7, '{: <16} : {}'.format(key, value), file=detail_file)
        print(' ' * 7, 'scores', file=detail_file)
        for k, v in card_info['scores'].items():
            print(' ' * 11,
                  '{: <16} : {:.4f} x {:.2f}'.format(k, v, params.__dict__.get(k, 0)),
                  file=detail_file)
        print('', file=detail_file)

    return schedule_outputs, update_outputs


if __name__ == '__main__':
    N_DAYS = 4
    N_TOTAL_CARDS = 100
    MAX_CARDS = 100
    TURN_AROUND = 0.5 * 60  # seconds
    MAX_REVIEW_WINDOW = 2 * 60 * 60  # seconds

    with open('data/diagnostic_questions.pkl', 'rb') as f:
        diagnostic_cards = pickle.load(f)
    cards = copy.deepcopy(diagnostic_cards[:N_TOTAL_CARDS])
    for i, card in enumerate(cards):
        card['user_id'] = USER_ID

    # requests.post('http://127.0.0.1:8000/api/karl/set_params', data=json.dumps(params))
    requests.post('http://127.0.0.1:8000/api/karl/reset', data=json.dumps({'user_id': USER_ID}))

    card_to_column = dict()
    start_date = parse_date('2028-06-1 08:00:00.000001')
    for days in range(N_DAYS):
        print('day {}'.format(days), file=fret_file)
        time_offset = 0
        n_cards = 0
        while True:
            current_date = start_date + timedelta(days=days) + timedelta(seconds=time_offset)

            # # stop if both True
            # #   1) no reivew scheduled within MAX_REVIEW_WINDOW by Leitner
            # #   2) already studied 10 new cards
            # r = requests.post('http://127.0.0.1:8000/api/karl/get_user', data=json.dumps({'user_id': USER_ID}))
            # user = User.unpack(json.loads(r.text))
            # leitner_scheduled_dates = list(user.leitner_scheduled_date.values())
            # if len(leitner_scheduled_dates) == 0:
            #     delta = 0
            # else:
            #     next_review_date = min(leitner_scheduled_dates)
            #     delta = (next_review_date - current_date).total_seconds()
            # if delta >= MAX_REVIEW_WINDOW:
            #     break
            if n_cards > MAX_CARDS:
                break

            schedule_outputs, update_outputs = schedule_and_update(cards, current_date)
            card_id = cards[schedule_outputs['order'][0]]['question_id']

            if card_id not in card_to_column:
                card_to_column[card_id] = len(card_to_column)

            print('{} {: <6} {: <3} {}{}'.format(
                current_date.strftime('%Y-%m-%d-%H-%M'),
                card_id,
                n_cards,
                ' ' * card_to_column[card_id],
                'o' if update_outputs['response'] == 'correct' else 'x'
            ), file=fret_file)

            time_offset += TURN_AROUND
            n_cards += 1
        print('', file=fret_file)
