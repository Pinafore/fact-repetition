#!/usr/bin/env python
# coding: utf-8

import json
import copy
import pickle
import requests
import textwrap
import logging
from datetime import datetime, timedelta

from util import parse_date, ScheduleRequest, Card, User
from retention import RetentionModel

logging.basicConfig(filename='test_web.log', filemode='w', level=logging.INFO)

params = {
    'n_topics': 10,
    'qrep': 3.,
    'skill': 0,
    'time': 1.,
    'category': 3.,
    'leitner': 1.,
    'cool_down_time': 20.,
    'sm2': 0,
    'decay_qrep': 0.9,
}

with open('data/diagnostic_questions.pkl', 'rb') as f:
    diagnostic_cards = pickle.load(f)

USER_ID = 'test_web_dummy'


model = RetentionModel()

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

    result = 'correct' if model.predict(user, card) else 'wrong'
    return result

    # # initial probablity=0.5
    # if card.card_id not in user.last_study_date:
    #     prob = 0.5

    # # after each study, no matter the result, probability=1
    # prob = 1

    # # exponential forgetting curve parameterized by number of repetition
    # alpha = user.sm2_repetition[

    # # if repeated within 30 minutes, return true
    # if (date - user.last_study_date).seconds
    # return result


def schedule_and_update(cards, date):
    cards[0]['date'] = str(date)
    for card in cards:
        card['date'] = str(date)

    r = requests.post('http://127.0.0.1:8000/api/karl/schedule', data=json.dumps(cards))
    schedule_outputs = json.loads(r.text)
    order = schedule_outputs['order']
    schedule_outputs['detail']

    card = cards[order[0]]
    card['label'] = get_result(card, date)

    card['history_id'] = 'dummy_{}_{}'.format(card['question_id'], str(date))
    r = requests.post('http://127.0.0.1:8000/api/karl/update', data=json.dumps([card]))
    update_outputs = json.loads(r.text)

    print(current_date.strftime('%Y-%m-%d-%H-%M'))
    print('    {: <16} : {}'.format('card_id', card['question_id']))
    print('    {: <16} : {}'.format('result', card['label']))
    print('    {: <16} : {}'.format('text', textwrap.fill(
        card['text'], width=50, subsequent_indent=' ' * 23)))
    print('    {: <16} : {}'.format('answer', card['answer']))
    print('    {: <16} : {}'.format('category', card['category']))

    for key, value in schedule_outputs['detail'].items():
        if key in params:
            print('    {: <16} : {:.4f} x {:.2f}'.format(key, value, params[key]))
        elif isinstance(value, float):
            print('    {: <16} : {:.4f}'.format(key, value))
        elif isinstance(value, int) or isinstance(value, str):
            print('    {: <16} : {}'.format(key, value))

    for key, value in update_outputs.items():
        if key in params:
            print('    {: <16} : {:.4f} x {:.2f}'.format(key, value, params[key]))
        elif isinstance(value, float):
            print('    {: <16} : {:.4f}'.format(key, value))
        elif isinstance(value, int) or isinstance(value, str):
            print('    {: <16} : {}'.format(key, value))

    return schedule_outputs, update_outputs


if __name__ == '__main__':
    cards = copy.deepcopy(diagnostic_cards[:30])
    for i, card in enumerate(cards):
        card['user_id'] = USER_ID

    requests.post('http://127.0.0.1:8000/api/karl/set_params', data=json.dumps(params))
    requests.post('http://127.0.0.1:8000/api/karl/reset', data=json.dumps({'user_id': USER_ID}))

    start_date = parse_date('2028-06-1 18:27:08.172341')
    card_to_column = dict()
    for days in range(4):
        logging.info('day {}'.format(days))
        for minutes in range(30):
            current_date = start_date + timedelta(days=days) + timedelta(minutes=minutes * 3)
            schedule_outputs, update_outputs = schedule_and_update(cards, current_date)
            order = schedule_outputs['order']
            response = update_outputs['response']
            card_id = cards[order[0]]['question_id']
            if card_id not in card_to_column:
                card_to_column[card_id] = len(card_to_column)
            logging.info('{} {: <6} {}{}'.format(
                current_date.strftime('%Y-%m-%d-%H-%M'),
                card_id,
                ' ' * card_to_column[card_id],
                'o' if response == 'correct' else 'x'
            ))
        logging.info('')
    # requests.post('http://127.0.0.1:8000/api/karl/set_params', data=json.dumps(params))
