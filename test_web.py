#!/usr/bin/env python
# coding: utf-8

import json
import copy
import pickle
import requests
import textwrap
import logging
import numpy as np
from datetime import timedelta
from collections import defaultdict
from util import parse_date

logging.basicConfig(filename='test_web.log', filemode='w', level=logging.INFO)

params = {
    'n_topics': 10,
    'qrep': 1.0,
    'skill': 1.0,
    'time': 1.0,
    'category': 1.0,
    'leitner': 1.0,
    'sm2': 1.0,
    'decay_qrep': 0.9,
}

with open('data/jeopardy_358974_questions_20190612.pkl', 'rb') as f:
    karl_to_question_id = pickle.load(f).to_dict()['question_id']
with open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb') as f:
    records_df = pickle.load(f)
with open('data/diagnostic_questions.pkl', 'rb') as f:
    diagnostic_cards = pickle.load(f)

estimate = {}

def get_result(card):
    if card['question_id'] not in estimate:
        record_id = karl_to_question_id[int(card['question_id'])]
        prob = (records_df[records_df.question_id == record_id]['correct'] / 2 + 0.5).mean()
        estimate[card['question_id']] = prob
    coin_flip = np.random.binomial(1, prob)
    return 'correct' if coin_flip == 1 else 'wrong'

def schedule_and_update(cards, date):
    cards[0]['date'] = str(date)
    r = requests.post('http://127.0.0.1:8000/api/karl/schedule',
                      data=json.dumps(cards))
    schedule_outputs = json.loads(r.text)
    order = schedule_outputs['order']
    detail_schedule = schedule_outputs['detail']

    card_selected = cards[order[0]]
    card_id = card_selected['question_id']
    if 'label' not in card_selected:
        card_selected['label'] = get_result(card_selected)

    card_selected['date'] = str(date)
    card_selected['history_id'] = 'dummy_{}_{}'.format(card_id, str(date))
    r = requests.post('http://127.0.0.1:8000/api/karl/update',
                      data=json.dumps([card_selected]))
    update_outputs = json.loads(r.text)
    detail_update = update_outputs['detail']

    logging.info('{: <8} : {}'.format('card_id', card_id))
    logging.info('{: <8} : {}'.format('result', card_selected['label']))
    logging.info('{: <8} : {}'.format('text', textwrap.fill(
        card_selected['text'], width=50, subsequent_indent=' ' * 11)))
    logging.info('{: <8} : {}'.format('answer', card_selected['answer']))
    logging.info('{: <8} : {}'.format('category', card_selected['category']))

    for key, value in detail_schedule.items():
        if key in params:
            logging.info('{: <8} : {:.4f} x {:.2f}'.format(key, value, params[key]))
        elif isinstance(value, float):
            logging.info('{: <8} : {:.4f}'.format(key, value))
        else:
            logging.info('{: <8} : {}'.format(key, value))

    logging.info('---')

    for key, value in detail_update.items():
        if key in params:
            logging.info('{: <8} : {:.4f} x {:.2f}'.format(key, value, params[key]))
        elif isinstance(value, float):
            logging.info('{: <8} : {:.4f}'.format(key, value))
        else:
            logging.info('{: <8} : {}'.format(key, value))
    return schedule_outputs, update_outputs

def repeat(cards):
    card = cards[0]
    card['user_id'] = 'dummy'
    start_date = parse_date('2028-06-1 18:27:08.172341')
    for days in range(10):
        current_date = start_date + timedelta(days=1)
        schedule_outputs, update_outputs = schedule_and_update([card], current_date)
        logging.info(str(days) + ' ----------------------')

def regular(cards):
    for i, card in enumerate(cards):
        cards[i]['user_id'] = 'dummy'

    # studies 30 cards everyday
    # keep the pool constant
    start_date = parse_date('2028-06-1 18:27:08.172341')
    for days in range(30):
        for minutes in range(30):
            current_date = start_date + timedelta(days=days) + timedelta(minutes=minutes * 3)
            schedule_outputs, update_outputs = schedule_and_update(cards, current_date)
            logging.info('\n\n')
        logging.info('\n\n')
        logging.info(str(days) + ' ----------------------')
        logging.info('\n\n')

def restore_params():
    params = {
        'n_topics': 10,
        'qrep': 1.0,
        'skill': 0,
        'time': 1.0,
        'category': 1.0,
        'leitner': 1.0,
        'sm2': 0,
        'decay_qrep': 0.9,
    }
    requests.post('http://127.0.0.1:8000/api/karl/set_params', data=json.dumps(params))


if __name__ == '__main__':
    cards = copy.deepcopy(diagnostic_cards[:30])
    for i, card in enumerate(cards):
        card['user_id'] = 'dummy'

    params = {
        'n_topics': 10,
        'qrep': 1.0,
        'skill': 0,
        'time': 0.0,
        'category': 0,
        'leitner': 1.0,
        'sm2': 0,
        'decay_qrep': 0.9,
    }

    requests.post('http://127.0.0.1:8000/api/karl/set_params', data=json.dumps(params))
    requests.post('http://127.0.0.1:8000/api/karl/reset', data=json.dumps({'user_id': 'dummy'}))

    start_date = parse_date('2028-06-1 18:27:08.172341')
    card_to_column = defaultdict(lambda: len(card_to_column))
    for days in range(10):
        print('day', days)
        for minutes in range(30):
            current_date = start_date + timedelta(days=days) + timedelta(minutes=minutes * 3)
            schedule_outputs, update_outputs = schedule_and_update(cards, current_date)
            order = schedule_outputs['order']
            response = update_outputs['detail']['response']
            column_number = card_to_column[cards[order[0]]['question_id']]
            print('{} {}{}{}'.format(
                current_date.strftime('%Y-%m-%d %H:%M'),
                ' ' * column_number,
                'o' if response == 'correct' else 'x',
                ' ' * (len(cards) - column_number)
            ))
        print()

    restore_params()
