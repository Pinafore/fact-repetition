#!/usr/bin/env python
# coding: utf-8

import json
import pickle
import requests
import textwrap
import logging
from datetime import timedelta

from util import parse_date

logging.basicConfig(filename='text_web.log', filemode='w', level=logging.DEBUG)

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

def get_result(card):
    record_id = karl_to_question_id[int(card['question_id'])]
    prob = (records_df[records_df.question_id == record_id]['correct'] / 2 + 0.5).mean()
    return 'correct' if prob > 0.5 else 'wrong'

def schedule_and_update(cards):
    r = requests.post('http://127.0.0.1:8000/api/karl/schedule',
                      data=json.dumps(cards))
    schedule_outputs = json.loads(r.text)
    order = schedule_outputs['order']
    detail_schedule = schedule_outputs['detail']

    card_selected = cards[order[0]]
    card_id = card_selected['question_id']
    if 'label' not in card_selected:
        card_selected['label'] = get_result(card_selected)

    r = requests.post('http://127.0.0.1:8000/api/karl/update',
                      data=json.dumps([card_selected]))
    update_outputs = json.loads(r.text)
    detail_update = update_outputs['detail']

    logging.debug('{: <8} : {}'.format('card_id', card_id))
    logging.debug('{: <8} : {}'.format('result', card_selected['label']))
    logging.debug('{: <8} : {}'.format('text', textwrap.fill(
        card_selected['text'], width=50, subsequent_indent=' ' * 11)))
    logging.debug('{: <8} : {}'.format('answer', card_selected['answer']))
    logging.debug('{: <8} : {}'.format('category', card_selected['category']))

    for key, value in detail_schedule.items():
        if key in params:
            logging.debug('{: <8} : {:.4f} x {:.2f}'.format(key, value, params[key]))
        elif isinstance(value, float):
            logging.debug('{: <8} : {:.4f}'.format(key, value))
        else:
            logging.debug('{: <8} : {}'.format(key, value))

    logging.debug('---')

    for key, value in detail_update.items():
        if key in params:
            logging.debug('{: <8} : {:.4f} x {:.2f}'.format(key, value, params[key]))
        elif isinstance(value, float):
            logging.debug('{: <8} : {:.4f}'.format(key, value))
        else:
            logging.debug('{: <8} : {}'.format(key, value))
    return schedule_outputs, update_outputs

def repeat(cards):
    card = cards[0]
    card['user_id'] = 'dummy'
    current_date = parse_date('2028-06-1 18:27:08.172341')
    for i in range(10):
        card['date'] = str(current_date)
        schedule_outputs, update_outputs = schedule_and_update([card])
        current_date += timedelta(days=1)
        logging.debug(str(i) + ' ----------------------')

def regular(cards):
    for i, card in enumerate(cards):
        cards[i]['user_id'] = 'dummy'

    # studies 30 cards everyday
    # keep the pool constant
    start_date = parse_date('2028-06-1 18:27:08.172341')
    for days in range(30):
        current_date = start_date + timedelta(days=days)
        for i, card in enumerate(cards):
            cards[i]['date'] = str(current_date)
        schedule_outputs, update_outputs = schedule_and_update(cards)
        logging.debug(str(i) + ' ----------------------')


if __name__ == '__main__':
    requests.post('http://127.0.0.1:8000/api/karl/set_params',
                  data=json.dumps(params))
    requests.post('http://127.0.0.1:8000/api/karl/reset',
                  data=json.dumps({'user_id': 'dummy'}))
    regular(diagnostic_cards)
