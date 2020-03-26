#!/usr/bin/env python
# coding: utf-8

import json
import pickle
import requests
import numpy as np
import textwrap
from datetime import datetime

USER_ID = 'dummy'


with open('data/jeopardy_358974_questions_20190612.pkl', 'rb') as f:
    karl_to_question_id = pickle.load(f).to_dict()['question_id']
with open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb') as f:
    records_df = pickle.load(f)
with open('data/diagnostic_questions.pkl', 'rb') as f:
    cards = pickle.load(f)
    for i, card in enumerate(cards):
        cards[i]['user_id'] = USER_ID
        cards[i]['date'] = str(datetime.now())


def get_result(card):
    record_id = karl_to_question_id[int(card['question_id'])]
    prob = (records_df[records_df.question_id == record_id]['correct'] / 2 + 0.5).mean()
    return 'correct' if prob > 0.5 else 'wrong'


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


requests.post('http://127.0.0.1:8000/api/karl/set_params',
              data=json.dumps(params))


requests.post('http://127.0.0.1:8000/api/karl/reset',
              data=json.dumps({'user_id': USER_ID}))

cards_selected = []
results = []
while len(cards_selected) < 5:
    r = requests.post('http://127.0.0.1:8000/api/karl/schedule',
                      data=json.dumps(cards))
    r = json.loads(r.text)
    order = r['order']
    detail = r['detail']
    card_selected = cards[order[0]]
    card_selected['label'] = get_result(card_selected)
    cards_selected.append(card_selected)
    results.append(1 if card_selected['label'] == 'correct' else 0)

    print('{: <8} : {}'.format('card_id', card_selected['question_id']))
    print('{: <8} : {}'.format('result', card_selected['label']))
    print('{: <8} : {}'.format('text', textwrap.fill(
        card_selected['text'], width=50, subsequent_indent=' ' * 11)))
    print('{: <8} : {}'.format('answer', card_selected['answer']))
    print('{: <8} : {}'.format('category', card_selected['category']))
    print('-----------')
    for key, value in detail.items():
        if key in params:
            print('{: <8} : {:.3f} x {:.3f}'.format(key, value, params[key]))
        elif isinstance(value, str):
            print('{: <8} : {}'.format(key, value))
    print('************************')

    r = requests.post('http://127.0.0.1:8000/api/karl/update',
                      data=json.dumps([card_selected]))
    cards = [x for x in cards
             if x['question_id'] != card_selected['question_id']]
