#!/usr/bin/env python
# coding: utf-8

import json
import pickle
import requests
import numpy as np


def get_result(card):
    record_id = karl_to_question_id[int(card['question_id'])]
    prob = (records_df[records_df.question_id == record_id]['correct'] / 2 + 0.5).mean()
    return 'correct' if prob > 0.5 else 'wrong'


with open('data/jeopardy_358974_questions_20190612.pkl', 'rb') as f:
    karl_to_question_id = pickle.load(f).to_dict()['question_id']
with open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb') as f:
    records_df = pickle.load(f)
with open('data/diagnostic_questions.pkl', 'rb') as f:
    cards = pickle.load(f)
    for i, card in enumerate(cards):
        cards[i]['user_id'] = 'diagnostic'

requests.post('http://127.0.0.1:8000/api/karl/reset')

shown_cards = []
results = []
while len(shown_cards) < 5:
    r = requests.post('http://127.0.0.1:8000/api/karl/schedule', data=json.dumps(cards))
    r = json.loads(r.text)
    show_card = cards[r['order'][0]]
    show_card['label'] = get_result(show_card)
    shown_cards.append(show_card)
    results.append(1 if show_card['label'] == 'correct' else 0)
    print('{: <40}{: <20}{: <20}{: <20}'.format(
        show_card['answer'],
        show_card['category'],
        show_card['label'],
        np.mean(results)
    ))
    print(r['rationale'])
    print()
    r = requests.post('http://127.0.0.1:8000/api/karl/update', data=json.dumps([show_card]))
    cards = [x for x in cards if x['question_id'] != show_card['question_id']]
