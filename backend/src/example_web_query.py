#!/usr/bin/env python
# coding: utf-8
import json
import requests
import numpy as np
from decimal import Decimal


with open('data/withanswer.question.json') as f:
    questions = json.load(f)
questions = {q['qid']: q for q in questions}

with open('data/protobowl_byuser_dict_alldata.json') as f:
    protobowl_byuser_dict_alldata = json.load(f)


# reset model
requests.post('http://127.0.0.1:8000/api/karl/reset')

# set hyperparameteers
params = {
    'learning_rate': 5e-4,
    'num_epochs': 10,
}
r = requests.post('http://127.0.0.1:8000/api/karl/set_hyperparameter',
                  data=json.dumps(params))

uid_1 = list(protobowl_byuser_dict_alldata['data'].keys())[1]
print('uid', uid_1)

# create set of flashcards from study record
flashcards = []
user_questions = protobowl_byuser_dict_alldata['data'][uid_1]['questions_per_user']
user_labels = protobowl_byuser_dict_alldata['data'][uid_1]['accuracy_per_user']
qid_set = set() # for dedup
user_questions, user_labels = user_questions[:10], user_labels[:10]
for qid, label in zip(user_questions, user_labels):
    # entries are ranked by date
    if qid in qid_set:
        # question dedup, only add the first occurrence
        continue
    qid_set.add(qid)
    flashcards.append({
            'text': questions[qid]['text'][:100],
            'user_id': uid_1,
            'question_id': qid,
            'label': 'correct' if label else 'wrong',
            'answer': questions[qid]['answer']
        })
print('# flashcards', len(flashcards))
print()

update_flashcards = []  # for user embedding update
previous_ranks = dict() # qid -> previous ranking
previous_probs = dict() # qid -> previous probability
for _ in range(20):
    if len(flashcards) <= 0:
        break
    # get new ranking
    r = requests.post('http://127.0.0.1:8000/api/karl/schedule',
                      data=json.dumps(flashcards))
    r = json.loads(r.text)
    # card_order = r['card_order']
    probs = [x[0] for x in r['probs']]
    # sort
    # card_order = np.argsort(np.abs(0.5 - np.asarray(probs))).tolist()
    card_order = np.argsort(-np.asarray(probs)).tolist()
    
    # print top-5 questions
    for rank, ind in enumerate(card_order[:5]):
        print(rank, '%.2e' % Decimal(probs[ind]), flashcards[ind]['answer'])
    
    # get ranking changes
    rank_diff = {}
    prob_diff = {}
    for rank, ind in enumerate(card_order):
        qid = flashcards[ind]['question_id']
        if qid in previous_ranks:
            # +1 to account for removal of the first card
            rank_diff[qid] = rank - previous_ranks[qid] + 1
        previous_ranks[qid] = rank
        if qid in previous_probs:
            prob_diff[qid] = probs[ind] - previous_probs[qid]
        previous_probs[qid] = probs[ind]
        
    # rank_diff = sorted(rank_diff.items(), key=lambda x: x[1])
    # for qid, diff in rank_diff[:5]:
    #     if diff >= 0:
    #         break
    #     s = '⇧' + str(-diff)
    #     print(s, questions[qid]['answer'])
    # for qid, diff in rank_diff[:-5:-1]:
    #     if diff <= 0:
    #         break
    #     s = '⇩' + str(diff)
    #     print(s, questions[qid]['answer'])
        
    # increase in probabilty
    prob_diff = sorted(prob_diff.items(), key=lambda x: -x[1])
    for qid, diff in prob_diff[:5]:
        if diff <= 0:
            break
        s = '⇧' + '%.2e' % Decimal(diff)
        print(s, questions[qid]['answer'])
    for qid, diff in prob_diff[:-5:-1]:
        if diff >= 0:
            break
        s = '⇩' + '%.2e' % Decimal(diff)
        print(s, questions[qid]['answer'])
    
    # update card pool
    show_card = flashcards[card_order[0]]
    update_flashcards = [show_card]
    if show_card['label'] == 'correct':
        flashcards.pop(card_order[0])
    
    # update model
    r = requests.post('http://127.0.0.1:8000/api/karl/update',
                      data=json.dumps(update_flashcards))
    r = json.loads(r.text)
    # print(r['loss'])
    
    print()
    print('------', show_card['label'], show_card['answer'], '-------')
    print()
