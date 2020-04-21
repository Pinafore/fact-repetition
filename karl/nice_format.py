#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import pickle
import requests
import numpy as np
from decimal import Decimal


# In[ ]:


def generate_schedule(flashcards, questions, max_studies=20):
    '''
    questions: qid -> text, answer
    flashcards: 
    '''
    # reset model
    requests.post('http://127.0.0.1:8000/api/karl/reset')
    
    # set hyperparameteers
    params = {
        'learning_rate': 1e-3,
        'num_epochs': 40,
    }
    r = requests.post('http://127.0.0.1:8000/api/karl/set_hyperparameter',
                      data=json.dumps(params))
    
    update_flashcards = []  # for user embedding update
    previous_ranks = dict() # qid -> previous ranking
    previous_probs = dict() # qid -> previous probability
    for _ in range(max_studies):
        if len(flashcards) <= 0:
            break
        # get new ranking
        r = requests.post('http://127.0.0.1:8000/api/karl/schedule',
                          data=json.dumps(flashcards))
        r = json.loads(r.text)
        # card_order = r['card_order']
        probs = [x[0] for x in r['probs']]
        # sort
        card_order = np.argsort(np.abs(0.5 - np.asarray(probs))).tolist()
        # card_order = np.argsort(-np.asarray(probs)).tolist()
        
        # get ranking changes
        rank_diff = {}
        prob_diff = {}
        for rank, ind in enumerate(card_order):
            qid = flashcards[ind]['question_id']
            if qid in previous_ranks:
                # +1 to account for removal of the first card
                rank_diff[qid] = rank - previous_ranks[qid]
            previous_ranks[qid] = rank
            if qid in previous_probs:
                prob_diff[qid] = probs[ind] - previous_probs[qid]
            previous_probs[qid] = probs[ind]
            
        for rank, ind in enumerate(card_order[:10]):
            qid = flashcards[ind]['question_id']
            answer = flashcards[ind]['answer']
            rdiff = rank_diff.get(qid, 0)
            rdiff_s = '⇧' + str(-rdiff) if rdiff < 0 else '⇩' + str(rdiff)
            rdiff_s = '' if rdiff == 0 else '(%s)' % rdiff_s
            pdiff = prob_diff.get(qid, 0)
            pdiff_s = '%.2e' % Decimal(pdiff)
            pdiff_s = pdiff_s if pdiff < 0 else '+' + pdiff_s
            prob_s = '%.2e' % Decimal(probs[ind])
            print('%d%s\t%s (%s)\t%s' % (rank, rdiff_s, prob_s, pdiff_s, answer))
            
        # # most changes in ranking
        # rank_diff_sorted = sorted(rank_diff.items(), key=lambda x: x[1])
        # for qid, diff in rank_diff_sorted[:5]:
        #     if diff >= 0:
        #         break
        #     print('%s%d\t%.2e\t%s' % (
        #         '⇧', -diff, Decimal(prob_diff[qid]), questions[qid]['answer']))
        # for qid, diff in rank_diff_sorted[:-5:-1]:
        #     if diff <= 0:
        #         break
        #     print('%s%d\t%.2e\t%s' % (
        #         '⇩', diff, Decimal(prob_diff[qid]), questions[qid]['answer']))
            
        # # most changes in probability
        # prob_diff_sorted = sorted(prob_diff.items(), key=lambda x: -x[1])
        # for qid, diff in prob_diff_sorted[:5]:
        #     if diff <= 0:
        #         break
        #     print('%s\t%.2e\t%s' % ('⇧', Decimal(dff[qid]), questions[qid]['answer']))
        # for qid, diff in prob_diff_sorted[:-5:-1]:
        #     if diff >= 0:
        #         break
        #     print('%s\t%.2e\t%s' % ('⇩', Decimal(dff[qid]), questions[qid]['answer']))
         
        # update card pool
        show_card = flashcards[card_order[0]]
        update_flashcards = [show_card]
        if show_card['label'] == 'correct' or show_card['label'] == 'wrong':
            flashcards.pop(card_order[0])
            for qid, rank in previous_ranks.items():
                if qid != show_card['question_id']:
                    previous_ranks[qid] = rank - 1
        
        # update model
        r = requests.post('http://127.0.0.1:8000/api/karl/update',
                          data=json.dumps(update_flashcards))
        r = json.loads(r.text)
        # print(r['loss'])
        
        print()
        print('------', show_card['label'], show_card['answer'], '-------')
        print()


# In[ ]:


# protobowl data
with open('data/withanswer.question.json') as f:
    proto_questions = json.load(f)
proto_questions = {q['qid']: q for q in proto_questions}

with open('data/protobowl_byuser_dict_alldata.json') as f:
    protobowl_byuser_dict_alldata = json.load(f)


# In[ ]:


def test_protobowl(max_cards=20, max_studies=20):
    # generate protobowl flashcards
    uid_1 = list(protobowl_byuser_dict_alldata['data'].keys())[1]
    print('uid', uid_1)
    
    # create set of flashcards from study record
    flashcards = []
    user_questions = protobowl_byuser_dict_alldata['data'][uid_1]['questions_per_user']
    user_labels = protobowl_byuser_dict_alldata['data'][uid_1]['accuracy_per_user']
    qid_set = set() # for dedup
    for qid, label in zip(user_questions[:max_cards], user_labels[:max_cards]):
        # entries are ranked by date
        if qid in qid_set:
            # question dedup, only add the first occurrence
            continue
        qid_set.add(qid)
        flashcards.append({
                'text': proto_questions[qid]['text'],
                'user_id': uid_1,
                'question_id': qid,
                'label': 'correct' if label else 'wrong',
                'answer': proto_questions[qid]['answer']
            })
    print('# flashcards', len(flashcards))
    print()
    
    generate_schedule(flashcards, proto_questions, max_studies)


# In[4]:


# load jeopardy data
jeopardy_questions_df = pickle.load(open('data/jeopardy_358974_questions_20190612.pkl', 'rb'))
jeopardy_records_df = pickle.load(open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb'))

jeopardy_questions_dict = jeopardy_questions_df.to_dict()
jeopardy_questions = {}
for i in jeopardy_questions_dict['clue'].keys():
    jeopardy_questions[jeopardy_questions_dict['questionid'][i]] = {
        'text': jeopardy_questions_dict['clue'][i],
        'answer': jeopardy_questions_dict['answer'][i]
    }


# In[29]:


def test_jeopardy(max_cards=20, max_studies=20):
    # generate jeopardy flashcard
    uid_1 = 'player_020'
    print('uid', uid_1)
    
    # create set of flashcards from study record
    flashcards = []
    for row in jeopardy_records_df.groupby('player_id').get_group(uid_1).iterrows():
        if len(flashcards) >= max_cards:
            break
        qid = row[1]['question_id']
        flashcards.append({
                'text': jeopardy_questions[qid]['text'],
                'user_id': uid_1,
                'question_id': qid,
                'label': 'correct' if row[1]['correct'] == 1 else 'wrong',
                'answer': jeopardy_questions[qid]['answer']
            })
    print('# flashcards', len(flashcards))
    print()
    
    generate_schedule(flashcards, jeopardy_questions, max_studies)


# In[ ]:


test_jeopardy(max_cards=10, max_studies=10)


