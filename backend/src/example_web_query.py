import json
import requests
import numpy as np


with open('data/withanswer.question.json') as f:
    question_data = json.load(f)
question_data = {q['qid']: q for q in question_data}

with open('data/protobowl_byuser_dict_alldata.json') as f:
    protobowl_byuser_dict_alldata = json.load(f)


uid_1 = '82990a64a4a8f77352144bb626074b5c2f35baf1'
print('uid', uid_1)
questions = protobowl_byuser_dict_alldata['data'][uid_1]['questions_per_user']
labels = protobowl_byuser_dict_alldata['data'][uid_1]['accuracy_per_user']
proto_flashcards = [
    {
        'text': question_data[qid]['text'][:100].lower(),
        'user_id': uid_1,
        'question_id': qid,
        'label': 'correct' if label else 'wrong'
    } for qid, label in zip(questions, labels)
]


while len(proto_flashcards) > 0:
    r = requests.post('http://127.0.0.1:8000/api/karl/schedule',
                      data=json.dumps(proto_flashcards[:10]))
    r = json.loads(r.text)
    ind = r['card_order'][0]
    qid = proto_flashcards[ind]['question_id']
    question = question_data[qid]
    print(question['answer'])
    print(question['text'][:100])
    print()
    proto_flashcards.pop(ind)
