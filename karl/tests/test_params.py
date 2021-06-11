import json
import requests


def test_set_params():
    URL = 'http://127.0.0.1:8000/api/karl'

    user_id = 'b123'
    env = 'prod'

    params = {
        'repetition_model': 'test_model',
        'card_embedding': 0.11,
        'recall': 0.22,
        'recall_target': 0.33,
        'category': 0.44,
        'answer': 0.55,
        'leitner': 0.66,
        'sm2': 0.77,
        'decay_qrep': 0.88,
        'cool_down': 0.99,
        'cool_down_time_correct': 22,
        'cool_down_time_wrong': 4,
        'max_recent_facts': 33,
    }
    r = requests.put(f'{URL}/set_params?user_id={user_id}&env={env}', data=json.dumps(params))
    print(r.text)

    r = requests.get(f'{URL}/get_params?user_id={user_id}&env={env}')
    print(r.text)


test_set_params()
