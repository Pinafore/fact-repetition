import requests


def test_predict_recall():
    URL = 'http://127.0.0.1:8000/api/karl'

    user_id = 'dummy'
    card_id = 'sim_0'
    env = 'prod'

    r = requests.get(
        f'{URL}/predict_recall?env={env}&user_id={user_id}&card_id={card_id}'
    )
    print(r.text)


test_predict_recall()
