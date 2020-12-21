import json
import requests


def test_get_user_charts():
    URL = 'http://127.0.0.1:8000/api/karl'

    user_id = '463'
    env = 'prod'

    r = requests.get(f'{URL}/get_user_charts?env={env}&user_id={user_id}')
    viz = json.loads(r.text)
    # print(viz)


test_get_user_charts()
