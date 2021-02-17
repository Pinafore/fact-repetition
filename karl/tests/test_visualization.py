import json
import requests


def test_get_user_charts():
    URL = 'http://127.0.0.1:8000/api/karl'

    user_id = '463'
    env = 'prod'
    date_start = '2020-08-23'
    date_end = '2020-11-01'

    r = requests.get(f'{URL}/get_user_charts?env={env}&user_id={user_id}&date_start={date_start}&date_end={date_end}')
    viz = json.loads(r.text)
    print(viz)


test_get_user_charts()
