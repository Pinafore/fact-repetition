# %%
import json
import random
import requests
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from sqlalchemy.orm import Session

from karl.models import User, StudyRecord
from karl.schemas import UserStatsSchema
from karl.db.session import SessionLocal
from karl.config import settings


URL = f'{settings.API_URL}/api/karl'


def get_session():
    return SessionLocal()


def _get_user_stats_slow(
    user_id: str,
    deck_id: str = None,
    min_studied: int = 0,
    date_start: str = '2008-06-01 08:00:00+00:00',
    date_end: str = '2038-06-01 08:00:00+00:00',
    session: Session = get_session(),
):
    date_start = parse_date(date_start).date()
    date_end = parse_date(date_end).date()

    if deck_id is None:
        deck_id = 'all'

    new_facts = 0
    reviewed_facts = 0
    new_correct = 0
    reviewed_correct = 0
    elapsed_milliseconds_text = 0
    elapsed_milliseconds_answer = 0
    day_count = {}  # date -> number of facts studied that day

    for record in session.query(StudyRecord).\
            filter(StudyRecord.user_id == user_id).\
            filter(StudyRecord.date >= date_start).\
            filter(StudyRecord.date <= date_end).\
            order_by(StudyRecord.date):

        if record.count == 0:
            new_facts += 1
            if record.label:
                new_correct += 1
        else:
            reviewed_facts += 1
            if record.label:
                reviewed_correct += 1
        elapsed_milliseconds_text += record.elapsed_milliseconds_text
        elapsed_milliseconds_answer += record.elapsed_milliseconds_answer

        date = record.date.date()
        if date not in day_count:
            day_count[date] = 0
        day_count[date] += 1

    total_seen = new_facts + reviewed_facts
    total_correct = new_correct + reviewed_correct
    known_rate = total_correct / total_seen if total_seen > 0 else 0
    new_known_rate = new_correct / new_facts if new_facts > 0 else 0
    review_known_rate = reviewed_correct / reviewed_facts if reviewed_facts > 0 else 0
    total_milliseconds = elapsed_milliseconds_text + elapsed_milliseconds_answer

    n_days_studied = 0
    for date, count in day_count.items():
        if count >= min_studied:
            n_days_studied += 1

    user_stat_schema = UserStatsSchema(
        user_id=user_id,
        deck_id=deck_id,
        date_start=str(date_start),
        date_end=str(date_end),
        new_facts=new_facts,
        reviewed_facts=reviewed_facts,
        new_correct=new_correct,
        reviewed_correct=reviewed_correct,
        total_seen=new_facts + reviewed_facts,
        total_milliseconds=total_milliseconds,
        total_seconds=total_milliseconds // 1000,
        total_minutes=total_milliseconds // 60000,
        elapsed_milliseconds_text=elapsed_milliseconds_text,
        elapsed_milliseconds_answer=elapsed_milliseconds_answer,
        # NOTE we always convert from milliseconds to avoid as much rounding error as possible
        elapsed_seconds_text=elapsed_milliseconds_text // 1000,
        elapsed_seconds_answer=elapsed_milliseconds_answer // 1000,
        elapsed_minutes_text=elapsed_milliseconds_text // 60000,
        elapsed_minutes_answer=elapsed_milliseconds_answer // 60000,
        known_rate=round(known_rate * 100, 2),
        new_known_rate=round(new_known_rate * 100, 2),
        review_known_rate=round(review_known_rate * 100, 2),
        n_days_studied=n_days_studied,
    )
    return user_stat_schema


# %%
def test_user_stats():
    for i in range(1):
        session = get_session()

        user_ids = ['463', '123', '413', '85', '496', '38']
        user_id = random.choice(user_ids)
        user = session.query(User).get(user_id)

        date_range = user.study_records[-1].date - user.study_records[0].date
        date_start = user.study_records[0].date + date_range * random.uniform(0, 0.3)
        date_end = user.study_records[0].date + date_range * random.uniform(0.7, 1.0)
        date_start = date_start.date().strftime('%Y-%m-%dT%H:%M:%S%z')
        date_end = date_end.date().strftime('%Y-%m-%dT%H:%M:%S%z')

        session.close()

        print('date_start', date_start)
        print('date_end', date_end)
        print()

        min_studied = 0

        user_stats_slow = _get_user_stats_slow(
            user_id=user_id,
            deck_id=None,
            min_studied=min_studied,
            date_start=date_start,
            date_end=date_end,
        ).__dict__

        r = requests.get(f'{URL}/get_user_stats?user_id={user_id}&date_start={date_start}&date_end={date_end}&min_studied={min_studied}')
        user_stats_fast = json.loads(r.text)

        # date_end = parse_date(date_end).date() - timedelta(days=1)
        # date_end = date_end.strftime('%Y-%m-%dT%H:%M:%S%z')

        # r = requests.get(f'{URL}/get_user_stats?user_id={user_id}&date_start={date_start}&date_end={date_end}&min_studied={min_studied}')
        # user_stats_stable = json.loads(r.text)

        for key in user_stats_slow.keys():
            a = user_stats_slow[key]
            b = user_stats_fast[key]
            # c = user_stats_stable[key]
            # print(a == b, b == c, key, a, b, c)
            print(f'{"pass" if a == b else "fail"} @ {key}, slow: {a}, fast: {b}')

    print()


def test_leaderboard():
    profile = {}
    for i in tqdm(range(10)):
        user_ids = ['463', '123', '413', '85', '496', '38']
        user_id = random.choice(user_ids)

        first_date = parse_date('2020-10-01 08:00:00.000001 -0400')
        last_date = parse_date('2020-12-01 08:00:00.000001 -0400')

        date_range = last_date - first_date
        date_start = first_date + date_range * random.uniform(0, 0.3)
        date_end = last_date + date_range * random.uniform(0.7, 1.0)
        date_start = date_start.date().strftime('%Y-%m-%dT%H:%M:%S%z')
        date_end = date_end.date().strftime('%Y-%m-%dT%H:%M:%S%z')

        min_studied = 0
        rank_type = 'n_days_studied'

        t0 = datetime.now()

        # r = requests.get(f'{URL}/leaderboard?user_id={user_id}&date_star={date_start}&date_end={date_end}&min_studied={min_studied}&rank_type={rank_type}')
        r = requests.get(f'{URL}/leaderboard/?rank_type=total_seen&limit=10&min_studied=10&user_id=44&date_start=2020-12-15+00%3A00%3A00-05%3A00')
        r = requests.get(f'{URL}/leaderboard/?rank_type=total_seen&limit=10&min_studied=10&user_id=44')
        # r = requests.get(f'{URL}/leaderboard?rank_type=total_seen&limit=10&min_studied=10&user_id=44')
        # leaderboard = json.loads(r.text)

        t1 = datetime.now()

        # r = requests.get(f'{URL}/leaderboard?user_id={user_id}&date_star={date_start}&date_end={date_end}&min_studied={min_studied}&rank_type={rank_type}')
        # r = requests.get(f'{URL}/leaderboard?rank_type=total_seen&limit=10&min_studied=10&user_id=44')
        r = requests.get(f'{URL}/leaderboard/?rank_type=total_seen&limit=10&min_studied=10&user_id=44&date_start=2020-12-15+00%3A00%3A00-05%3A00')
        r = requests.get(f'{URL}/leaderboard/?rank_type=total_seen&limit=10&min_studied=10&user_id=44')
        # leaderboard = json.loads(r.text)

        t2 = datetime.now()

        if 'new' not in profile:
            profile['new'] = []
        if 'old' not in profile:
            profile['old'] = []
        profile['new'].append(t1 - t0)
        profile['old'].append(t2 - t1)

    print()
    for key, values in profile.items():
        print(key, np.mean(values))


test_user_stats()
# test_leaderboard()
