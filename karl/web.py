#!/usr/bin/env python
# coding: utf-8

import json
import logging
from fastapi import FastAPI, HTTPException
from typing import List
from datetime import datetime
from dateutil.parser import parse as parse_date
from concurrent.futures import ProcessPoolExecutor
# from cachetools import cached, TTLCache
from sqlalchemy.exc import SQLAlchemyError

from karl.util import ScheduleRequest, Params, \
    Ranking, Leaderboard, UserStatSchema, Visualization, SchedulerOutputSchema
from karl.util import get_session_makers
from karl.models import User, Fact, UserStat
from karl.scheduler import MovingAvgScheduler
from karl.metrics import get_user_charts


default_start_date = '2008-06-01 08:00:00.000001 -0400'
default_end_date = '2038-06-01 08:00:00.000001 -0400'


app = FastAPI()
scheduler = MovingAvgScheduler(preemptive=False)
session_makers = get_session_makers()

# create logger with 'scheduler'
logger = logging.getLogger('scheduler')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('/fs/www-users/shifeng/scheduler.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def get_session(env: str = 'prod'):
    return session_makers[env]()


@app.post('/api/karl/schedule')
def schedule(
    requests: List[ScheduleRequest],
) -> SchedulerOutputSchema:
    if len(requests) == 0:
        return {
            'order': [],
            'rationale': '<p>no fact received</p>',
            'facts_info': '',
        }

    logger.info(f'/karl/schedule with {len(requests)} facts and env={requests[0].env}')

    # NOTE assuming single user single date
    date = parse_date(datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'))
    if requests[0].date is not None:
        date = parse_date(requests[0].date)

    env = 'dev' if requests[0].env == 'dev' else 'prod'
    session = get_session(env)

    try:
        return scheduler.schedule(session, requests, date, plot=False)
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()
        raise HTTPException(status_code=404, detail='Scheduling failed due to SQLAlchemyError.')
    finally:
        session.close()


@app.post('/api/karl/update')
def update(
    requests: List[ScheduleRequest],
    response_model=bool,
):
    logger.info(f'/karl/update with {len(requests)} facts and env={requests[0].env} and debug_id={requests[0].debug_id}')

    # NOTE assuming single user single date
    date = parse_date(datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'))
    if requests[0].date is not None:
        date = parse_date(requests[0].date)

    env = 'dev' if requests[0].env == 'dev' else 'prod'
    session = get_session(env)

    try:
        scheduler.update(session, requests, date)
        session.commit()
        return True
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()
        raise HTTPException(status_code=404, detail='Update failed due to SQLAlchemyError.')
    finally:
        session.close()


@app.put('/api/karl/set_params', response_model=Params)
def set_params(
    user_id: str,
    env: str,
    params: Params,
):
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)
    try:
        scheduler.set_user_params(session, user_id, params)
        session.commit()
        return params
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()
        raise HTTPException(status_code=404, detail='Set_params failed due to SQLAlchemyError.')
    finally:
        session.close()


@app.put('/api/karl/set_repetition_model', response_model=Params)
def set_repetition_model(
    user_id: str,
    env: str,
    repetition_model: str,
):
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    if repetition_model == 'sm2':
        params = Params(
            repetition_model='sm2',
            qrep=0,
            skill=0,
            recall=0,
            category=0,
            answer=0,
            leitner=0,
            sm2=1,
            cool_down=0,
            recall_target=1,
        )
        scheduler.set_user_params(session, user_id, params)
        session.commit()
        session.close()
        return params
    elif repetition_model == 'leitner':
        params = Params(
            repetition_model='leitner',
            qrep=0,
            skill=0,
            recall=0,
            category=0,
            answer=0,
            leitner=1,
            sm2=0,
            cool_down=0,
            recall_target=1,
        )
        scheduler.set_user_params(session, user_id, params)
        session.commit()
        session.close()
        return params
    elif repetition_model.startswith('karl'):
        recall_target = int(repetition_model[4:])
        params = Params(
            repetition_model=f'karl{recall_target}',
            qrep=1,
            skill=0,
            recall=1,
            category=1,
            answer=1,
            leitner=1,
            sm2=0,
            recall_target=float(recall_target) / 100,
        )
        scheduler.set_user_params(session, user_id, params)
        session.commit()
        session.close()
        return params
    else:
        raise HTTPException(status_code=404, detail='Unrecognized repetition model.')


@app.post('/api/karl/get_fact', response_model=dict)
def get_fact(
    fact_id: str,
    env: str,
):
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    fact = session.query(Fact).get(fact_id)
    if fact is None:
        return
    fact_json = json.dumps({
        k: v for k, v in fact.__dict__.items() if k != '_sa_instance_state'
    })

    session.close()
    return fact_json


@app.get('/api/karl/reset_user', response_model=dict)
def reset_user(
    user_id: str = None,
    env: str = None,
):
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    try:
        scheduler.reset_user(session, user_id=user_id)
        session.commit()
        return get_user(user_id, env)
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()
        raise HTTPException(status_code=404, detail='Reset user failed.')
    finally:
        session.close()


@app.get('/api/karl/reset_fact', response_model=dict)
def reset_fact(
    fact_id: str = None,
    env: str = None,
):
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    try:
        scheduler.reset_fact(session, fact_id=fact_id)
        session.commit()
        return get_fact(fact_id, env)
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()
    finally:
        session.close()


@app.get('/api/karl/status')
def status():
    return True


@app.get('/api/karl/get_user')
def get_user(
    user_id: str,
    env: str = None,
):
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    user = scheduler.get_user(session, user_id)
    user_dict = {
        k: v for k, v in user.__dict__.items()
        if k != '_sa_instance_state'
    }
    user_dict['params'] = user_dict['params'].__dict__
    user_json = json.dumps(user_dict)

    session.close()
    return user_json


def _get_user_history(
    user_id: str,
    env: str = None,
    deck_id: str = None,
    date_start: str = '2008-06-01 08:00:00.000001 -0400',
    date_end: str = '2038-06-01 08:00:00.000001 -0400',
):
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    history = scheduler.get_records(
        session,
        user_id,
        deck_id,
        date_start,
        date_end
    )
    session.close()
    return history


def _get_user_stats(
    user_id: str,
    env: str = None,
    deck_id: str = None,
    min_studied: int = 0,
    date_start: str = '2008-06-01 08:00:00.000001 -0400',
    date_end: str = '2038-06-01 08:00:00.000001 -0400',
) -> UserStatSchema:
    '''
    Return in a dictionary the following user stats within given date range.

    new_facts: int
    reviewed_facts: int
    total_seen: int
    total_seconds: int
    known_rate: float
    new_known_rate: float
    review_known_rate: float
    '''
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    date_start = parse_date(date_start).date()
    date_end = parse_date(date_end).date()

    if deck_id is None:
        deck_id = 'all'

    # last record before start date
    before_stat = session.query(UserStat).\
        filter(UserStat.user_id == user_id).\
        filter(UserStat.deck_id == deck_id).\
        filter(UserStat.date < date_start).\
        order_by(UserStat.date.desc()).\
        first()

    # last record no later than end date
    after_stat = session.query(UserStat).\
        filter(UserStat.user_id == user_id).\
        filter(UserStat.deck_id == deck_id).\
        filter(UserStat.date < date_end).\
        order_by(UserStat.date.desc()).\
        first()

    if after_stat is None or after_stat.date < date_start:
        session.close()
        return UserStatSchema(
            user_id=user_id,
            deck_id=deck_id,
            date_start=str(date_start),
            date_end=str(date_end),
            # zero for all other fields
        )

    if before_stat is None:
        before_stat = UserStat(
            user_stat_id=json.dumps({
                'user_id': user_id,
                'date': str(date_start),
                'deck_id': deck_id,
            }),
            user_id=user_id,
            deck_id=deck_id,
            date=date_start,
            new_facts=0,
            reviewed_facts=0,
            new_correct=0,
            reviewed_correct=0,
            total_seen=0,
            total_milliseconds=0,
            total_seconds=0,
            total_minutes=0,
            elapsed_milliseconds_text=0,
            elapsed_milliseconds_answer=0,
            elapsed_seconds_text=0,
            elapsed_seconds_answer=0,
            elapsed_minutes_text=0,
            elapsed_minutes_answer=0,
            n_days_studied=0,
        )

    total_correct = (
        after_stat.new_correct
        + after_stat.reviewed_correct
        - before_stat.new_correct
        - before_stat.reviewed_correct
    )

    known_rate = 0
    if after_stat.total_seen > before_stat.total_seen:
        known_rate = (
            total_correct / (after_stat.total_seen - before_stat.total_seen)
        )

    new_known_rate = 0
    if after_stat.new_facts > before_stat.new_facts:
        new_known_rate = (
            (after_stat.new_correct - before_stat.new_correct)
            / (after_stat.new_facts - before_stat.new_facts)
        )

    review_known_rate = 0
    if after_stat.reviewed_facts > before_stat.reviewed_facts:
        review_known_rate = (
            (after_stat.reviewed_correct - before_stat.reviewed_correct)
            / (after_stat.reviewed_facts - before_stat.reviewed_facts)
        )

    total_milliseconds = after_stat.total_milliseconds - before_stat.total_milliseconds
    elapsed_milliseconds_text = after_stat.elapsed_milliseconds_text - before_stat.elapsed_milliseconds_text
    elapsed_milliseconds_answer = after_stat.elapsed_milliseconds_answer - before_stat.elapsed_milliseconds_answer

    if min_studied > 0:
        n_days_studied = 0
        prev_total_seen = before_stat.total_seen
        # go through user stats within the interval
        for user_stat in session.query(UserStat).\
                filter(UserStat.user_id == user_id).\
                filter(UserStat.deck_id == deck_id).\
                filter(UserStat.date >= date_start, UserStat.date < date_end).\
                order_by(UserStat.date):
            if user_stat.total_seen - prev_total_seen >= min_studied:
                n_days_studied += 1
            prev_total_seen = user_stat.total_seen
    else:
        n_days_studied = after_stat.n_days_studied - before_stat.n_days_studied

    user_stat_schema = UserStatSchema(
        user_id=user_id,
        deck_id=deck_id,
        date_start=str(date_start),
        date_end=str(date_end),
        new_facts=after_stat.new_facts - before_stat.new_facts,
        reviewed_facts=after_stat.reviewed_facts - before_stat.reviewed_facts,
        new_correct=after_stat.new_correct - before_stat.new_correct,
        reviewed_correct=after_stat.reviewed_correct - before_stat.reviewed_correct,
        total_seen=after_stat.total_seen - before_stat.total_seen,
        total_milliseconds=total_milliseconds,
        # NOTE we always convert from milliseconds to avoid as much rounding error as possible
        total_seconds=total_milliseconds // 1000,
        total_minutes=total_milliseconds // 60000,
        elapsed_milliseconds_text=elapsed_milliseconds_text,
        elapsed_milliseconds_answer=elapsed_milliseconds_answer,
        elapsed_seconds_text=elapsed_milliseconds_text // 1000,
        elapsed_seconds_answer=elapsed_milliseconds_answer // 1000,
        elapsed_minutes_text=elapsed_milliseconds_text // 60000,
        elapsed_minutes_answer=elapsed_milliseconds_answer // 60000,
        known_rate=round(known_rate * 100, 2),
        new_known_rate=round(new_known_rate * 100, 2),
        review_known_rate=round(review_known_rate * 100, 2),
        n_days_studied=n_days_studied,
    )

    session.close()
    return user_stat_schema


@app.get('/api/karl/get_user_stats', response_model=UserStatSchema)
# @cached(cache=TTLCache(maxsize=1024, ttl=600))
def get_user_stats(
    user_id: str,
    env: str = None,
    deck_id: str = None,
    min_studied: int = 0,
    date_start: str = '2008-06-01 08:00:00.000001 -0400',
    date_end: str = '2038-06-01 08:00:00.000001 -0400',
) -> UserStatSchema:
    return _get_user_stats(
        user_id=user_id,
        env=env,
        min_studied=min_studied,
        deck_id=deck_id,
        date_start=date_start,
        date_end=date_end
    )


@app.get('/api/karl/leaderboard', response_model=Leaderboard)
# @cached(cache=TTLCache(maxsize=1024, ttl=1800))
def get_leaderboard(
    user_id: str = None,
    env: str = None,
    skip: int = 0,
    limit: int = 10,
    rank_type: str = 'total_seen',
    min_studied: int = 0,
    deck_id: str = None,
    date_start: str = '2008-06-01 08:00:00.000001 -0400',
    date_end: str = '2038-06-01 08:00:00.000001 -0400',
) -> Leaderboard:
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    executor = ProcessPoolExecutor()
    stats = {}
    for user in session.query(User):
        if not user.user_id.isdigit():
            continue

        stats[user.user_id] = executor.submit(
            _get_user_stats,
            user_id=user.user_id,
            deck_id=deck_id,
            min_studied=min_studied,
            date_start=date_start,
            date_end=date_end,
        )

    for user_id, future in stats.items():
        stats[user_id] = future.result()

    # from high value to low
    stats = sorted(stats.items(), key=lambda x: x[1].__dict__[rank_type])[::-1]
    stats = [(k, v) for k, v in stats if v.total_seen >= min_studied]

    rankings = []
    user_place = None
    for i, (k, v) in enumerate(stats):
        if user_id == k:
            user_place = i
        rankings.append(Ranking(
            user_id=k,
            rank=i + 1,
            value=v.__dict__[rank_type]
        ))

    session.close()

    return Leaderboard(
        leaderboard=rankings[skip: skip + limit],
        total=len(rankings),
        rank_type=rank_type,
        user_place=user_place,
        user_id=user_id,
        skip=skip,
        limit=limit,
    )


@app.get('/api/karl/user_charts')
def user_charts(
    user_id: str = None,
    env: str = None,
    deck_id: str = None,
    date_start: str = '2008-06-01 08:00:00.000001 -0400',
    date_end: str = '2038-06-01 08:00:00.000001 -0400',
) -> List[Visualization]:
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    charts = get_user_charts(
        session,
        user_id=user_id,
        deck_id=deck_id,
        date_start=date_start,
        date_end=date_end,
    )

    visualizations = []
    for chart_name, chart in charts.items():
        visualizations.append(
            Visualization(
                name=chart_name,
                specs=chart.to_json(),
                user_id=user_id,
                env=env,
                deck_id=deck_id,
                date_start=date_start,
                date_end=date_end,
            )
        )

    session.close()
    return visualizations
