#!/usr/bin/env python
# coding: utf-8

import json
import logging
from fastapi import FastAPI, HTTPException, Depends
from typing import List, Generator
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
# from cachetools import cached, TTLCache
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

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
session_maker = get_session_makers()['prod']

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


def get_session() -> Generator:
    try:
        session = session_maker()
        yield session
    finally:
        session.close()


@app.post('/api/karl/schedule')
def schedule(
    requests: List[ScheduleRequest],
    session: Session = Depends(get_session),
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

    # env = 'dev' if requests[0].env == 'dev' else 'prod'
    # session = get_session(env)

    try:
        return scheduler.schedule(session, requests, date, plot=False)
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()
        raise HTTPException(status_code=404, detail='Scheduling failed due to SQLAlchemyError.')


@app.post('/api/karl/update')
def update(
    requests: List[ScheduleRequest],
    response_model=bool,
    session: Session = Depends(get_session),
):
    logger.info(f'/karl/update with {len(requests)} facts and env={requests[0].env} and debug_id={requests[0].debug_id}')

    # NOTE assuming single user single date
    date = parse_date(datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'))
    if requests[0].date is not None:
        date = parse_date(requests[0].date)

    # env = 'dev' if requests[0].env == 'dev' else 'prod'
    # session = get_session(env)

    try:
        scheduler.update(session, requests, date)
        session.commit()
        return True
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()
        raise HTTPException(status_code=404, detail='Update failed due to SQLAlchemyError.')


@app.put('/api/karl/set_params', response_model=Params)
def set_params(
    user_id: str,
    env: str,
    params: Params,
    session: Session = Depends(get_session),
):
    # env = 'dev' if env == 'dev' else 'prod'
    # session = get_session(env)
    try:
        scheduler.set_user_params(session, user_id, params)
        session.commit()
        return params
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()
        raise HTTPException(status_code=404, detail='Set_params failed due to SQLAlchemyError.')


@app.put('/api/karl/set_repetition_model', response_model=Params)
def set_repetition_model(
    user_id: str,
    env: str,
    repetition_model: str,
    session: Session = Depends(get_session),
):
    # env = 'dev' if env == 'dev' else 'prod'
    # session = get_session(env)
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
        return params
    else:
        raise HTTPException(status_code=404, detail='Unrecognized repetition model.')


@app.post('/api/karl/get_fact', response_model=dict)
def get_fact(
    fact_id: str,
    env: str,
    session: Session = Depends(get_session),
):
    # env = 'dev' if env == 'dev' else 'prod'
    # session = get_session(env)
    fact = session.query(Fact).get(fact_id)
    if fact is None:
        return
    return json.dumps({
        k: v for k, v in fact.__dict__.items() if k != '_sa_instance_state'
    })


@app.get('/api/karl/reset_user', response_model=dict)
def reset_user(
    user_id: str = None,
    env: str = None,
    session: Session = Depends(get_session),
):
    # env = 'dev' if env == 'dev' else 'prod'
    # session = get_session(env)
    try:
        scheduler.reset_user(session, user_id=user_id)
        session.commit()
        return get_user(user_id, env)
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()
        raise HTTPException(status_code=404, detail='Reset user failed.')


@app.get('/api/karl/reset_fact', response_model=dict)
def reset_fact(
    fact_id: str = None,
    env: str = None,
    session: Session = Depends(get_session),
):
    # env = 'dev' if env == 'dev' else 'prod'
    # session = get_session(env)
    try:
        scheduler.reset_fact(session, fact_id=fact_id)
        session.commit()
        return get_fact(fact_id, env)
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()


@app.get('/api/karl/status')
def status():
    return True


@app.get('/api/karl/get_user')
def get_user(
    user_id: str,
    env: str = None,
    session: Session = Depends(get_session),
):
    # env = 'dev' if env == 'dev' else 'prod'
    # session = get_session(env)
    user = scheduler.get_user(session, user_id)
    user_dict = {
        k: v for k, v in user.__dict__.items()
        if k != '_sa_instance_state'
    }
    user_dict['params'] = user_dict['params'].__dict__
    return json.dumps(user_dict)


# @app.get('/api/karl/get_user_history', response_model=List[dict])
def get_user_history(
    user_id: str,
    env: str = None,
    deck_id: str = None,
    date_start: str = '2008-06-01 08:00:00.000001 -0400',
    date_end: str = '2038-06-01 08:00:00.000001 -0400',
    session: Session = Depends(get_session),
):
    # env = 'dev' if env == 'dev' else 'prod'
    # session = get_session(env)
    return scheduler.get_records(
        session,
        user_id,
        deck_id,
        date_start,
        date_end
    )


@app.get('/api/karl/get_user_stats', response_model=dict)
# @cached(cache=TTLCache(maxsize=1024, ttl=600))
def get_user_stats(
    user_id: str,
    env: str = None,
    deck_id: str = None,
    date_start: str = '2008-06-01 08:00:00.000001 -0400',
    date_end: str = '2038-06-01 08:00:00.000001 -0400',
    session: Session = Depends(get_session),
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
    # env = 'dev' if env == 'dev' else 'prod'
    # session = get_session(env)
    return scheduler.get_user_stats(session, user_id, deck_id, date_start, date_end)


def n_days_studied(
    session: Session = None,
    user_id: str = None,
    env: str = None,
    skip: int = 0,
    limit: int = 10,
    rank_type: str = 'total_seen',
    min_studied: int = 0,
    deck_id: str = None,
    date_start: str = '2008-06-01 08:00:00.000001 -0400',
    date_end: str = '2038-06-01 08:00:00.000001 -0400',
):
    # env = 'dev' if env == 'dev' else 'prod'
    # session = get_session()

    date_start = parse_date(date_start).date()
    date_end = parse_date(date_end).date() + timedelta(days=1)  # TODO temporary fix

    deck_id = 'all' if deck_id is None else deck_id

    n_days = {}  # user_id -> (number of days studied #cards >= min_studied)
    for user in session.query(User):
        if not user.user_id.isdigit():
            continue

        # find all UserStats from the interval [date_start, date_end),
        # there should be one on each day the user studied > 0 cards.
        # enumerate through those UserStats, and subtract the `total_seen` from
        # the previous one to check if the user studied >= min_studied cards.

        # the last user stat before the interval
        before_stat = session.query(UserStat).\
            filter(UserStat.user_id == user_id).\
            filter(UserStat.deck_id == deck_id).\
            filter(UserStat.date < date_start).\
            order_by(UserStat.date.desc()).first()

        prev_total_seen = 0 if before_stat is None else before_stat.total_seen
        n_days[user.user_id] = 0

        # go through user stats within the interval
        for user_stat in session.query(UserStat).\
                filter(UserStat.user_id == user.user_id).\
                filter(UserStat.deck_id == deck_id).\
                filter(UserStat.date >= date_start, UserStat.date < date_end).\
                order_by(UserStat.date):
            if user_stat.total_seen - prev_total_seen >= min_studied:
                n_days[user.user_id] += 1
            prev_total_seen = user_stat.total_seen

    # from high value to low
    n_days = sorted(n_days.items(), key=lambda x: x[1])[::-1]
    # remove users with 0 days studied
    n_days = [(k, v) for k, v in n_days if v > 0]

    rankings = []
    user_place = None
    for i, (k, v) in enumerate(n_days):
        if user_id == k:
            user_place = i
        rankings.append(Ranking(user_id=k, rank=i + 1, value=v))

    leaderboard = Leaderboard(
        leaderboard=rankings[skip: skip + limit],
        total=len(rankings),
        rank_type=rank_type,
        user_place=user_place,
        user_id=user_id,
        skip=skip,
        limit=limit,
    )

    session.close()

    return leaderboard


@app.get('/api/karl/leaderboard', response_model=Leaderboard)
# @cached(cache=TTLCache(maxsize=1024, ttl=1800))
def leaderboard(
    user_id: str = None,
    env: str = None,
    skip: int = 0,
    limit: int = 10,
    rank_type: str = 'total_seen',
    min_studied: int = 0,
    deck_id: str = None,
    date_start: str = '2008-06-01 08:00:00.000001 -0400',
    date_end: str = '2038-06-01 08:00:00.000001 -0400',
    session: Session = Depends(get_session),
) -> Leaderboard:
    '''
    return [(user_id: str, rank_type: 'total_seen', value: 'value')]
    that ranks [skip: skip + limit)
    '''
    if rank_type == 'n_days_studied':
        return n_days_studied(
            session,
            user_id,
            env,
            skip,
            limit,
            rank_type,
            min_studied,
            deck_id,
            date_start,
            date_end,
        )

    # env = 'dev' if env == 'dev' else 'prod'
    # session = get_session()

    stats = {}
    for user in scheduler.get_all_users(session):
        if not user.user_id.isdigit():
            continue

        stats[user.user_id] = get_user_stats(
            user_id=user.user_id,
            env=env,
            deck_id=deck_id,
            date_start=date_start,
            date_end=date_end,
            session=session,
        )

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

    leaderboard = Leaderboard(
        leaderboard=rankings[skip: skip + limit],
        total=len(rankings),
        rank_type=rank_type,
        user_place=user_place,
        user_id=user_id,
        skip=skip,
        limit=limit,
    )

    session.close()

    return leaderboard


@app.get('/api/karl/user_charts')
def user_charts(
    user_id: str = None,
    env: str = None,
    deck_id: str = None,
    date_start: str = '2008-06-01 08:00:00.000001 -0400',
    date_end: str = '2038-06-01 08:00:00.000001 -0400',
    session: Session = Depends(get_session),
) -> List[Visualization]:
    # env = 'dev' if env == 'dev' else 'prod'
    # session = get_session(env)

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
    return visualizations
