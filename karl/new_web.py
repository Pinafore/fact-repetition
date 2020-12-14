#!/usr/bin/env python
# coding: utf-8

import json
import logging
from fastapi import FastAPI, HTTPException
from dateutil.parser import parse as parse_date
from cachetools import cached, TTLCache
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.exc import SQLAlchemyError

from karl.schemas import UserStatsSchema, Ranking, Leaderboard, \
    OldParametersSchema, ParametersSchema
from karl.models import User, UserStats, Parameters
from karl.db.session import SessionLocal


app = FastAPI()

# create logger with 'scheduler'
logger = logging.getLogger('scheduler')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('/Users/shifeng/workspace/fact-repetition/scheduler.log')
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
    return SessionLocal()


@app.put('/api/karl/set_params', response_model=ParametersSchema)
def set_params(
    user_id: str,
    env: str,
    params: OldParametersSchema,
) -> ParametersSchema:
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    try:
        curr_params = session.query(Parameters).get(user_id)
        is_new_params = False
        if curr_params is None:
            is_new_params = True
            curr_params = Parameters(id=user_id)

        curr_params.repetition_model = params.repetition_model
        curr_params.card_embedding = params.qrep
        curr_params.recall = params.recall
        curr_params.recall_target = params.recall_target
        curr_params.category = params.category
        curr_params.answer = params.answer
        curr_params.leitner = params.leitner
        curr_params.sm2 = params.sm2
        curr_params.decay_qrep = params.decay_qrep
        curr_params.cool_down = params.cool_down
        curr_params.cool_down_time_correct = params.cool_down_time_correct
        curr_params.cool_down_time_wrong = params.cool_down_time_wrong
        curr_params.max_recent_facts = params.max_recent_facts

        if is_new_params:
            session.add(User(id=user_id))
            session.add(curr_params)

        return_value = ParametersSchema(**curr_params.__dict__)

        session.commit()
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()
        raise HTTPException(status_code=404, detail='Set_params failed due to SQLAlchemyError.')
    finally:
        session.close()

    return return_value


@app.get('/api/karl/get_params', response_model=ParametersSchema)
def get_params(
    user_id: str,
    env: str,
) -> ParametersSchema:
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    try:
        curr_params = session.query(Parameters).get(user_id)
        if curr_params is None:
            curr_params = Parameters(id=user_id)
        return_value = ParametersSchema(**curr_params.__dict__)
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()
        raise HTTPException(status_code=404, detail='Set_params failed due to SQLAlchemyError.')
    finally:
        session.close()

    return return_value


def _get_user_stats(
    user_id: str,
    deck_id: str = None,
    min_studied: int = 0,
    date_start: str = '2008-06-01 08:00:00+00:00',
    date_end: str = '2038-06-01 08:00:00+00:00',
    env: str = None,
) -> UserStatsSchema:
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    date_start = parse_date(date_start).date()
    date_end = parse_date(date_end).date()

    if deck_id is None:
        deck_id = 'all'

    # last record before start date
    before = session.query(UserStats).\
        filter(UserStats.user_id == user_id).\
        filter(UserStats.deck_id == deck_id).\
        filter(UserStats.date < date_start).\
        order_by(UserStats.date.desc()).\
        first()

    # last record no later than end date
    after = session.query(UserStats).\
        filter(UserStats.user_id == user_id).\
        filter(UserStats.deck_id == deck_id).\
        filter(UserStats.date < date_end).\
        order_by(UserStats.date.desc()).\
        first()

    if after is None or after.date < date_start:
        session.close()
        return UserStatsSchema(
            user_id=user_id,
            deck_id=deck_id,
            date_start=str(date_start),
            date_end=str(date_end),
            # zero for all other fields
        )

    if before is None:
        before = UserStats(
            id=json.dumps({
                'user_id': user_id,
                'deck_id': deck_id,
                'date': str(date_start),
            }),
            user_id=user_id,
            deck_id=deck_id,
            date=date_start,
            n_new_cards_total=0,
            n_old_cards_total=0,
            n_new_cards_positive=0,
            n_old_cards_positive=0,
            elapsed_milliseconds_text=0,
            elapsed_milliseconds_answer=0,
            n_days_studied=0,
        )

    n_cards_positive = (
        after.n_new_cards_positive
        + after.n_old_cards_positive
        - before.n_new_cards_positive
        - before.n_old_cards_positive
    )
    n_cards_total_before = before.n_new_cards_total + before.n_old_cards_total
    n_cards_total_after = after.n_new_cards_total + after.n_old_cards_total

    rate_positive = 0
    if n_cards_total_after > n_cards_total_before:
        rate_positive = (
            n_cards_positive / (n_cards_total_after - n_cards_total_before)
        )

    rate_new_positive = 0
    if after.n_new_cards_total > before.n_new_cards_total:
        rate_new_positive = (
            (after.n_new_cards_positive - before.n_new_cards_positive)
            / (after.n_new_cards_total - before.n_new_cards_total)
        )

    rate_old_positive = 0
    if after.n_old_cards_total > before.n_old_cards_total:
        rate_old_positive = (
            (after.n_old_cards_positive - before.n_old_cards_positive)
            / (after.n_old_cards_total - before.n_old_cards_total)
        )

    total_milliseconds = (
        after.elapsed_milliseconds_text
        + after.elapsed_milliseconds_answer
        - before.elapsed_milliseconds_text
        - before.elapsed_milliseconds_answer
    )
    elapsed_milliseconds_text = after.elapsed_milliseconds_text - before.elapsed_milliseconds_text
    elapsed_milliseconds_answer = after.elapsed_milliseconds_answer - before.elapsed_milliseconds_answer

    if min_studied > 0:
        n_days_studied = 0
        prev_n_cards_total = before.n_new_cards_total + before.n_old_cards_total
        # go through user stats within the interval
        for user_stat in session.query(UserStats).\
                filter(UserStats.user_id == user_id).\
                filter(UserStats.deck_id == deck_id).\
                filter(UserStats.date >= date_start, UserStats.date < date_end).\
                order_by(UserStats.date):
            curr_n_cards_total = user_stat.n_new_cards_total + user_stat.n_old_cards_total
            if curr_n_cards_total - prev_n_cards_total >= min_studied:
                n_days_studied += 1
            prev_n_cards_total = curr_n_cards_total
    else:
        n_days_studied = after.n_days_studied - before.n_days_studied

    user_stat_schema = UserStatsSchema(
        user_id=user_id,
        deck_id=deck_id,
        date_start=str(date_start),
        date_end=str(date_end),
        new_facts=after.n_new_cards_total - before.n_new_cards_total,
        reviewed_facts=after.n_old_cards_total - before.n_old_cards_total,
        new_correct=after.n_new_cards_positive - before.n_new_cards_positive,
        reviewed_correct=after.n_old_cards_positive - before.n_old_cards_positive,
        total_seen=n_cards_total_after - n_cards_total_before,
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
        known_rate=round(rate_positive * 100, 2),
        new_known_rate=round(rate_new_positive * 100, 2),
        review_known_rate=round(rate_old_positive * 100, 2),
        n_days_studied=n_days_studied,
    )

    session.close()
    return user_stat_schema


@app.get('/api/karl/get_user_stats', response_model=UserStatsSchema)
@cached(cache=TTLCache(maxsize=1024, ttl=600))
def get_user_stats(
    user_id: str,
    deck_id: str = None,
    min_studied: int = 0,
    date_start: str = '2008-06-01 08:00:00+00:00',
    date_end: str = '2038-06-01 08:00:00+00:00',
    env: str = None,
) -> UserStatsSchema:
    return _get_user_stats(
        user_id=user_id,
        min_studied=min_studied,
        deck_id=deck_id,
        date_start=date_start,
        date_end=date_end,
        env=env,
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
    date_start: str = '2008-06-01 08:00:00+00:00',
    date_end: str = '2038-06-01 08:00:00+00:00',
) -> Leaderboard:
    env = 'dev' if env == 'dev' else 'prod'
    session = get_session(env)

    if False:
        executor = ProcessPoolExecutor()
        stats = {}
        for user in session.query(User):
            if not user.id.isdigit():
                continue

            stats[user.id] = executor.submit(
                _get_user_stats,
                user_id=user.id,
                deck_id=deck_id,
                min_studied=min_studied,
                date_start=date_start,
                date_end=date_end,
            )

        for user_id, future in stats.items():
            stats[user_id] = future.result()
    else:
        stats = {}
        for user in session.query(User):
            if not user.id.isdigit():
                continue

            stats[user.id] = _get_user_stats(
                user_id=user.id,
                deck_id=deck_id,
                min_studied=min_studied,
                date_start=date_start,
                date_end=date_end,
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
