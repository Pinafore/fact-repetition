#!/usr/bin/env python
# coding: utf-8

import json
import pytz
import logging
import atexit
import multiprocessing
import pandas as pd
from typing import Generator, List
from datetime import datetime
from fastapi import FastAPI, Depends
from dateutil.parser import parse as parse_date
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.orm import Session
# from cachetools import cached, TTLCache

from karl.schemas import UserStatsSchema, RankingSchema, LeaderboardSchema, \
    OldParametersSchema, ParametersSchema, \
    ScheduleResponseSchema, ScheduleRequestSchema, UpdateRequestSchema, \
    Visualization
from karl.models import User, UserStats, Parameters
from karl.scheduler import KARLScheduler
from karl.db.session import SessionLocal, engine
from karl.config import settings
from karl import metrics


app = FastAPI()
scheduler = KARLScheduler()
user_progress_df = pd.read_hdf(f'{settings.CODE_DIR}/user_processed_df.h5', 'df')

# create logger with 'scheduler'
logger = logging.getLogger('scheduler')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f'{settings.CODE_DIR}/scheduler.log')
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


def conditional_decorator(dec, condition, **kwargs):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func(**kwargs)
        return dec(func)
    return decorator


def get_session() -> Generator:
    try:
        session = SessionLocal()
        yield session
    finally:
        session.close()


@app.put('/api/karl/set_params', response_model=ParametersSchema)
def set_params(
    # env: str,
    user_id: str,
    params: OldParametersSchema,
    session: Session = Depends(get_session),
) -> ParametersSchema:
    curr_params = session.query(Parameters).get(user_id)
    is_new_params = False
    if curr_params is None:
        is_new_params = True
        curr_params = Parameters(id=user_id, **ParametersSchema().__dict__)

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

    session.commit()
    return ParametersSchema(**curr_params.__dict__)


@app.get('/api/karl/get_params', response_model=ParametersSchema)
def get_params(
    # env: str,
    user_id: str,
    session: Session = Depends(get_session),
) -> ParametersSchema:
    curr_params = session.query(Parameters).get(user_id)
    if curr_params is None:
        curr_params = Parameters(id=user_id)
    return ParametersSchema(**curr_params.__dict__)


@app.put('/api/karl/set_repetition_model', response_model=ParametersSchema)
def set_repetition_model(
    # env: str,
    user_id: str,
    repetition_model: str,
    session: Session = Depends(get_session),
):
    if repetition_model == 'sm2':
        params = ParametersSchema(
            repetition_model='sm2',
            card_embedding=0,
            recall=0,
            recall_target=0,
            category=0,
            answer=0,
            leitner=0,
            sm2=1,
            decay_qrep=0,
            cool_down=0,
            cool_down_time_correct=0,
            cool_down_time_wrong=0,
            max_recent_facts=0,
        )
    elif repetition_model == 'leitner':
        params = ParametersSchema(
            repetition_model='leitner',
            card_embedding=0,
            recall=0,
            recall_target=0,
            category=0,
            answer=0,
            leitner=1,
            sm2=0,
            decay_qrep=0,
            cool_down=0,
            cool_down_time_correct=0,
            cool_down_time_wrong=0,
            max_recent_facts=0,
        )
    elif repetition_model.startswith('karl'):
        recall_target = int(repetition_model[4:])
        params = ParametersSchema(
            repetition_model=f'karl{recall_target}',
            recall_target=float(recall_target) / 100,
        )

    curr_params = session.query(Parameters).get(user_id)
    is_new_params = False
    if curr_params is None:
        is_new_params = True
        curr_params = Parameters(id=user_id, **params)

    curr_params.repetition_model = params.repetition_model
    curr_params.card_embedding = params.card_embedding
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

    session.commit()
    return params


@app.get('/api/karl/get_user_stats', response_model=UserStatsSchema)
def get_user_stats(
    # env: str,
    user_id: str,
    deck_id: str = None,
    min_studied: int = 0,
    date_start: str = '2008-06-01 08:00:00+00:00',
    date_end: str = '2038-06-01 08:00:00+00:00',
) -> UserStatsSchema:
    # print('user_stats starts at', time.strftime('%X'), f'tid={threading.get_ident()}', f'pid={getpid()}')
    stats = _get_user_stats(
        user_id,
        deck_id,
        min_studied,
        date_start,
        date_end,
        True,
    )
    # print('user_stats finishes at', time.strftime('%X'), f'tid={threading.get_ident()}', f'pid={getpid()}')
    return stats


def _get_user_stats(
    # env: str,
    user_id: str,
    deck_id: str = None,
    min_studied: int = 0,
    date_start: str = '2008-06-01 08:00:00+00:00',
    date_end: str = '2038-06-01 08:00:00+00:00',
    from_user_stats: bool = False
) -> UserStatsSchema:
    session = SessionLocal()

    # print('_user_stats', from_user_stats, f'tid={threading.get_ident()}', f'pid={getpid()}', session.hash_key)

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


@app.get('/api/karl/leaderboard', response_model=LeaderboardSchema)
# @cached(cache=TTLCache(maxsize=1024, ttl=600))
def get_leaderboard(
    # env: str,
    user_id: str = None,
    skip: int = 0,
    limit: int = 10,
    rank_type: str = 'total_seen',
    min_studied: int = 0,
    deck_id: str = None,
    date_start: str = '2008-06-01 08:00:00+00:00',
    date_end: str = '2038-06-01 08:00:00+00:00',
) -> LeaderboardSchema:
    session = SessionLocal()
    # print('leaderboard starts at', time.strftime('%X'), f'tid={threading.get_ident()}', f'pid={getpid()}', session.hash_key)

    stats = {}
    if not settings.USE_MULTIPROCESSING:
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
    else:
        executor = ProcessPoolExecutor(
            mp_context=multiprocessing.get_context(settings.MP_CONTEXT),
            initializer=engine.dispose,
        )
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

    # from high value to low
    stats = sorted(stats.items(), key=lambda x: x[1].__dict__[rank_type])[::-1]
    stats = [(k, v) for k, v in stats if v.total_seen >= min_studied]

    rankings = []
    user_place = None
    for i, (k, v) in enumerate(stats):
        if user_id == k:
            user_place = i
        rankings.append(RankingSchema(
            user_id=k,
            rank=i + 1,
            value=v.__dict__[rank_type]
        ))
    session.close()

    return LeaderboardSchema(
        leaderboard=rankings[skip: skip + limit],
        total=len(rankings),
        rank_type=rank_type,
        user_place=user_place,
        user_id=user_id,
        skip=skip,
        limit=limit,
    )


@app.post('/api/karl/schedule')
def schedule(
    # env: str,
    schedule_requests: List[ScheduleRequestSchema],
    session: Session = Depends(get_session),
) -> ScheduleResponseSchema:
    # print('schedule starts at', time.strftime('%X'), f'tid={threading.get_ident()}', f'pid={getpid()}')

    if schedule_requests[0].date is not None:
        date = parse_date(schedule_requests[0].date)
    else:
        date = datetime.now(pytz.utc)
    schedule_response = scheduler.schedule(session, schedule_requests, date)
    session.commit()
    return schedule_response


@app.post('/api/karl/update')
def update(
    # env: str,
    update_requests: List[UpdateRequestSchema],
    session: Session = Depends(get_session),
) -> None:
    update_request = update_requests[0]  # only accept one request. for backward compatability
    if update_request.date is not None:
        date = parse_date(update_request.date)
    else:
        date = datetime.now(pytz.utc)
    scheduler.update(session, update_request, date)
    session.commit()
    return


@app.get('/api/karl/get_user_charts')
def get_user_charts(
    # env: str = None,
    user_id: str = None,
    deck_id: str = None,
    date_start: str = '2008-06-01 08:00:00+00:00',
    date_end: str = '2038-06-01 08:00:00+00:00',
    session: Session = Depends(get_session),
) -> List[Visualization]:

    charts = metrics.get_user_charts(
        df=user_progress_df,
        user_id=user_id,
        deck_id=deck_id,
        # date_start=date_start,
        # date_end=date_end,
    )

    visualizations = []
    for chart_name, chart in charts.items():
        visualizations.append(
            Visualization(
                name=chart_name,
                specs=chart.to_json(),
                user_id=user_id,
                deck_id=deck_id,
                date_start=date_start,
                date_end=date_end,
            )
        )
    return visualizations


@atexit.register
def dispose():
    engine.dispose()
    return


@app.get('/api/karl/status')
def status():
    return True
