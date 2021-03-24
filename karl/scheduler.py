#!/usr/bin/env python
# coding: utf-8

import json
import pytz
import requests
import numpy as np
import multiprocessing
from typing import List, Dict
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.orm import Session

from karl.schemas import ScheduleRequestSchema, UpdateRequestSchema, ScheduleResponseSchema, ParametersSchema
from karl.models import User, Card, Record, Parameters, UserStats,\
    UserCardFeatureVector, UserFeatureVector, CardFeatureVector,\
    CurrUserCardFeatureVector, CurrUserFeatureVector, CurrCardFeatureVector,\
    Leitner, SM2
from karl.retention_hf import vectors_to_features
from karl.db.session import SessionLocal, engine
from karl.config import settings


class KARLScheduler:

    def get_user(self, user_id: str, session: Session) -> User:
        """
        Get user from DB. Create new if user does not exist.

        :param user_id: the `user_id` of the user to load.
        :return: the user.
        """
        user = session.query(User).get(user_id)
        if user is None:
            user = User(id=user_id)
            session.add(user)
            session.commit()

        if user.parameters is None:
            params = Parameters(id=user_id, **(ParametersSchema().__dict__))
            session.add(params)
            session.commit()

        return user

    def get_card(self, request: ScheduleRequestSchema, session: Session) -> Card:
        card = session.query(Card).get(request.fact_id)  # TODO will change to card_id at some point

        if card is not None:
            return card

        card = Card(
            id=request.fact_id,  # TODO will change to card_id at some point
            text=request.text,
            answer=request.answer,
            category=request.category,
            deck_name=request.deck_name,
            deck_id=request.deck_id,
        )

        v_card = CurrCardFeatureVector(
            card_id=card.id,
            n_study_positive=0,
            n_study_negative=0,
            n_study_total=0,
            previous_delta=None,
            previous_study_date=None,
            previous_study_response=None,
        )
        session.add(card)
        session.add(v_card)
        session.commit()

        return card

    def score_user_cards(self, user: User, cards: List[Card], date: datetime, session: Session) -> List[Dict[str, float]]:
        recall_scores, profile = self.score_recall_batch(user, cards, date)
        scores = [
            {
                'recall': recall_scores[i],
                # 'category': self.score_category(user, card),
                'cool_down': self.score_cool_down(user, card, date, session),
                'leitner': self.score_leitner(user, card, date, session),
                'sm2': self.score_sm2(user, card, date, session),
            }
            for i, card in enumerate(cards)
        ]
        return scores, profile

    def schedule(
        self,
        schedule_requests: List[ScheduleRequestSchema],
        date: datetime,
        rationale=False,
        details=False,
    ) -> ScheduleResponseSchema:
        with SessionLocal().begin() as session:
            # get user and cards
            user = self.get_user(schedule_requests[0].user_id, session)
            cards = [self.get_card(request, session) for request in schedule_requests]

            # score cards
            scores, profile = self.score_user_cards(user, cards, date, session)

            # computer total score
            for i, _ in enumerate(scores):
                scores[i]['sum'] = sum([
                    user.parameters.__dict__.get(key, 0) * value
                    for key, value in scores[i].items()
                ])

            # sort cards
            order = np.argsort([s['sum'] for s in scores]).tolist()
            card_selected = cards[order[0]]

            # determine if is new card
            if session.query(Record).\
                    filter(Record.user_id == user.id).\
                    filter(Record.card_id == card_selected.id).count() > 0:
                is_new_fact = False
            else:
                is_new_fact = True

            # generate record.id (debug_id)
            record_id = json.dumps({
                'user_id': user.id,
                'card_id': card_selected.id,
                'date': str(date.replace(tzinfo=pytz.UTC)),
            })

            # store record with front_end_id empty @ debug_id
            record = Record(
                id=record_id,
                user_id=user.id,
                card_id=card_selected.id,
                deck_id=card_selected.deck_id,
                is_new_fact=is_new_fact,
                date=date,
            )
            session.add(record)
            session.commit()
            session.close()

            # store feature vectors @ debug_id
            self.save_feature_vectors(record.id, user.id, card_selected.id, date)

        rationale = self.get_rationale(
            record_id=record_id,
            user=user,
            cards=cards,
            date=date,
            scores=scores,
            order=order,
        )

        # return
        return ScheduleResponseSchema(
            order=order,
            debug_id=record_id,
            scores=scores,
            rationale=rationale,
            profile=profile,
        )

    def get_curr_user_vector(self, user_id: str) -> CurrUserFeatureVector:
        session = SessionLocal()
        v_user = session.query(CurrUserFeatureVector).get(user_id)
        params = ParametersSchema(**self.get_user(user_id, session).parameters.__dict__)
        if v_user is None:
            v_user = CurrUserFeatureVector(
                user_id=user_id,
                n_study_positive=0,
                n_study_negative=0,
                n_study_total=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
                parameters=json.dumps(params.__dict__),
            )
            session.add(v_user)
        session.commit()
        session.close()
        return v_user

    def get_curr_card_vector(self, card_id: str) -> CurrCardFeatureVector:
        session = SessionLocal()
        v_card = session.query(CurrCardFeatureVector).get(card_id)
        if v_card is None:
            v_card = CurrCardFeatureVector(
                card_id=card_id,
                n_study_positive=0,
                n_study_negative=0,
                n_study_total=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
            )
            session.add(v_card)
        session.commit()
        session.close()
        return v_card

    def get_curr_usercard_vector(self, user_id: str, card_id: str) -> CurrUserCardFeatureVector:
        session = SessionLocal()
        v_usercard = session.query(CurrUserCardFeatureVector).get((user_id, card_id))
        if v_usercard is None:
            v_usercard = CurrUserCardFeatureVector(
                user_id=user_id,
                card_id=card_id,
                n_study_positive=0,
                n_study_negative=0,
                n_study_total=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
                correct_on_first_try=None,
                leitner_box=None,
                leitner_scheduled_date=None,
                sm2_efactor=None,
                sm2_interval=None,
                sm2_repetition=None,
                sm2_scheduled_date=None,
            )
            session.add(v_usercard)
        session.commit()
        session.close()
        return v_usercard

    def score_recall_batch(
        self,
        user: User,
        cards: List[Card],
        date: datetime,
    ) -> List[float]:

        t0 = datetime.now(pytz.utc)

        # gather card features
        feature_vectors = []
        v_user = self.get_curr_user_vector(user.id)

        if not settings.USE_MULTIPROCESSING:
            for card in cards:
                v_card = self.get_curr_card_vector(card.id)
                v_usercard = self.get_curr_usercard_vector(user.id, card.id)
                feature_vectors.append(vectors_to_features(v_usercard, v_user, v_card, date, card.text))
        else:
            executor = ProcessPoolExecutor(
                mp_context=multiprocessing.get_context(settings.MP_CONTEXT),
                initializer=engine.dispose,
            )
            v_card_futures, v_usercard_futures = [], []
            for card in cards:
                v_card_futures.append(executor.submit(self.get_curr_card_vector, card.id))
                v_usercard_futures.append(executor.submit(self.get_curr_usercard_vector, user.id, card.id))
            for card, v_card_future, v_usercard_future in zip(cards, v_card_futures, v_usercard_futures):
                feature_vectors.append(
                    vectors_to_features(v_usercard_future.result(), v_user, v_card_future.result(), date, card.text)
                )

        t1 = datetime.now(pytz.utc)

        feature_vectors = [x.__dict__ for x in feature_vectors]
        for x in feature_vectors:
            x['utc_date'] = str(x['utc_date'])

        scores = json.loads(
            requests.get(
                'http://127.0.0.1:8001/api/karl/predict',
                data=json.dumps(feature_vectors)
            ).text
        )

        t2 = datetime.now(pytz.utc)

        profile = {
            'schedule gather features': (t1 - t0).total_seconds(),
            'schedule model prediction': (t2 - t1).total_seconds(),
        }
        return scores, profile

    def score_category(self, user: User, card: Card) -> float:
        """
        Penalize shift in predefined categories.
        1 if card category is different than the
        previous card the user studied, 0 otherwise.

        :param user:
        :param card:
        :return: 0 if same category, 1 if otherwise.
        """
        if len(user.records) == 0:
            return 0

        prev_card = user.records[-1].card
        if prev_card is None or prev_card.category is None:
            return 0

        return float(card.category != prev_card.category)

    def score_cool_down(self, user: User, card: Card, date: datetime, session: Session) -> float:
        """
        Avoid repetition of the same card within a cool down time.
        We set a longer cool down time for correctly recalled cards.

        :param user:
        :param card:
        :param date: current study date.
        :return: portion of cool down period remained. 0 if passed.
        """
        v_usercard = session.query(CurrUserCardFeatureVector).get((user.id, card.id))

        if v_usercard is None or v_usercard.previous_study_date is None:
            return 0
        else:
            if v_usercard.previous_study_response:
                cool_down_period = user.parameters.cool_down_time_correct
            else:
                cool_down_period = user.parameters.cool_down_time_wrong

            delta_minutes = (date - v_usercard.previous_study_date).total_seconds() // 60
            if delta_minutes > cool_down_period:
                return 0
            else:
                return float(delta_minutes / cool_down_period)

    def score_leitner(self, user: User, card: Card, date: datetime, session: Session) -> float:
        """
        Time till the scheduled date by Leitner measured by number of hours.
        The value can be negative when the card is over-due in Leitner.

        :param user:
        :param card:
        :return: distance in number of hours.
        """
        leitner = session.query(Leitner).get((user.id, card.id))

        if leitner is None or leitner.scheduled_date is None:
            return 0
        else:
            # NOTE distance in hours, can be negative
            return (leitner.scheduled_date - date).total_seconds() / (60 * 60)

    def score_sm2(self, user: User, card: Card, date: datetime, session: Session) -> float:
        """
        Time till the scheduled date by SM-2 measured by number of hours.
        The value can be negative when the card is over-due in SM-2.

        :param user:
        :param card:
        :return: distance in number of hours.
        """
        sm2 = session.query(SM2).get((user.id, card.id))

        if sm2 is None or sm2.scheduled_date is None:
            return 0
        else:
            # NOTE distance in hours, can be negative
            return (sm2.scheduled_date - date).total_seconds() / (60 * 60)

    def save_feature_vectors(
        self,
        record_id: str,
        user_id: str,
        card_id: str,
        date: datetime,
    ) -> None:
        v_user = self.get_curr_user_vector(user_id)
        v_card = self.get_curr_card_vector(card_id)
        v_usercard = self.get_curr_usercard_vector(user_id, card_id)

        session = SessionLocal()

        v_user = session.query(CurrUserFeatureVector).get(user_id)
        v_card = session.query(CurrCardFeatureVector).get(card_id)
        v_usercard = session.query(CurrUserCardFeatureVector).get((user_id, card_id))

        delta_usercard = None
        if v_usercard.previous_study_date is not None:
            delta_usercard = (date - v_usercard.previous_study_date).total_seconds()
        leitner = session.query(Leitner).get((user_id, card_id))
        sm2 = session.query(SM2).get((user_id, card_id))
        session.add(
            UserCardFeatureVector(
                id=record_id,
                user_id=user_id,
                card_id=card_id,
                date=date,
                n_study_positive=v_usercard.n_study_positive,
                n_study_negative=v_usercard.n_study_negative,
                n_study_total=v_usercard.n_study_total,
                delta=delta_usercard,
                previous_delta=v_usercard.previous_delta,
                previous_study_date=v_usercard.previous_study_date,
                previous_study_response=v_usercard.previous_study_response,
                leitner_box=None if leitner is None else leitner.box,
                leitner_scheduled_date=None if leitner is None else leitner.scheduled_date,
                sm2_efactor=None if sm2 is None else sm2.efactor,
                sm2_interval=None if sm2 is None else sm2.interval,
                sm2_repetition=None if sm2 is None else sm2.repetition,
                sm2_scheduled_date=None if sm2 is None else sm2.scheduled_date,
                correct_on_first_try=v_usercard.correct_on_first_try,
            ))

        delta_user = None
        if v_user.previous_study_date is not None:
            delta_user = (date - v_user.previous_study_date).total_seconds()
        params = ParametersSchema(**self.get_user(user_id, session).parameters.__dict__)
        session.add(
            UserFeatureVector(
                id=record_id,
                user_id=user_id,
                date=date,
                n_study_positive=v_user.n_study_positive,
                n_study_negative=v_user.n_study_negative,
                n_study_total=v_user.n_study_total,
                delta=delta_user,
                previous_delta=v_user.previous_delta,
                previous_study_date=v_user.previous_study_date,
                previous_study_response=v_user.previous_study_response,
                parameters=json.dumps(params.__dict__),
            ))

        delta_card = None
        if v_card.previous_study_date is not None:
            delta_card = (date - v_card.previous_study_date).total_seconds()
        session.add(
            CardFeatureVector(
                id=record_id,
                card_id=card_id,
                date=date,
                n_study_positive=v_card.n_study_positive,
                n_study_negative=v_card.n_study_negative,
                n_study_total=v_card.n_study_total,
                delta=delta_card,
                previous_delta=v_card.previous_delta,
                previous_study_date=v_card.previous_study_date,
                previous_study_response=v_card.previous_study_response,
            ))
        session.commit()
        session.close()

    def update(self, request: UpdateRequestSchema, date: datetime) -> dict:
        with SessionLocal().begin() as session:
            t0 = datetime.now(pytz.utc)

            # read debug_id from request, find corresponding record
            record = session.query(Record).get(request.debug_id)

            if record is None:
                return {}

            # add front_end_id to record, response, elapsed_times to record
            record.date = date
            record.front_end_id = request.history_id
            record.response = request.label
            record.elapsed_milliseconds_text = request.elapsed_milliseconds_text
            record.elapsed_milliseconds_answer = request.elapsed_milliseconds_answer
            t1 = datetime.now(pytz.utc)
            session.commit()

            # update leitner
            self.update_leitner(record, date, session)
            t2 = datetime.now(pytz.utc)

            # update sm2
            self.update_sm2(record, date)
            t3 = datetime.now(pytz.utc)

            # update user stats
            utc_date = date.astimezone(pytz.utc).date()
            self.update_user_stats(record, deck_id='all', utc_date=utc_date, session=session)
            t4 = datetime.now(pytz.utc)
            if record.deck_id is not None:
                self.update_user_stats(record, deck_id=record.deck_id, utc_date=utc_date, session=session)
            t5 = datetime.now(pytz.utc)

            # update current features
            # NOTE do this last, especially after leitner and sm2
            self.update_feature_vectors(record, date)
            t6 = datetime.now(pytz.utc)
            session.commit()

        return {
            'update record': (t1 - t0).total_seconds(),
            'update leitner': (t2 - t1).total_seconds(),
            'update sm2': (t3 - t2).total_seconds(),
            'update user_stats all': (t4 - t3).total_seconds(),
            'update user_stats deck': (t5 - t4).total_seconds(),
            'update feature vectors': (t6 - t5).total_seconds(),
        }

    def update_feature_vectors(self, record: Record, date: datetime, session: Session):
        v_user = self.get_curr_user_vector(record.user_id)
        v_card = self.get_curr_card_vector(record.card_id)
        v_usercard = self.get_curr_usercard_vector(record.user_id, record.card_id)

        v_user = session.query(CurrUserFeatureVector).get(record.user_id)
        v_card = session.query(CurrCardFeatureVector).get(record.card_id)
        v_usercard = session.query(CurrUserCardFeatureVector).get((record.user_id, record.card_id))

        delta_usercard = None
        if v_usercard.previous_study_date is not None:
            delta_usercard = (date - v_usercard.previous_study_date).total_seconds()
        v_usercard.n_study_positive += record.response
        v_usercard.n_study_negative += (not record.response)
        v_usercard.n_study_total += 1
        v_usercard.previous_delta = delta_usercard
        v_usercard.previous_study_date = date
        v_usercard.previous_study_response = record.response
        if v_usercard.correct_on_first_try is None:
            v_usercard.correct_on_first_try = record.response
            # NOTE the correponding `UserCardFeatureVector` saved in # `save_feature_vectors`
            # from the scheduling request also has None in its `correct_on_first_try`.
            # this is fine, we leave it as None, since that saved feature vector is what was
            # visible to the scheduler before the response was sent by the user.
        leitner = session.query(Leitner).get((v_usercard.user_id, v_usercard.card_id))
        sm2 = session.query(SM2).get((v_usercard.user_id, v_usercard.card_id))
        if leitner is not None:
            v_usercard.leitner_box = leitner.box
            v_usercard.leitner_scheduled_date = leitner.scheduled_date
        else:
            pass
            # print('leitner is none')
        if sm2 is not None:
            v_usercard.sm2_efactor = sm2.efactor
            v_usercard.sm2_interval = sm2.interval
            v_usercard.sm2_repetition = sm2.repetition
            v_usercard.sm2_scheduled_date = sm2.scheduled_date
        else:
            pass
            # print('sm2 is none')

        delta_user = None
        if v_user.previous_study_date is not None:
            delta_user = (date - v_user.previous_study_date).total_seconds()
        v_user.n_study_positive += record.response
        v_user.n_study_negative += (not record.response)
        v_user.n_study_total += 1
        v_user.previous_delta = delta_user
        v_user.previous_study_date = date
        v_user.previous_study_response = record.response

        delta_card = None
        if v_card.previous_study_date is not None:
            delta_card = (date - v_card.previous_study_date).total_seconds()
        v_card.n_study_positive += record.response
        v_card.n_study_negative += (not record.response)
        v_card.n_study_total += 1
        v_card.previous_delta = delta_card
        v_card.previous_study_date = date
        v_card.previous_study_response = record.response

    def update_user_stats(
        self,
        record: Record,
        utc_date,
        deck_id: str,
        session: Session,
    ):
        # get the latest user_stat ordered by date
        curr_stats = session.query(UserStats).\
            filter(UserStats.user_id == record.user_id).\
            filter(UserStats.deck_id == deck_id).\
            order_by(UserStats.date.desc()).first()

        is_new_stat = False
        if curr_stats is None:
            stats_id = json.dumps({
                'user_id': record.user_id,
                'deck_id': deck_id,
                'date': str(utc_date),
            })
            curr_stats = UserStats(
                id=stats_id,
                user_id=record.user_id,
                deck_id=deck_id,
                date=utc_date,
                n_cards_total=0,
                n_cards_positive=0,
                n_new_cards_total=0,
                n_old_cards_total=0,
                n_new_cards_positive=0,
                n_old_cards_positive=0,
                elapsed_milliseconds_text=0,
                elapsed_milliseconds_answer=0,
                n_days_studied=0,
            )
            is_new_stat = True

        if utc_date != curr_stats.date:
            # there is a previous user_stat, but not from today
            # copy user stat to today
            stats_id = json.dumps({
                'user_id': record.user_id,
                'deck_id': deck_id,
                'date': str(utc_date),
            })
            new_stat = UserStats(
                id=stats_id,
                user_id=record.user_id,
                deck_id=deck_id,
                date=utc_date,
                n_cards_total=curr_stats.n_cards_total,
                n_cards_positive=curr_stats.n_cards_positive,
                n_new_cards_total=curr_stats.n_new_cards_total,
                n_old_cards_total=curr_stats.n_old_cards_total,
                n_new_cards_positive=curr_stats.n_new_cards_positive,
                n_old_cards_positive=curr_stats.n_old_cards_positive,
                elapsed_milliseconds_text=curr_stats.elapsed_milliseconds_text,
                elapsed_milliseconds_answer=curr_stats.elapsed_milliseconds_answer,
                n_days_studied=curr_stats.n_days_studied + 1,
            )
            curr_stats = new_stat
            is_new_stat = True

        if record.is_new_fact:
            curr_stats.n_new_cards_total += 1
            curr_stats.n_new_cards_positive += record.response
        else:
            curr_stats.n_old_cards_total += 1
            curr_stats.n_old_cards_positive += record.response

        curr_stats.n_cards_total += 1
        curr_stats.n_cards_positive += record.response
        curr_stats.elapsed_milliseconds_text += record.elapsed_milliseconds_text
        curr_stats.elapsed_milliseconds_answer += record.elapsed_milliseconds_answer

        if is_new_stat:
            session.add(curr_stats)

    def update_leitner(self, record: Record, date: datetime, session: Session) -> None:
        # leitner boxes 1~10
        # days[0] = None as placeholder since we don't have box 0
        # days[9] and days[10] = 999 to make it never repeat
        days = [0, 0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 999, 999]
        increment_days = {i: x for i, x in enumerate(days)}

        leitner = session.query(Leitner).get((record.user_id, record.card_id))
        if leitner is None:
            # boxes: 1 ~ 10
            leitner = Leitner(user_id=record.user_id, card_id=record.card_id, box=1)
            session.add(leitner)

        leitner.box += (1 if record.response else -1)
        leitner.box = max(min(leitner.box, 10), 1)
        interval = timedelta(days=increment_days[leitner.box])
        leitner.scheduled_date = date + interval

    def update_sm2(self, record: Record, date: datetime, session: Session) -> None:
        def get_quality_from_response(response: bool) -> int:
            return 4 if response else 1

        sm2 = session.query(SM2).get((record.user_id, record.card_id))
        if sm2 is None:
            sm2 = SM2(
                user_id=record.user_id,
                card_id=record.card_id,
                efactor=2.5,
                interval=1,
                repetition=0,
            )
            session.add(sm2)

        q = get_quality_from_response(record.response)
        sm2.repetition += 1
        sm2.efactor = max(1.3, sm2.efactor + 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02))

        if not record.response:
            sm2.interval = 0
            sm2.repetition = 0
        else:
            if sm2.repetition == 1:
                sm2.interval = 1
            elif sm2.repetition == 2:
                sm2.interval = 6
            else:
                sm2.interval *= sm2.efactor

        sm2.interval = max(500, sm2.interval)  # TODO prevent date overflow
        try:
            sm2.scheduled_date = date + timedelta(days=sm2.interval)
        except OverflowError:
            pass

    def get_rationale(
        self,
        record_id: str,
        user: User,
        cards: List[Card],
        date: datetime,
        scores: List[Dict[str, float]],
        order: List[int],
        top_n_cards: int = 3,
    ) -> str:
        rr = """
             <style>
             table {
               border-collapse: collapse;
             }
             td, th {
               padding: 0.5rem;
               text-align: left;
             }
             tr:nth-child(even) {background-color: #f2f2f2;}
             </style>
             """

        rr += '<p>Debug ID: {}</p>'.format(record_id)
        rr += '<p>Model: {}</p>'.format(user.parameters.repetition_model)
        row_template = '<tr><td><b>{}</b></td> <td>{}</td></tr>'
        row_template_3 = '<tr><td><b>{}</b></td> <td>{:.4f} x {:.2f}</td></tr>'
        for i in order[:top_n_cards]:
            card = cards[i]
            v_usercard = self.get_curr_usercard_vector(user.id, card.id)
            rr += '<table style="float: left;">'
            rr += row_template.format('Card ID', card.id)
            rr += row_template.format('Answer', card.answer)
            rr += row_template.format('Category', card.category)
            rr += row_template.format('Category', card.category)

            params = user.parameters.__dict__
            if 'karl' in params['repetition_model']:
                rr += row_template.format('Recall target', params['recall_target'])
            for key, val in scores[i].items():
                if key in params:
                    rr += row_template_3.format('Score: ' + key, val, params[key])
            rr += '<tr><td><b>{}</b></td> <td>{:.4f}</td></tr>'.format('Score: total', scores[i]['sum'])

            if v_usercard.leitner_scheduled_date is not None:
                delta_to_leitner_scheduled_date = (v_usercard.leitner_scheduled_date - date).total_seconds() // 3600
            else:
                delta_to_leitner_scheduled_date = None

            if v_usercard.sm2_scheduled_date is not None:
                delta_to_sm2_scheduled_date = (v_usercard.sm2_scheduled_date - date).total_seconds() // 3600
            else:
                delta_to_sm2_scheduled_date = None

            if v_usercard.previous_study_date is not None:
                usercard_delta = (date - v_usercard.previous_study_date).total_seconds() // 3600
            else:
                usercard_delta = None

            if v_usercard.leitner_scheduled_date is not None:
                leitner_scheduled_date = v_usercard.leitner_scheduled_date.strftime("%Y/%m/%d %H:%M:%S")
            else:
                leitner_scheduled_date = None

            if v_usercard.sm2_scheduled_date is not None:
                sm2_scheduled_date = v_usercard.sm2_scheduled_date.strftime("%Y/%m/%d %H:%M:%S")
            else:
                sm2_scheduled_date = None

            rr += row_template.format('#Correct', v_usercard.n_study_positive)
            rr += row_template.format('#Wrong', v_usercard.n_study_negative)
            rr += row_template.format('Previous delta', v_usercard.previous_delta)
            rr += row_template.format('Previous date', v_usercard.previous_study_date)
            rr += row_template.format('Previous result', v_usercard.previous_study_response)
            rr += row_template.format('Current date', date.strftime("%Y/%m/%d %H:%M:%S"))
            rr += row_template.format('Hours since prev', usercard_delta)
            rr += row_template.format('Correct on first try', v_usercard.correct_on_first_try)
            rr += row_template.format('Leitner box', v_usercard.leitner_box)
            rr += row_template.format('Leitner schedule', leitner_scheduled_date)
            rr += row_template.format('Hours to Leitner', delta_to_leitner_scheduled_date)
            rr += row_template.format('SM2 e-factor', v_usercard.sm2_efactor)
            rr += row_template.format('SM2 interval', v_usercard.sm2_interval)
            rr += row_template.format('SM2 repetition', v_usercard.sm2_repetition)
            rr += row_template.format('SM2 schedule', sm2_scheduled_date)
            rr += row_template.format('Hours to SM2', delta_to_sm2_scheduled_date)

            rr += '</table>'
        return rr
