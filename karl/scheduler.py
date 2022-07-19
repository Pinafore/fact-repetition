#!/usr/bin/env python
# coding: utf-8

import json
import pytz
import requests
import numpy as np
import multiprocessing
from typing import List, Dict, Union
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.orm import Session

from karl.schemas import ScheduleRequestSchema, UpdateRequestSchema, ScheduleResponseSchema,\
    ScheduleRequestV2, UpdateRequestV2, KarlFactV2
from karl.schemas import ParametersSchema
from karl.schemas import VUser, VCard, VUserCard
from karl.models import User, Card, Parameters, UserStatsV2,\
    CurrUserCardFeatureVector, CurrUserFeatureVector, CurrCardFeatureVector,\
    SimUserCardFeatureVector, SimUserFeatureVector, SimCardFeatureVector,\
    UserCardSnapshot, UserSnapshot, CardSnapshot,\
    StudyRecord, TestRecord, ScheduleRequest

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

    def get_card(
        self,
        request: Union[ScheduleRequestSchema, KarlFactV2],
        session: Session,
    ) -> Card:
        card = session.query(Card).get(request.fact_id)

        if card is not None:
            return card

        card = Card(
            id=request.fact_id,
            text=request.text,
            answer=request.answer,
            category=request.category,
            deck_name=request.deck_name,
            deck_id=request.deck_id,
        )
        session.add(card)
        session.commit()

        v_card = CurrCardFeatureVector(
            card_id=card.id,
            count_positive=0,
            count_negative=0,
            count=0,
            previous_delta=None,
            previous_study_date=None,
            previous_study_response=None,
        )
        session.add(v_card)
        session.commit()

        return card

    def score_user_cards(
        self,
        user: User,
        cards: List[Card],
        date: datetime,
        session: Session,
        simulated: bool = False,
    ) -> List[Dict[str, float]]:
        recall_scores, profile = self.score_recall_batch(user, cards, date, simulated)
        scores = [
            {
                'recall': recall_scores[i],
                'cool_down': self.score_cool_down(user, card, date, session, simulated),
                'leitner': self.score_leitner(user, card, date, session, simulated),
                'sm2': self.score_sm2(user, card, date, session, simulated),
                # 'category': self.score_category(user, card),
            }
            for i, card in enumerate(cards)
        ]
        return scores, profile

    def schedule_v2(
        self,
        schedule_request: ScheduleRequestV2,
        date: datetime,
    ) -> ScheduleResponseSchema:
        session = SessionLocal()
        # get user and cards
        user = self.get_user(schedule_request.user_id, session)
        cards = [self.get_card(fact, session) for fact in schedule_request.facts]

        # schedule_request_id is also debug_id
        schedule_request_id = json.dumps({
            'user_id': user.id,
            'repetition_model': schedule_request.repetition_model,
            'date': str(date.replace(tzinfo=pytz.UTC)),
        })

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

        # store schedule request
        session.add(
            ScheduleRequest(
                id=schedule_request_id,
                user_id=user.id,
                card_ids=[x.id for x in cards],
                repetition_model=schedule_request.repetition_model,
                date=date,
            )
        )
        session.commit()

        self.save_snapshots(schedule_request_id, user.id, card_selected.id, date)

        rationale = self.get_rationale(
            schedule_request_id=schedule_request_id,
            user=user,
            cards=cards,
            date=date,
            scores=scores,
            order=order,
        )

        # session.commit()
        session.close()

        # return
        return ScheduleResponseSchema(
            order=order,
            debug_id=schedule_request_id,
            scores=scores,
            rationale=rationale,
            profile=profile,
        )

    def get_curr_user_vector(self, user_id: str) -> VUser:
        session = SessionLocal()
        v_user = session.query(CurrUserFeatureVector).get(user_id)
        params = ParametersSchema(**self.get_user(user_id, session).parameters.__dict__)
        if v_user is None:
            v_user = CurrUserFeatureVector(
                user_id=user_id,
                count_positive=0,
                count_negative=0,
                count=0,
                count_positive_session=0,
                count_negative_session=0,
                count_session=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
                previous_delta_session=None,
                previous_study_date_session=None,
                previous_study_response_session=None,
                schedule_request_id=None,
                parameters=json.dumps(params.__dict__),
            )
            session.add(v_user)
        v_user = VUser(**v_user.__dict__)
        session.commit()
        session.close()
        return v_user

    def get_curr_card_vector(self, card_id: str) -> VCard:
        session = SessionLocal()
        v_card = session.query(CurrCardFeatureVector).get(card_id)
        if v_card is None:
            v_card = CurrCardFeatureVector(
                card_id=card_id,
                count_positive=0,
                count_negative=0,
                count=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
            )
            session.add(v_card)
        v_card = VCard(**v_card.__dict__)
        session.commit()
        session.close()
        return v_card

    def get_curr_usercard_vector(self, user_id: str, card_id: str) -> VUserCard:
        session = SessionLocal()
        v_usercard = session.query(CurrUserCardFeatureVector).get((user_id, card_id))
        if v_usercard is None:
            v_usercard = CurrUserCardFeatureVector(
                user_id=user_id,
                card_id=card_id,
                count_positive=0,
                count_negative=0,
                count=0,
                count_positive_session=0,
                count_negative_session=0,
                count_session=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
                correct_on_first_try=None,
                previous_delta_session=None,
                previous_study_date_session=None,
                previous_study_response_session=None,
                correct_on_first_try_session=None,
                leitner_box=None,
                leitner_scheduled_date=None,
                sm2_efactor=None,
                sm2_interval=None,
                sm2_repetition=None,
                sm2_scheduled_date=None,
                schedule_request_id=None,
            )
            session.add(v_usercard)
        v_usercard = VUserCard(**v_usercard.__dict__)
        session.commit()
        session.close()
        return v_usercard

    def get_simulated_user_vector(self, user_id: str) -> VUser:
        session = SessionLocal()
        v_user = session.query(SimUserFeatureVector).get(user_id)
        params = ParametersSchema(**self.get_user(user_id, session).parameters.__dict__)
        if v_user is None:
            v_user = SimUserFeatureVector(
                user_id=user_id,
                count_positive=0,
                count_negative=0,
                count=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
                parameters=json.dumps(params.__dict__),
            )
            session.add(v_user)
        v_user = VUser(**v_user.__dict__)
        session.commit()
        session.close()
        return v_user

    def get_simulated_card_vector(self, card_id: str) -> VCard:
        session = SessionLocal()
        v_card = session.query(SimCardFeatureVector).get(card_id)
        if v_card is None:
            v_card = SimCardFeatureVector(
                card_id=card_id,
                count_positive=0,
                count_negative=0,
                count=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
            )
            session.add(v_card)
        v_card = VCard(**v_card.__dict__)
        session.commit()
        session.close()
        return v_card

    def get_simulated_usercard_vector(self, user_id: str, card_id: str) -> VUserCard:
        session = SessionLocal()
        v_usercard = session.query(SimUserCardFeatureVector).get((user_id, card_id))
        if v_usercard is None:
            print('v_usercard is none')
            v_usercard = SimUserCardFeatureVector(
                user_id=user_id,
                card_id=card_id,
                count_positive=0,
                count_negative=0,
                count=0,
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
        v_usercard = VUserCard(**v_usercard.__dict__)
        session.commit()
        session.close()
        return v_usercard

    def get_latest_user_vector(self, user_id: str, date: datetime) -> VUser:
        session = SessionLocal()
        v_user = session.query(UserFeatureVector).\
            filter(UserFeatureVector.id == user_id).\
            filter(UserFeatureVector.date <= date).\
            order_by(UserFeatureVector.date.desc()).first()
        params = ParametersSchema(**self.get_user(user_id, session).parameters.__dict__)
        if v_user is None:
            v_user = VUser(
                user_id=user_id,
                count_positive=0,
                count_negative=0,
                count=0,
                parameters=json.dumps(params.__dict__),
            )
        else:
            v_user = VUser(**v_user.__dict__)
        session.commit()
        session.close()
        return v_user

    def get_latest_card_vector(self, card_id: str, date: datetime) -> VCard:
        session = SessionLocal()
        v_card = session.query(CardFeatureVector).\
            filter(CardFeatureVector.id == card_id).\
            filter(CardFeatureVector.date <= date).\
            order_by(CardFeatureVector.date.desc()).first()
        if v_card is None:
            v_card = VCard(
                card_id=card_id,
                count_positive=0,
                count_negative=0,
                count=0,
            )
        else:
            v_card = VCard(**v_card.__dict__)
        session.commit()
        session.close()
        return v_card

    def get_latest_usercard_vector(self, user_id: str, card_id: str, date: datetime) -> VUserCard:
        session = SessionLocal()
        v_usercard = session.query(UserCardFeatureVector).\
            filter(UserCardFeatureVector.user_id == user_id).\
            filter(UserCardFeatureVector.card_id == card_id).\
            filter(UserCardFeatureVector.date <= date).\
            order_by(UserCardFeatureVector.date.desc()).first()
        if v_usercard is None:
            v_usercard = VUserCard(
                user_id=user_id,
                card_id=card_id,
                count_positive=0,
                count_negative=0,
                count=0,
            )
        else:
            v_usercard = VUserCard(**v_usercard.__dict__)
        session.commit()
        session.close()
        return v_usercard

    def collect_features(self, user_id, card_id, card_text, v_user, date):
        '''helper for multiprocessing'''
        v_card = self.get_curr_card_vector(card_id)
        v_usercard = self.get_curr_usercard_vector(user_id, card_id)
        return vectors_to_features(v_usercard, v_user, v_card, date, card_text)

    def score_recall_batch(
        self,
        user: User,
        cards: List[Card],
        date: datetime,
        simulated: bool = False,  # simulating old history
    ) -> List[float]:

        t0 = datetime.now(pytz.utc)

        # gather card features
        feature_vectors = []
        if simulated:
            v_user = self.get_simulated_user_vector(user.id)
        else:
            v_user = self.get_curr_user_vector(user.id)

        if not settings.USE_MULTIPROCESSING:
            feature_vectors = [
                self.collect_features(user.id, card.id, card.text, v_user, date).__dict__
                for card in cards
            ]
        else:
            # https://docs.sqlalchemy.org/en/14/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
            # https://pythonspeed.com/articles/python-multiprocessing/
            executor = ProcessPoolExecutor(
                mp_context=multiprocessing.get_context(settings.MP_CONTEXT),
                initializer=engine.dispose,
            )
        futures = [
            executor.submit(self.collect_features, user.id, card.id, card.text, v_user, date)
            for card in cards
        ]
        feature_vectors = [x.result().__dict__ for x in futures]

        t1 = datetime.now(pytz.utc)

        for x in feature_vectors:
            x['utc_date'] = str(x['utc_date'])
            x['utc_datetime'] = str(x['utc_datetime'])

        scores = json.loads(
            requests.get(
                f'{settings.MODEL_API_URL}/api/karl/predict',
                data=json.dumps(feature_vectors)
            ).text
        )

        if 'karl' in user.parameters.repetition_model:
            scores = [abs(user.parameters.recall_target - x) for x in scores]

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

    def score_cool_down(
        self,
        user: User,
        card: Card,
        date: datetime,
        session: Session,
        simulated: bool = False,
    ) -> float:
        """
        Avoid repetition of the same card within a cool down time.
        We set a longer cool down time for correctly recalled cards.
        :return: portion of cool down period remained. 0 if passed.
        """
        if simulated:
            v_usercard = self.get_simulated_usercard_vector(user.id, card.id)
        else:
            v_usercard = self.get_curr_usercard_vector(user.id, card.id)

        if v_usercard is None or v_usercard.previous_study_date is None:
            return 0
        else:
            if v_usercard.previous_study_response:
                cool_down_period = user.parameters.cool_down_time_correct
            else:
                cool_down_period = user.parameters.cool_down_time_wrong

            delta_minutes = (date - v_usercard.previous_study_date).total_seconds() // 60
            return max(cool_down_period - delta_minutes, 0) / cool_down_period

    def score_leitner(
        self,
        user: User,
        card: Card,
        date: datetime,
        session: Session,
        simulated: bool = False,
    ) -> float:
        """
        Time till the scheduled date by Leitner measured by number of hours.
        The value can be negative when the card is over-due in Leitner.
        :return: distance in number of hours.
        """
        if simulated:
            v_usercard = self.get_simulated_usercard_vector(user.id, card.id)
        else:
            v_usercard = self.get_curr_usercard_vector(user.id, card.id)

        if v_usercard.leitner_scheduled_date is None:
            return 0
        else:
            # NOTE distance in days, can be negative
            return (v_usercard.leitner_scheduled_date - date).total_seconds() / 86400

    def score_sm2(
        self,
        user: User,
        card: Card,
        date: datetime,
        session: Session,
        simulated: bool = False,
    ) -> float:
        """
        Time till the scheduled date by SM-2 measured by number of hours.
        The value can be negative when the card is over-due in SM-2.

        :param user:
        :param card:
        :return: distance in number of hours.
        """
        if simulated:
            v_usercard = self.get_simulated_usercard_vector(user.id, card.id)
        else:
            v_usercard = self.get_curr_usercard_vector(user.id, card.id)

        if v_usercard.sm2_scheduled_date is None:
            return 0
        else:
            # NOTE distance in days, can be negative
            return (v_usercard.sm2_scheduled_date - date).total_seconds() / 86400

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

        # v_user = session.query(CurrUserFeatureVector).get(user_id)
        # v_card = session.query(CurrCardFeatureVector).get(card_id)
        # v_usercard = session.query(CurrUserCardFeatureVector).get((user_id, card_id))

        delta_usercard = None
        if v_usercard.previous_study_date is not None:
            delta_usercard = (date - v_usercard.previous_study_date).total_seconds()
        session.add(
            UserCardFeatureVector(
                id=record_id,
                user_id=user_id,
                card_id=card_id,
                date=date,
                count_positive=v_usercard.count_positive,
                count_negative=v_usercard.count_negative,
                count=v_usercard.count,
                delta=delta_usercard,
                previous_delta=v_usercard.previous_delta,
                previous_study_date=v_usercard.previous_study_date,
                previous_study_response=v_usercard.previous_study_response,
                leitner_box=v_usercard.leitner_box,
                leitner_scheduled_date=v_usercard.leitner_scheduled_date,
                sm2_efactor=v_usercard.sm2_efactor,
                sm2_interval=v_usercard.sm2_interval,
                sm2_repetition=v_usercard.sm2_repetition,
                sm2_scheduled_date=v_usercard.sm2_scheduled_date,
                correct_on_first_try=v_usercard.correct_on_first_try,
            ))
        session.commit()

        delta_user = None
        if v_user.previous_study_date is not None:
            delta_user = (date - v_user.previous_study_date).total_seconds()
        params = ParametersSchema(**self.get_user(user_id, session).parameters.__dict__)
        session.add(
            UserFeatureVector(
                id=record_id,
                user_id=user_id,
                date=date,
                count_positive=v_user.count_positive,
                count_negative=v_user.count_negative,
                count=v_user.count,
                delta=delta_user,
                previous_delta=v_user.previous_delta,
                previous_study_date=v_user.previous_study_date,
                previous_study_response=v_user.previous_study_response,
                parameters=json.dumps(params.__dict__),
            ))
        session.commit()

        delta_card = None
        if v_card.previous_study_date is not None:
            delta_card = (date - v_card.previous_study_date).total_seconds()
        session.add(
            CardFeatureVector(
                id=record_id,
                card_id=card_id,
                date=date,
                count_positive=v_card.count_positive,
                count_negative=v_card.count_negative,
                count=v_card.count,
                delta=delta_card,
                previous_delta=v_card.previous_delta,
                previous_study_date=v_card.previous_study_date,
                previous_study_response=v_card.previous_study_response,
            ))
        session.commit()
        session.close()

    def update_v2_test(self, request: UpdateRequestV2, date: datetime) -> dict:
        session = SessionLocal()
        session.add(
            StudyRecord(
                id=request.history_id,
                studyset_id=request.studyset_id,
                user_id=request.user_id,
                card_id=request.fact_id,
                deck_id=request.deck_id,
                label=request.label,
                date=date,
                elapsed_milliseconds_text=request.elapsed_milliseconds_text,
                elapsed_milliseconds_answer=request.elapsed_milliseconds_answer,
                count=0,  # TODO
                count_session=0,  # TODO
            )
        )
        session.commit()
        session.close()


    def update_v2(self, request: UpdateRequestV2, date: datetime) -> dict:
        session = SessionLocal()
        t0 = datetime.now(pytz.utc)

        if request.test_mode:
            self.update_v2_test(request, date)
            return

        v_usercard = session.query(CurrUserCardFeatureVector).get((request.user_id, request.fact_id))
        count = 0
        count_session = 0
        if v_usercard is not None:
            count = v_usercard.count
            if v_usercard.schedule_request_id == request.debug_id:
                count_session = v_usercard.count_session

        record = StudyRecord(
                id=request.history_id,
                debug_id=request.debug_id,
                studyset_id=request.studyset_id,
                user_id=request.user_id,
                card_id=request.fact_id,
                deck_id=request.deck_id,
                label=request.label,
                date=date,
                elapsed_milliseconds_text=request.elapsed_milliseconds_text,
                elapsed_milliseconds_answer=request.elapsed_milliseconds_answer,
                count=count,
                count_session=count_session,
            )
        session.add(record)
        session.commit()

        # update user stats
        t1 = datetime.now(pytz.utc)
        utc_date = date.astimezone(pytz.utc).date()
        self.update_user_stats(record, deck_id='all', utc_date=utc_date, session=session)
        t2 = datetime.now(pytz.utc)
        if record.deck_id is not None:
            self.update_user_stats(record, deck_id=record.deck_id, utc_date=utc_date, session=session)
        t3 = datetime.now(pytz.utc)

        # update current features
        # includes leitner and sm2 updates
        self.update_feature_vectors(record, date, session)
        session.commit()
        session.close()

        t4 = datetime.now(pytz.utc)

        return {
            'save record': (t1 - t0).total_seconds(),
            'update user_stats all': (t2 - t1).total_seconds(),
            'update user_stats deck': (t3 - t2).total_seconds(),
            'update feature vectors': (t4 - t3).total_seconds(),
        }

    def save_snapshots(
        self,
        schedule_request_id: str,
        user_id: str,
        card_id: str,
        date: datetime,
    ) -> None:
        v_user = self.get_curr_user_vector(user_id)
        v_card = self.get_curr_card_vector(card_id)
        v_usercard = self.get_curr_usercard_vector(user_id, card_id)

        session = SessionLocal()

        delta = None
        if v_usercard.previous_study_date is not None:
            delta = (date - v_usercard.previous_study_date).total_seconds()
        if v_usercard.schedule_request_id is None or v_usercard.schedule_request_id != schedule_request_id:
            # new session/studyset
            # using schedule_request_id in place of studyset_id here since the 
            # latter is created after the schedule request on Matthew's end
            count_positive_session = count_negative_session = count_session = 0
            delta_session = previous_delta_session = previous_study_date_session = previous_study_response_session = None
            correct_on_first_try_session = None
        else:
            # same session
            count_positive_session = v_usercard.count_positive_session 
            count_negative_session = v_usercard.count_negative_session 
            count_session = v_usercard.count_session 
            previous_delta_session = v_usercard.previous_delta_session 
            previous_study_date_session = v_usercard.previous_study_date_session 
            previous_study_response_session = v_usercard.previous_study_response_session 
            correct_on_first_try_session = v_usercard.correct_on_first_try_session 
            if v_usercard.previous_study_date_session is not None:
                delta_session = (date - v_usercard.previous_study_date_session).total_seconds()
            else:
                delta_session = None
        session.add(
            UserCardSnapshot(
                id=schedule_request_id,
                user_id=user_id,
                card_id=card_id,
                date=date,
                count_positive=v_usercard.count_positive,
                count_negative=v_usercard.count_negative,
                count=v_usercard.count,
                count_positive_session=count_positive_session,
                count_negative_session=count_negative_session,
                count_session=count_session,
                delta=delta,
                previous_delta=v_usercard.previous_delta,
                previous_study_date=v_usercard.previous_study_date,
                previous_study_response=v_usercard.previous_study_response,
                delta_session=delta_session,
                previous_delta_session=previous_delta_session,
                previous_study_date_session=previous_study_date_session,
                previous_study_response_session=previous_study_response_session,
                leitner_box=v_usercard.leitner_box,
                leitner_scheduled_date=v_usercard.leitner_scheduled_date,
                sm2_efactor=v_usercard.sm2_efactor,
                sm2_interval=v_usercard.sm2_interval,
                sm2_repetition=v_usercard.sm2_repetition,
                sm2_scheduled_date=v_usercard.sm2_scheduled_date,
                correct_on_first_try=v_usercard.correct_on_first_try,
                correct_on_first_try_session=correct_on_first_try_session,
            ))
        session.commit()

        delta = None
        if v_user.previous_study_date is not None:
            delta = (date - v_user.previous_study_date).total_seconds()

        if v_user.schedule_request_id is None or v_user.schedule_request_id != schedule_request_id:
            # new session/studyset
            # using schedule_request_id in place of studyset_id here since the 
            # latter is created after the schedule request on Matthew's end
            count_positive_session = count_negative_session = count_session = 0
            delta_session = previous_delta_session = previous_study_date_session = previous_study_response_session = None
        else:
            # same session
            count_positive_session = v_user.count_positive_session 
            count_negative_session = v_user.count_negative_session 
            count_session = v_user.count_session 
            previous_delta_session = v_user.previous_delta_session 
            previous_study_date_session = v_user.previous_study_date_session 
            previous_study_response_session = v_user.previous_study_response_session 
            if v_user.previous_study_date_session is not None:
                delta_session = (date - v_user.previous_study_date_session).total_seconds()
            else:
                delta_session = None

        params = ParametersSchema(**self.get_user(user_id, session).parameters.__dict__)
        session.add(
            UserSnapshot(
                id=schedule_request_id,
                user_id=user_id,
                date=date,
                count_positive=v_user.count_positive,
                count_negative=v_user.count_negative,
                count=v_user.count,
                count_positive_session=count_positive_session,
                count_negative_session=count_negative_session,
                count_session=count_session,
                delta=delta,
                previous_delta=v_user.previous_delta,
                previous_study_date=v_user.previous_study_date,
                previous_study_response=v_user.previous_study_response,
                delta_session=delta_session,
                previous_delta_session=previous_delta_session,
                previous_study_date_session=previous_study_date_session,
                previous_study_response_session=previous_study_response_session,
                parameters=json.dumps(params.__dict__),
            ))
        session.commit()

        delta = None
        if v_card.previous_study_date is not None:
            delta = (date - v_card.previous_study_date).total_seconds()
        session.add(
            CardSnapshot(
                id=schedule_request_id,
                card_id=card_id,
                date=date,
                count_positive=v_card.count_positive,
                count_negative=v_card.count_negative,
                count=v_card.count,
                delta=delta,
                previous_delta=v_card.previous_delta,
                previous_study_date=v_card.previous_study_date,
                previous_study_response=v_card.previous_study_response,
            ))
        session.commit()
        session.close()

    def update_feature_vectors(
        self,
        record: StudyRecord,
        date: datetime,
        session: Session,
        simulated: bool = False,
    ):
        if simulated:
            v_user = session.query(SimUserFeatureVector).get(record.user_id)
            v_card = session.query(SimCardFeatureVector).get(record.card_id)
            v_usercard = session.query(SimUserCardFeatureVector).get((record.user_id, record.card_id))
        else:
            v_user = session.query(CurrUserFeatureVector).get(record.user_id)
            v_card = session.query(CurrCardFeatureVector).get(record.card_id)
            v_usercard = session.query(CurrUserCardFeatureVector).get((record.user_id, record.card_id))

        delta_usercard = None
        if v_usercard.previous_study_date is not None:
            delta_usercard = (date - v_usercard.previous_study_date).total_seconds()
        v_usercard.count_positive += record.label
        v_usercard.count_negative += (not record.label)
        v_usercard.count += 1
        v_usercard.previous_delta = delta_usercard
        v_usercard.previous_study_date = date
        v_usercard.previous_study_response = record.label
        if v_usercard.correct_on_first_try is None:
            v_usercard.correct_on_first_try = record.label
            # NOTE the correponding `UserCardFeatureVector` saved in # `save_feature_vectors`
            # from the scheduling request also has None in its `correct_on_first_try`.
            # this is fine, we leave it as None, since that saved feature vector is what was
            # visible to the scheduler before the response was sent by the user.

        if v_usercard.schedule_request_id is None or v_usercard.schedule_request_id != record.debug_id:
            v_usercard.count_positive_session = int(record.label)
            v_usercard.count_negative_session = int(not record.label)
            v_usercard.count_session = 1
            v_usercard.previous_delta_session = None
            v_usercard.previous_study_date_session = date
            v_usercard.previous_study_response_session = record.label
            v_usercard.correct_on_first_try_session = record.label
        else:
            v_usercard.count_positive_session += record.label
            v_usercard.count_negative_session += (not record.label)
            v_usercard.count_session += 1
            v_usercard.previous_delta_session = delta_usercard
            v_usercard.previous_study_date_session = date
            v_usercard.previous_study_response_session = record.label
            if v_usercard.correct_on_first_try_session is None:
                v_usercard.correct_on_first_try_session = record.label

        # update leitner
        self.update_leitner(v_usercard, record, date, session)
        # update sm2
        self.update_sm2(v_usercard, record, date, session)

        delta_user = None
        if v_user.previous_study_date is not None:
            delta_user = (date - v_user.previous_study_date).total_seconds()
        v_user.count_positive += record.label
        v_user.count_negative += (not record.label)
        v_user.count += 1
        v_user.previous_delta = delta_user
        v_user.previous_study_date = date
        v_user.previous_study_response = record.label

        if v_user.schedule_request_id is None or v_user.schedule_request_id != record.debug_id:
            v_user.count_positive_session = int(record.label)
            v_user.count_negative_session = int(not record.label)
            v_user.count_session = 1
            v_user.previous_delta_session = None
            v_user.previous_study_date_session = date
            v_user.previous_study_response_session = record.label
        else:
            v_user.count_positive_session += record.label
            v_user.count_negative_session += (not record.label)
            v_user.count_session += 1
            v_user.previous_delta_session = delta_usercard
            v_user.previous_study_date_session = date
            v_user.previous_study_response_session = record.label

        delta_card = None
        if v_card.previous_study_date is not None:
            delta_card = (date - v_card.previous_study_date).total_seconds()
        v_card.count_positive += record.label
        v_card.count_negative += (not record.label)
        v_card.count += 1
        v_card.previous_delta = delta_card
        v_card.previous_study_date = date
        v_card.previous_study_response = record.label

    def update_user_stats(
        self,
        record: StudyRecord,
        utc_date,
        deck_id: str,
        session: Session,
    ):
        # get the latest user_stat ordered by date
        curr_stats = session.query(UserStatsV2).\
            filter(UserStatsV2.user_id == record.user_id).\
            filter(UserStatsV2.deck_id == deck_id).\
            order_by(UserStatsV2.date.desc()).first()

        is_new_stat = False
        if curr_stats is None:
            stats_id = json.dumps({
                'user_id': record.user_id,
                'deck_id': deck_id,
                'date': str(utc_date),
            })
            curr_stats = UserStatsV2(
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
            new_stat = UserStatsV2(
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

        if record.count == 0:
            curr_stats.n_new_cards_total += 1
            curr_stats.n_new_cards_positive += record.label
        else:
            curr_stats.n_old_cards_total += 1
            curr_stats.n_old_cards_positive += record.label

        curr_stats.n_cards_total += 1
        curr_stats.n_cards_positive += record.label
        curr_stats.elapsed_milliseconds_text += record.elapsed_milliseconds_text
        curr_stats.elapsed_milliseconds_answer += record.elapsed_milliseconds_answer

        if is_new_stat:
            session.add(curr_stats)

    def update_leitner(
        self,
        v_usercard: Union[CurrUserCardFeatureVector, SimUserCardFeatureVector],
        record: StudyRecord,
        date: datetime,
        session: Session,
    ) -> None:
        # leitner boxes 1~10
        # days[0] = None as placeholder since we don't have box 0
        # days[9] and days[10] = 999 to make it never repeat
        days = [0, 0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 999, 999]
        increment_days = {i: x for i, x in enumerate(days)}

        if v_usercard.leitner_box is None:
            # boxes: 1 ~ 10
            v_usercard.leitner_box = 1

        v_usercard.leitner_box += (1 if record.label else -1)
        v_usercard.leitner_box = max(min(v_usercard.leitner_box, 10), 1)
        interval = timedelta(days=increment_days[v_usercard.leitner_box])
        v_usercard.leitner_scheduled_date = date + interval

    def update_sm2(
        self,
        v_usercard: Union[CurrUserCardFeatureVector, SimUserCardFeatureVector],
        record: StudyRecord,
        date: datetime,
        session: Session,
    ) -> None:
        def get_quality_from_response(response: bool) -> int:
            return 4 if response else 1

        if v_usercard.sm2_repetition is None:
            v_usercard.sm2_repetition = 0
            v_usercard.sm2_efactor = 2.5
            v_usercard.sm2_interval = 1
            v_usercard.sm2_repetition = 0

        q = get_quality_from_response(record.label)
        v_usercard.sm2_repetition += 1
        v_usercard.sm2_efactor = max(1.3, v_usercard.sm2_efactor + 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02))

        if not record.label:
            v_usercard.sm2_interval = 0
            v_usercard.sm2_repetition = 0
        else:
            if v_usercard.sm2_repetition == 1:
                v_usercard.sm2_interval = 1
            elif v_usercard.sm2_repetition == 2:
                v_usercard.sm2_interval = 6
            else:
                v_usercard.sm2_interval *= v_usercard.sm2_efactor

        v_usercard.sm2_interval = min(500, v_usercard.sm2_interval)
        try:
            v_usercard.sm2_scheduled_date = date + timedelta(days=v_usercard.sm2_interval)
        except OverflowError:
            pass

    def get_rationale(
        self,
        schedule_request_id: str,
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

        rr += '<p>Debug ID: {}</p>'.format(schedule_request_id)
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

            rr += row_template.format('#Correct', v_usercard.count_positive)
            rr += row_template.format('#Wrong', v_usercard.count_negative)
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
