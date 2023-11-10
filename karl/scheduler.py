#!/usr/bin/env python
# coding: utf-8

import json
from fastapi import HTTPException
import pytz
import requests
import numpy as np
import multiprocessing
from typing import List, Dict, Union
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.orm import Session
from karl.fsrs_models import FSRSCard, State, Rating
from karl.fsrs import FSRS

from karl.schemas import ScheduleResponseSchema,\
    ScheduleRequestSchema, UpdateRequestSchema, KarlFactSchema
from karl.schemas import ParametersSchema
from karl.schemas import VUser, VCard, VUserCard
from karl.schemas import RepetitionModel
from karl.models import User, Card, Parameters, UserStatsV2,\
    UserCardFeatureVector, UserFeatureVector, CardFeatureVector,\
    UserCardSnapshotV2, UserSnapshotV2, CardSnapshotV2,\
    StudyRecord, TestRecord, ScheduleRequest

from karl.retention_phase1 import vectors_to_features, fsrs_vectors_to_features
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

        params = session.query(Parameters).get(user_id)
        if params is None:
            params = Parameters(id=user_id, **(ParametersSchema().__dict__))
            session.add(params)
            session.commit()

        return user

    def get_card(
        self,
        request: Union[ScheduleRequestSchema, KarlFactSchema],
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

        v_card = CardFeatureVector(card_id=card.id)
        session.add(v_card)
        session.commit()

        return card


    def schedule_delta(
        self,
        request: ScheduleRequestSchema,
        date: datetime,
    ) -> ScheduleResponseSchema:
        '''
        Average of the correct wrong in one day’s time vs the recall
        probability in one day’s time if there was no study at that point.
        '''

        # schedule_request_id === debug_id
        schedule_request_id = json.dumps({
            'user_id': request.user_id,
            'repetition_model': request.repetition_model,
            'date': str(date.replace(tzinfo=pytz.UTC)),
        })

        # store schedule request
        session = SessionLocal()
        session.add(
            ScheduleRequest(
                id=schedule_request_id,
                user_id=request.user_id,
                card_ids=[x.fact_id for x in request.facts],
                repetition_model=request.repetition_model,
                recall_target=request.recall_target.target,
                recall_target_lowest=request.recall_target.target_window_lowest,
                recall_target_highest=request.recall_target.target_window_highest,
                date=date,
            )
        )
        session.commit()
        session.close()

        if len(request.facts) == 0:
            return ScheduleResponseSchema(
                debug_id=schedule_request_id,
                order=[],
                scores=[],
            )

        session = SessionLocal(expire_on_commit=False)
        user = self.get_user(request.user_id, session)
        cards = [self.get_card(fact, session) for fact in request.facts]

        tomorrow = date + timedelta(days=1)
        # get prediction of cards in a days time *if* there is no study now
        scores_no_study, profile, _ = self.karl_score_recall_batch(user, cards, date, session, request)
        # get prediction of cards in a days time *if* answered correctly
        scores_correct, profile = self.karl_score_recall_batch_future(user, cards, date, tomorrow, True, session)
        # get prediction of cards in a days time *if* answered incorrectly
        scores_wrong, profile = self.karl_score_recall_batch_future(user, cards, date, tomorrow, False, session)
        # difference between the average prediction of correct/wrong outcomes *now* and that without studying now
        # TODO multiply by predicted retention probability?
        deltas = [abs((a + b) / 2 - c) for a, b, c in zip(scores_correct, scores_wrong, scores_no_study)]
        indexed_deltas = [(i, delta) for i, delta in enumerate(deltas)]
        order = [i for i, _ in sorted(indexed_deltas, key=lambda x: x[1], reverse=True)]
        deltas_ordered = [delta for _, delta in sorted(indexed_deltas, key=lambda x: x[1], reverse=True)]
        print('*********')
        print(len(order))
        # print(scores_no_study)
        # print(scores_correct)
        # print(scores_wrong)
        # print(deltas)
        # print(deltas_ordered)
        print()

        session.commit()
        session.close()

        return ScheduleResponseSchema(
            order=order,
            debug_id=schedule_request_id,
            scores=deltas,
            profile=profile,
        )


    def schedule_fsrs_karl_no_delta(
        self,
        request: ScheduleRequestSchema,
        date: datetime,
    ) -> ScheduleResponseSchema:
        # schedule_request_id === debug_id
        schedule_request_id = json.dumps({
            'user_id': request.user_id,
            'repetition_model': request.repetition_model,
            'date': str(date.replace(tzinfo=pytz.UTC)),
        })

        # store schedule request
        session = SessionLocal()
        session.add(
            ScheduleRequest(
                id=schedule_request_id,
                user_id=request.user_id,
                card_ids=[x.fact_id for x in request.facts],
                repetition_model=request.repetition_model,
                recall_target=request.recall_target.target,
                recall_target_lowest=request.recall_target.target_window_lowest,
                recall_target_highest=request.recall_target.target_window_highest,
                date=date,
            )
        )
        session.commit()
        session.close()

        if len(request.facts) == 0:
            return ScheduleResponseSchema(
                debug_id=schedule_request_id,
                order=[],
                scores=[],
            )

        session = SessionLocal(expire_on_commit=False)

        user = self.get_user(request.user_id, session)
        cards = [self.get_card(fact, session) for fact in request.facts]

        # score cards
        # if request.repetition_model == RepetitionModel.karl:
        #     scores, profile = self.karl_score_recall_batch(user, cards, date, session)
        # elif request.repetition_model == RepetitionModel.settles:
        #     pass # TODO
        # elif request.repetition_model == RepetitionModel.leitner:
        #     pass
        # else:
        #     pass # TODO

        if request.repetition_model == RepetitionModel.fsrs:
            scores, profile, order = self.fsrs_score_recall_batch(user, cards, date, session, request)
        elif request.repetition_model in {RepetitionModel.karlAblation, RepetitionModel.karl}:
            scores, profile, order = self.karl_score_recall_batch(user, cards, date, session)
        else:
            raise HTTPException(status_code=557, detail="Scheduler not implemented")
        
        session.commit()
        session.close()

        return ScheduleResponseSchema(
            order=order,
            debug_id=schedule_request_id,
            scores=scores,
            profile=profile,
        )

    def get_user_vector(self, user_id: str, session):
        v_user = session.query(UserFeatureVector).get(user_id)
        params = ParametersSchema(**self.get_user(user_id, session).parameters.__dict__)
        if v_user is None:
            v_user = UserFeatureVector(user_id=user_id, parameters=json.dumps(params.__dict__))
            session.add(v_user)
            session.commit()
        return v_user

    def get_card_vector(self, card_id: str, session):
        v_card = session.query(CardFeatureVector).get(card_id)
        if v_card is None:
            v_card = CardFeatureVector(card_id=card_id)
            session.add(v_card)
            session.commit()
        return v_card

    def get_usercard_vector(self, user_id: str, card_id: str, session):
        v_usercard = session.query(UserCardFeatureVector).get((user_id, card_id))
        if v_usercard is None:
            v_usercard = UserCardFeatureVector(user_id=user_id, card_id=card_id)
            session.add(v_usercard)
            session.commit()
        return v_usercard
    
    def collect_features_fsrs(self, user_id, card_id, card_text, v_user, date):
        '''helper for multiprocessing'''
        session = SessionLocal(expire_on_commit=False)
        v_usercard = self.get_usercard_vector(user_id, card_id, session).__dict__
        session.close()
        return fsrs_vectors_to_features(v_usercard)

    def collect_features(self, user_id, card_id, card_text, v_user, date):
        '''helper for multiprocessing'''
        session = SessionLocal(expire_on_commit=False)
        v_card = VCard(**self.get_card_vector(card_id, session).__dict__)
        v_usercard = VUserCard(**self.get_usercard_vector(user_id, card_id, session).__dict__)
        session.close()
        return vectors_to_features(v_usercard, v_user, v_card, date, card_text)
    
    def collect_features_for_future(self, user_id, card_id, card_text, v_user, date, future, forced_result):
        '''helper for multiprocessing'''
        session = SessionLocal(expire_on_commit=False)
        v_card = VCard(**self.get_card_vector(card_id, session).__dict__)
        v_usercard = VUserCard(**self.get_usercard_vector(user_id, card_id, session).__dict__)
        session.close()

        previous_delta = None
        if v_usercard.previous_study_date is not None:
            previous_delta = (date - v_usercard.previous_study_date).total_seconds()
        v_usercard.previous_delta = previous_delta 
        v_usercard.count_positive += int(forced_result)
        v_usercard.count_negative += int(not forced_result)
        v_usercard.count += 1
        v_usercard.previous_study_date = date
        v_usercard.previous_study_response = forced_result
        if v_usercard.correct_on_first_try is None:
            v_usercard.correct_on_first_try = forced_result

        # update leitner
        self.update_leitner(v_usercard, forced_result, date, session)
        # update sm2
        self.update_sm2(v_usercard, forced_result, date, session)

        previous_delta = None
        if v_card.previous_study_date is not None:
            previous_delta = (date - v_card.previous_study_date).total_seconds()
        v_card.previous_delta = previous_delta
        v_card.count_positive += int(forced_result)
        v_card.count_negative += int(not forced_result)
        v_card.count += 1
        v_card.previous_study_date = date
        v_card.previous_study_response = forced_result

        return vectors_to_features(v_usercard, v_user, v_card, future, card_text)

    def fsrs_score_recall_batch(
        self,
        user: User,
        cards: List[Card],
        date: datetime,
        session: Session,
        request: ScheduleRequestSchema
    ):
        
        t0 = datetime.now(pytz.utc)

        # gather card features
        feature_vectors = []
        v_user = VUser(**self.get_user_vector(user.id, session).__dict__)

        if not settings.USE_MULTIPROCESSING:
            feature_vectors = [
                self.collect_features_fsrs(user.id, card.id, card.text, v_user, date).__dict__
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
            executor.submit(self.collect_features_fsrs, user.id, card.id, card.text, v_user, date)
            for card in cards
        ]
        feature_vectors = [x.result().__dict__ for x in futures]
        print("FEATURE VECTORS:")
        print(feature_vectors)

        t1 = datetime.now(pytz.utc)
        
        index_score_in_window = []
        for index, fsrs_feature in enumerate(feature_vectors):
            index_score_in_window.append((index, fsrs_feature['fsrs_scheduled_date'].date().toordinal()))
        scores = [x[1] for x in index_score_in_window]
        # print('scores:', scores)
        index_score_in_window = sorted(
            index_score_in_window,
            key=lambda x: x[1],
        )
        order = [x[0] for x in index_score_in_window]
        print(len(order))

        t2 = datetime.now(pytz.utc)

        profile = {
            'schedule gather features': (t1 - t0).total_seconds(),
            'schedule model prediction': (t2 - t1).total_seconds(),
        }
        return scores, profile, order


            #scores = [index for index, _ in sorted(enumerate(feature_vectors), key=lambda x: x[1]['fsrs_due'])]

    def karl_score_recall_batch(
        self,
        user: User,
        cards: List[Card],
        date: datetime,
        session: Session,
        request: ScheduleRequestSchema
    ) -> List[float]:
        t0 = datetime.now(pytz.utc)

        # gather card features
        feature_vectors = []
        v_user = VUser(**self.get_user_vector(user.id, session).__dict__)

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

        if request.repetition_model == RepetitionModel.karl or request.repetition_model == RepetitionModel.karlAblation:
            
            time_start = datetime.now()
            data_dump = json.dumps(feature_vectors)
            print('\n\nTIME DUMPING:', datetime.now() - time_start, '\n\n')
            time_start = datetime.now()
            req = requests.get(
                    f'{settings.MODEL_API_URL}/api/karl/predict',
                    data=data_dump
                ).text
            print('\n\nREQUEST:', datetime.now() - time_start, '\n\n')
            time_start = datetime.now()
            scores = json.loads(req)
            print('\n\nTIME LOADING:', datetime.now() - time_start, '\n\n')
            # sort cards
            index_score_in_window = []
            for i, score in enumerate(scores):
                if score >= request.recall_target.target_window_lowest and \
                        score <= request.recall_target.target_window_highest:
                    index_score_in_window.append((i, score))
            index_score_in_window = sorted(
                index_score_in_window,
                key=lambda x: abs(x[1] - request.recall_target.target),
            )
            order = [x[0] for x in index_score_in_window]
            print(len(order))
        else:
            raise HTTPException(status_code=557, detail="Scheduler not implemented")

        t2 = datetime.now(pytz.utc)

        profile = {
            'schedule gather features': (t1 - t0).total_seconds(),
            'schedule model prediction': (t2 - t1).total_seconds(),
        }
        return scores, profile, order

    def karl_score_recall_batch_future(
        self,
        user: User,
        cards: List[Card],
        date: datetime,
        future: datetime,
        forced_result: bool,
        session: Session,
    ) -> List[float]:
        '''Get predicted retention probability at *future* timestamp given
        *forced result* at current *date*'''
        t0 = datetime.now(pytz.utc)

        # gather card features
        feature_vectors = []
        v_user = VUser(**self.get_user_vector(user.id, session).__dict__)
        previous_delta = None
        if v_user.previous_study_date is not None:
            previous_delta = (date - v_user.previous_study_date).total_seconds()
        v_user.previous_delta = previous_delta
        v_user.count_positive += int(forced_result)
        v_user.count_negative += int(not forced_result)
        v_user.count += 1
        v_user.previous_study_date = date
        v_user.previous_study_response = forced_result

        if not settings.USE_MULTIPROCESSING:
            feature_vectors = [
                self.collect_features_for_future(user.id, card.id, card.text, v_user, date, future, forced_result).__dict__
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
            executor.submit(self.collect_features_for_future, user.id, card.id, card.text, v_user, date,
                            future, forced_result)
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

        t2 = datetime.now(pytz.utc)

        profile = {
            'schedule gather features': (t1 - t0).total_seconds(),
            'schedule model prediction': (t2 - t1).total_seconds(),
        }
        return scores, profile

    def score_cool_down(
        self,
        user: User,
        card: Card,
        date: datetime,
        session: Session,
    ) -> float:
        """
        Avoid repetition of the same card within a cool down time.
        We set a longer cool down time for correctly recalled cards.
        :return: portion of cool down period remained. 0 if passed.
        """
        v_usercard = self.get_usercard_vector(user.id, card.id, session)

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
    ) -> float:
        """
        Time till the scheduled date by Leitner measured by number of hours.
        The value can be negative when the card is over-due in Leitner.
        :return: distance in number of hours.
        """
        v_usercard = self.get_usercard_vector(user.id, card.id, session)

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
    ) -> float:
        """
        Time till the scheduled date by SM-2 measured by number of hours.
        The value can be negative when the card is over-due in SM-2.

        :param user:
        :param card:
        :return: distance in number of hours.
        """
        v_usercard = self.get_usercard_vector(user.id, card.id, session)

        if v_usercard.sm2_scheduled_date is None:
            return 0
        else:
            # NOTE distance in days, can be negative
            return (v_usercard.sm2_scheduled_date - date).total_seconds() / 86400

    def update(self, request: UpdateRequestSchema, date: datetime) -> dict:
        session = SessionLocal()
        t0 = datetime.now(pytz.utc)

        if request.fact is not None:
            card = self.get_card(request.fact, session)

        user = self.get_user(request.user_id, session)

        if request.test_mode:
            session.add(
                TestRecord(
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
            return

        v_usercard = session.query(UserCardFeatureVector).get((request.user_id, request.fact_id))
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
                typed=request.typed,
                recommendation=request.recommendation,
            )
        session.add(record)
        session.commit()

        schedule_request = session.query(ScheduleRequest).get(request.debug_id)
        if schedule_request is None:
            print('******* schedule request not found**********')
            return
        self.save_snapshots(request.debug_id, request.user_id, request.fact_id, schedule_request.date, session)
        session.commit()

        # update user stats
        t1 = datetime.now(pytz.utc)
        utc_date = date.astimezone(pytz.utc).date()
        self.update_user_stats(record, deck_id='all', utc_date=utc_date, session=session)
        t2 = datetime.now(pytz.utc)
        if record.deck_id is not None:
            self.update_user_stats(record, deck_id=record.deck_id, utc_date=utc_date, session=session)
        t3 = datetime.now(pytz.utc)

        # update features
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
        session: Session,
    ) -> None:
        v_user = self.get_user_vector(user_id, session)
        v_card = self.get_card_vector(card_id, session)
        v_usercard = self.get_usercard_vector(user_id, card_id, session)

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
            UserCardSnapshotV2(
                user_id=user_id,
                card_id=card_id,
                schedule_request_id=schedule_request_id,
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
            UserSnapshotV2(
                user_id=user_id,
                schedule_request_id=schedule_request_id,
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

        delta = None
        if v_card.previous_study_date is not None:
            delta = (date - v_card.previous_study_date).total_seconds()
        session.add(
            CardSnapshotV2(
                card_id=card_id,
                schedule_request_id=schedule_request_id,
                date=date,
                count_positive=v_card.count_positive,
                count_negative=v_card.count_negative,
                count=v_card.count,
                delta=delta,
                previous_delta=v_card.previous_delta,
                previous_study_date=v_card.previous_study_date,
                previous_study_response=v_card.previous_study_response,
            ))

    def update_feature_vectors(
        self,
        record: StudyRecord,
        date: datetime,
        session: Session,
    ):
        
        print('UPDATING FEATURE VECTORS')

        v_user = self.get_user_vector(record.user_id, session)
        v_card = self.get_card_vector(record.card_id, session)
        v_usercard = self.get_usercard_vector(record.user_id, record.card_id, session)

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
            # NOTE the correponding `UserCardSnapshot` saved in # `save_feature_vectors`
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
        self.update_leitner(v_usercard, record.label, date, session)
        # update sm2
        self.update_sm2(v_usercard, record.label, date, session)
        # update fsrs
        self.update_fsrs(v_usercard, record, date, session)

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
        v_usercard: UserCardFeatureVector,
        record_label: bool,
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

        v_usercard.leitner_box += (1 if record_label else -1)
        v_usercard.leitner_box = max(min(v_usercard.leitner_box, 10), 1)
        interval = timedelta(days=increment_days[v_usercard.leitner_box])
        v_usercard.leitner_scheduled_date = date + interval

    def update_fsrs(
        self,
        v_usercard: UserCardFeatureVector, 
        record: StudyRecord,
        date: datetime,
        session: Session,
    ) -> None:
        
        due = date if v_usercard.fsrs_scheduled_date == None else v_usercard.fsrs_scheduled_date  # IDK if this is right. I think it is?

        print()
        print('OLD DATE:', due)

        old_state = State.New if v_usercard.state == None else v_usercard.state # Added a new column for state

        elapsed_days = 0 if v_usercard.previous_delta == None else v_usercard.previous_delta // 3600
        reps = v_usercard.count_positive
        lapses = v_usercard.count_negative
        last_review = v_usercard.previous_study_date
        stability = v_usercard.stability
        difficulty = v_usercard.difficulty

        f = FSRS()
        fsrs_card = FSRSCard(due, stability, difficulty, elapsed_days, reps, lapses, old_state, last_review)
        scheduling_cards = f.repeat(fsrs_card, due)
        rating = Rating.Good if record.label else Rating.Again
        card = scheduling_cards[rating].card

        state_mapping = {State.New: {False: State.Learning, True: State.Review},
                    State.Learning: {False: State.Learning, True: State.Review},
                    State.Review: {False: State.Learning, True: State.Review}}
        
        state = state_mapping[old_state][record.label] # This is how I updated the states during training
        
        v_usercard.fsrs_scheduled_date = card.due
        v_usercard.stability = card.stability
        v_usercard.difficulty = card.difficulty
        v_usercard.state = state

        print('NEW DATE:', v_usercard.fsrs_scheduled_date)
        print('OLD STATE:', old_state)
        print('NEW STATE:', state)

        print()

        #print(v_usercard.fsrs_scheduled_date)

    def update_sm2(
        self,
        v_usercard: UserCardFeatureVector, 
        record_label: bool,
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

        q = get_quality_from_response(record_label)
        v_usercard.sm2_repetition += 1
        v_usercard.sm2_efactor = max(1.3, v_usercard.sm2_efactor + 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02))

        if not record_label:
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
