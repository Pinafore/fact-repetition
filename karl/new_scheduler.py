#!/usr/bin/env python
# coding: utf-8

import json
import pytz
import requests
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict
from sqlalchemy.orm import Session

from karl.schemas import ScheduleRequestSchema, UpdateRequestSchema, ScheduleResponseSchema, \
    RetentionFeaturesSchema, ParametersSchema
from karl.models import User, Card, Record, Parameters, \
    UserCardFeatureVector, UserFeatureVector, CardFeatureVector,\
    CurrUserCardFeatureVector, CurrUserFeatureVector, CurrCardFeatureVector


class KARLScheduler:

    def get_user(
        self,
        user_id: str,
        session: Session,
    ) -> User:
        """
        Get user from DB. Create new if user does not exist.

        :param user_id: the `user_id` of the user to load.
        :return: the user.
        """
        user = session.query(User).get(user_id)
        if user is not None:
            return user

        # create new user and insert to db
        new_user = User(id=user_id)
        new_params = Parameters(id=user_id, **ParametersSchema().__dict__)
        session.add(new_user)
        session.add(new_params)
        return new_user

    def get_card(
        self,
        request: ScheduleRequestSchema,
        session: Session,
    ) -> Card:
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
        session.add(card)

        # TODO create embedding

        return card

    def score_user_card(
        self,
        user: User,
        card: Card,
        date: datetime,
        v_usercard: CurrUserCardFeatureVector,
        v_user: CurrUserFeatureVector,
        v_card: CurrCardFeatureVector,
    ) -> Dict[str, float]:
        scores = {
            'recall': self.score_recall(v_usercard, v_user, v_card),
            # 'category': self.dist_category(user, fact),
            # 'answer': self.dist_answer(user, fact),
            # 'cool_down': self.dist_cool_down(user, fact, date),
            # 'leitner': self.dist_leitner(user, fact, date),
            # 'sm2': self.dist_sm2(user, fact, date),
        }
        return scores

    def score_user_cards(
        self,
        user: User,
        cards: List[Card],
        date: datetime,
        vs_usercard: List[CurrUserCardFeatureVector],
        vs_user: List[CurrUserFeatureVector],
        vs_card: List[CurrCardFeatureVector],
    ) -> List[Dict[str, float]]:
        recall_scores = self.score_recall_batch(vs_usercard, vs_user, vs_card)
        scores = [
            {
                'recall': recall_scores[i],
                # 'category': self.dist_category(user, card),
                # 'answer': self.dist_answer(user, card),
                # 'cool_down': self.dist_cool_down(user, card, date),
                # 'leitner': self.dist_leitner(user, card, date),
                # 'sm2': self.dist_sm2(user, card, date),
            }
            for i, card in enumerate(cards)
        ]
        return scores

    def schedule(
        self,
        session,
        schedule_requests: List[ScheduleRequestSchema],
        date: datetime,
        rationale=False,
        details=False,
    ) -> ScheduleResponseSchema:
        # get user and cards
        user = self.get_user(schedule_requests[0].user_id, session)
        cards = [self.get_card(request, session) for request in schedule_requests]

        # score cards
        vs_usercard, vs_user, vs_card = [], [], []
        for card in cards:
            v_usercard, v_user, v_card = self.get_curr_features(user.id, card.id, session)
            vs_usercard.append(v_usercard)
            vs_user.append(v_user)
            vs_card.append(v_card)
        if False:
            scores = []
            for i, card in enumerate(tqdm(cards)):
                scores.append(
                    self.score_user_card(user, card, date, vs_usercard[i], vs_user[i], vs_card[i])
                )
        else:
            scores = self.score_user_cards(user, cards, date, vs_usercard, vs_user, vs_card)

        # computer total score
        user_parameters = session.query(Parameters).get(user.id)
        for i, _ in enumerate(scores):
            scores[i]['sum'] = sum([
                user_parameters.__dict__.get(key, 0) * value
                for key, value in scores[i].items()
            ])

        # sort cards
        order = np.argsort([s['sum'] for s in scores]).tolist()
        card_selected = cards[order[0]]

        # determin if is new card
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

        # store feature vectors @ debug_id
        self.save_feature_vectors(record.id, user.id, card_selected.id, date, session)

        # return
        return ScheduleResponseSchema(
            order=order,
            debug_id=record_id,
            scores=scores,
        )

    def update(
        self,
        session,
        request: UpdateRequestSchema,
        date: datetime
    ) -> dict:
        # read debug_id from request
        # add front_end_id to record, response, elapsed_times to record
        # update current features
        # update user stats
        # update leitner
        # update sm2
        pass

    def get_curr_features(
        self,
        user_id: str,
        card_id: str,
        session: Session,
    ):
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
            )
            session.add(v_usercard)

        v_user = session.query(CurrUserFeatureVector).get(user_id)
        if v_user is None:
            v_user = CurrUserFeatureVector(
                user_id=user_id,
                n_study_positive=0,
                n_study_negative=0,
                n_study_total=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
            )
            session.add(v_user)

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
        return v_usercard, v_user, v_card

    def predict_recall(
        self,
        user_id: str,
        card_id: str,
        date: datetime,
        session: Session,
    ) -> float:
        v_usercard, v_user, v_card = self.get_curr_features(user_id, card_id, session)
        return self.score_recall(v_usercard, v_user, v_card)

    def score_recall(
        self,
        v_usercard: CurrUserCardFeatureVector,
        v_user: CurrUserFeatureVector,
        v_card: CurrCardFeatureVector,
    ) -> float:
        user_previous_result = v_user.previous_study_response
        if user_previous_result is None:
            user_previous_result = False
        features = RetentionFeaturesSchema(
            user_count_correct=v_usercard.n_study_positive,
            user_count_wrong=v_usercard.n_study_negative,
            user_count_total=v_usercard.n_study_total,
            user_average_overall_accuracy=0 if v_user.n_study_total == 0 else v_user.n_study_positive / v_user.n_study_total,
            user_average_question_accuracy=0 if v_card.n_study_total == 0 else v_card.n_study_positive / v_card.n_study_negative,
            user_previous_result=user_previous_result,
            user_gap_from_previous=0,
            question_average_overall_accuracy=0,
            question_count_total=0,
            question_count_correct=0,
            question_count_wrong=0,
        )

        return float(
            requests.get(
                'http://127.0.0.1:8001/api/karl/predict_one',
                data=json.dumps(features.__dict__)
            ).text
        )

    def score_recall_batch(
        self,
        vs_usercard: List[CurrUserCardFeatureVector],
        vs_user: List[CurrUserFeatureVector],
        vs_card: List[CurrCardFeatureVector],
    ) -> List[float]:
        feature_vectors = []
        for v_usercard, v_user, v_card in zip(vs_usercard, vs_user, vs_card):
            user_previous_result = v_user.previous_study_response
            if user_previous_result is None:
                user_previous_result = False
            feature_vectors.append(RetentionFeaturesSchema(
                user_count_correct=v_usercard.n_study_positive,
                user_count_wrong=v_usercard.n_study_negative,
                user_count_total=v_usercard.n_study_total,
                user_average_overall_accuracy=0 if v_user.n_study_total == 0 else v_user.n_study_positive / v_user.n_study_total,
                user_average_question_accuracy=0 if v_card.n_study_total == 0 else v_card.n_study_positive / v_card.n_study_negative,
                user_previous_result=user_previous_result,
                user_gap_from_previous=0,
                question_average_overall_accuracy=0,
                question_count_total=0,
                question_count_correct=0,
                question_count_wrong=0,
            ))

        return json.loads(
            requests.get(
                'http://127.0.0.1:8001/api/karl/predict',
                data=json.dumps([x.__dict__ for x in feature_vectors])
            ).text
        )

    def save_feature_vectors(
        self,
        record_id: str,
        user_id: str,
        card_id: str,
        date: datetime,
        session: Session,
    ) -> None:
        v_usercard, v_user, v_card = self.get_curr_features(user_id, card_id, session)
        delta_usercard = None
        if v_usercard.previous_study_date is not None:
            delta_usercard = (date - v_usercard.previous_study_date).total_seconds()
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
            ))
        delta_user = None
        if v_user.previous_study_date is not None:
            delta_user = (date - v_user.previous_study_date).total_seconds()
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
