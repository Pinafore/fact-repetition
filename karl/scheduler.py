#!/usr/bin/env python
# coding: utf-8

import json
import pytz
import requests
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from karl.schemas import ScheduleRequestSchema, UpdateRequestSchema, ScheduleResponseSchema,\
    RetentionFeaturesSchema, ParametersSchema
from karl.models import User, Card, Record, Parameters, UserStats,\
    UserCardFeatureVector, UserFeatureVector, CardFeatureVector,\
    CurrUserCardFeatureVector, CurrUserFeatureVector, CurrCardFeatureVector,\
    Leitner, SM2


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
        session.commit()
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
        session.commit()

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
        session: Session,
    ) -> Dict[str, float]:
        scores = {
            'recall': self.score_recall(v_usercard, v_user, v_card, date),
            'category': self.score_category(user, card),
            'cool_down': self.score_cool_down(user, card, date, session),
            'leitner': self.score_leitner(user, card, date, session),
            'sm2': self.score_sm2(user, card, date, session),
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
        session: Session,
    ) -> List[Dict[str, float]]:
        recall_scores = self.score_recall_batch(vs_usercard, vs_user, vs_card, date)
        scores = [
            {
                'recall': recall_scores[i],
                'category': self.score_category(user, card),
                'cool_down': self.score_cool_down(user, card, date, session),
                'leitner': self.score_leitner(user, card, date, session),
                'sm2': self.score_sm2(user, card, date, session),
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

        # gather card features
        vs_usercard, vs_user, vs_card = [], [], []
        for card in cards:
            v_usercard, v_user, v_card = self.get_feature_vectors(user.id, card.id, session)
            vs_usercard.append(v_usercard)
            vs_user.append(v_user)
            vs_card.append(v_card)

        # score cards
        if False:
            scores = [
                self.score_user_card(
                    user, card, date,
                    vs_usercard[i], vs_user[i], vs_card[i],
                    session
                ) for i, card in enumerate(tqdm(cards))
            ]
        else:
            scores = self.score_user_cards(
                user, cards, date, vs_usercard, vs_user, vs_card, session,
            )

        # computer total score
        for i, _ in enumerate(scores):
            scores[i]['sum'] = sum([
                user.parameters.__dict__.get(key, 0) * value
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
        session.commit()

        # store feature vectors @ debug_id
        self.save_feature_vectors(record.id, user.id, card_selected.id, date, session)

        # return
        return ScheduleResponseSchema(
            order=order,
            debug_id=record_id,
            scores=scores,
        )

    def get_feature_vectors(
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
            session.commit()

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
            session.commit()

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

        return v_usercard, v_user, v_card

    def predict_recall(
        self,
        user_id: str,
        card_id: str,
        date: datetime,
        session: Session,
    ) -> float:
        v_usercard, v_user, v_card = self.get_feature_vectors(user_id, card_id, session)
        return self.score_recall(v_usercard, v_user, v_card, date)

    def score_recall(
        self,
        v_usercard: CurrUserCardFeatureVector,
        v_user: CurrUserFeatureVector,
        v_card: CurrCardFeatureVector,
        date: datetime,
    ) -> float:
        user_previous_result = v_user.previous_study_response
        if user_previous_result is None:
            user_previous_result = False
        delta_usercard = 0
        if v_usercard.previous_study_date is not None:
            delta_usercard = (date - v_usercard.previous_study_date).total_seconds()
        features = RetentionFeaturesSchema(
            user_count_correct=v_usercard.n_study_positive,
            user_count_wrong=v_usercard.n_study_negative,
            user_count_total=v_usercard.n_study_total,
            user_average_overall_accuracy=0 if v_user.n_study_total == 0 else v_user.n_study_positive / v_user.n_study_total,
            user_average_question_accuracy=0 if v_usercard.n_study_total == 0 else v_usercard.n_study_positive / v_usercard.n_study_total,
            user_previous_result=user_previous_result,
            user_gap_from_previous=delta_usercard,
            question_average_overall_accuracy=0 if v_card.n_study_total == 0 else v_card.n_study_positive / v_card.n_study_total,
            question_count_total=v_card.n_study_total,
            question_count_correct=v_card.n_study_positive,
            question_count_wrong=v_card.n_study_negative,
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
        date: datetime,
    ) -> List[float]:
        feature_vectors = []
        for v_usercard, v_user, v_card in zip(vs_usercard, vs_user, vs_card):
            user_previous_result = v_user.previous_study_response
            if user_previous_result is None:
                user_previous_result = False
            delta_usercard = 0
            if v_usercard.previous_study_date is not None:
                delta_usercard = (date - v_usercard.previous_study_date).total_seconds()
            feature_vectors.append(RetentionFeaturesSchema(
                user_count_correct=v_usercard.n_study_positive,
                user_count_wrong=v_usercard.n_study_negative,
                user_count_total=v_usercard.n_study_total,
                user_average_overall_accuracy=0 if v_user.n_study_total == 0 else v_user.n_study_positive / v_user.n_study_total,
                user_average_question_accuracy=0 if v_usercard.n_study_total == 0 else v_usercard.n_study_positive / v_usercard.n_study_total,
                user_previous_result=user_previous_result,
                user_gap_from_previous=delta_usercard,
                question_average_overall_accuracy=0 if v_card.n_study_total == 0 else v_card.n_study_positive / v_card.n_study_total,
                question_count_total=v_card.n_study_total,
                question_count_correct=v_card.n_study_positive,
                question_count_wrong=v_card.n_study_negative,
            ))

        return json.loads(
            requests.get(
                'http://127.0.0.1:8001/api/karl/predict',
                data=json.dumps([x.__dict__ for x in feature_vectors])
            ).text
        )

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
        session: Session,
    ) -> None:
        v_usercard, v_user, v_card = self.get_feature_vectors(user_id, card_id, session)

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
        session.commit()

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
        session.commit()

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

    def update(
        self,
        session,
        request: UpdateRequestSchema,
        date: datetime
    ) -> dict:
        # read debug_id from request, find corresponding record
        record = session.query(Record).get(request.debug_id)

        # add front_end_id to record, response, elapsed_times to record
        record.date = date
        record.front_end_id = request.history_id
        record.response = request.label
        record.elapsed_milliseconds_text = request.elapsed_milliseconds_text
        record.elapsed_milliseconds_answer = request.elapsed_milliseconds_answer

        # update leitner
        self.update_leitner(record, date, session)
        # update sm2
        self.update_sm2(record, date, session)

        # update user stats
        utc_date = date.astimezone(pytz.utc).date()
        self.update_user_stats(record, deck_id='all', utc_date=utc_date, session=session)
        if record.deck_id is not None:
            self.update_user_stats(record, deck_id=record.deck_id, utc_date=utc_date, session=session)

        # update current features
        # NOTE do this last, especially after leitner and sm2
        self.update_feature_vectors(record, date, session)

    def update_feature_vectors(self, record: Record, date: datetime, session: Session):
        v_usercard, v_user, v_card = self.get_feature_vectors(record.user_id, record.card_id, session)

        delta_usercard = None
        if v_usercard.previous_study_date is not None:
            delta_usercard = (date - v_usercard.previous_study_date).total_seconds()
        v_usercard.n_study_positive += record.response
        v_usercard.n_study_negative += (not record.response)
        v_usercard.n_study_total += 1
        v_usercard.previous_delta = delta_usercard
        v_usercard.previous_study_date = date
        v_usercard.previous_study_response = record.response

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

    def update_user_stats(self, record: Record, utc_date, deck_id: str, session: Session):
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
            session.commit()

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
            session.commit()

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

        sm2.scheduled_date = date + timedelta(days=sm2.interval)
