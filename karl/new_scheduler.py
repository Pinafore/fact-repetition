#!/usr/bin/env python
# coding: utf-8

from karl.schemas import ScheduleRequest, UpdateRequest, Params, UserStatSchema, SchedulerOutputSchema
from karl.models import User, Fact, Record, UserStats


class RetentionModel:

    def predict(self, user_id, card_ids, date):
        return [
            self.predict_one(user_id, card_id, date)
            for card_id in card_ids
        ]

    def predict_one(self, user_id, card_id, date):
        r = requests.get(
            f'http://127.0.0.1:8000/api/karl/predict_recall?env=prod&user_id={user_id}&card_id={card_id}'
        )
        return float(r.text)


class KARLScheduler:

    def __init__(self):
        self.retention_model = RetentionModel()

    def schedule(
        self,
        session,
        requests: List[ScheduleRequest],
        date: datetime,
        plot=False
    ) -> SchedulerOutputSchema:
        # score cards
        # generate debug_id
        # store record with front_end_id empty @ debug_id
        # store feature vectors @ debug_id
        output_dict = {
            'order': order,
            'debug_id': debug_id,
            'scores': scores,
            'rationale': rationale,
            'facts_info': facts_info,
        }

    def update(
        self,
        session,
        requests: List[ScheduleRequest],
        date: datetime
    ) -> dict:
        # read debug_id from request
        # add front_end_id to record, response, elapsed_times to record
        # update current features
        # update user stats
        # update leitner
        # update sm2
        pass
