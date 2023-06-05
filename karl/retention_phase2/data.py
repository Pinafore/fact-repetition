#!/usr/bin/env python
# coding: utf-8

import os
import json
import pytz
import dataclasses
from datetime import date, datetime
from pydantic import BaseModel
from typing import Optional, List, Dict
from dataclasses import dataclass

import torch
from transformers import DistilBertTokenizerFast

from karl.config import settings
from karl.schemas import VUserCard, VUser, VCard


class RetentionFeaturesSchema(BaseModel):
    '''
    Essentially all you can tell about an individual study record.
    '''
    user_id: str
    card_id: str
    card_text: str
    answer: str
    is_new_fact: bool
    count_positive_user: int
    count_negative_user: int
    count_user: int
    count_positive_session_user: int
    count_negative_session_user: int
    count_session_user: int
    count_positive_card: int
    count_negative_card: int
    count_positive_session_card: int
    count_negative_session_card: int
    count_session_card: int
    count_positive_usercard: int
    count_negative_usercard: int
    count_usercard: int
    # no need for usercard_count_session since it's the same as card_count_session
    acc_user: float
    acc_card: float
    acc_usercard: float
    acc_session_user: float
    acc_session_card: float
    delta: int
    delta_previous: int
    usercard_previous_study_response: bool
    usercard_delta_session: int
    usercard_delta_previous_session: int
    usercard_previous_study_response_session: bool
    leitner_box: int
    sm2_efactor: float
    sm2_interval: float
    sm2_repetition: int
    delta_to_leitner_scheduled_date: int
    delta_to_sm2_scheduled_date: int
    repetition_model: str
    elapsed_milliseconds: int
    correct_on_first_try: bool
    correct_on_first_try_session: bool
    utc_datetime: datetime
    utc_date: date


feature_fields = [
    field_name for field_name, field_info in RetentionFeaturesSchema.__fields__.items()
    if field_info.type_ in [int, float, bool]
    and field_name != 'is_new_fact'
]


def vectors_to_features(
    v_usercard: VUserCard,
    v_user: VUser,
    v_card: VCard,
    date: datetime,
    card_text: str,
    card_answer: str,
    elapsed_milliseconds: int = 0,
) -> RetentionFeaturesSchema:
    if v_usercard.previous_study_date is not None:
        delta = (date - v_usercard.previous_study_date).total_seconds() // 3600  # hrs
    else:
        delta = 0

    if v_usercard.previous_study_date_session is not None:
        delta_session = (date - v_usercard.previous_study_date_session).total_seconds() // 60  # mins
    else:
        delta_session = 0

    if v_usercard.previous_delta is not None:
        previous_delta = v_usercard.previous_delta // 3600  # hrs
    else:
        previous_delta = 0

    if v_usercard.previous_delta_session is not None:
        previous_delta_session = v_usercard.previous_delta_session // 60  # hrs
    else:
        previous_delta_session = 0

    if v_usercard.leitner_scheduled_date is not None:
        delta_to_leitner_scheduled_date = (v_usercard.leitner_scheduled_date - date).total_seconds() // 3600  # hrs
    else:
        delta_to_leitner_scheduled_date = 0

    if v_usercard.sm2_scheduled_date is not None:
        delta_to_sm2_scheduled_date = (v_usercard.sm2_scheduled_date - date).total_seconds() // 3600  # hrs
    else:
        delta_to_sm2_scheduled_date = 0

    return RetentionFeaturesSchema(
        user_id=v_usercard.user_id,
        card_id=v_usercard.card_id,
        card_text=card_text,
        answer=card_answer,
        is_new_fact=(v_usercard.correct_on_first_try is None),
        count_positive_user=v_user.count_positive,
        count_negative_user=v_user.count_negative,
        count_user=v_user.count,
        count_positive_session_user=v_user.count_positive_session,
        count_negative_session_user=v_user.count_negative_session,
        count_session_user=v_user.count_session,
        count_positive_card=v_card.count_positive,
        count_negative_card=v_card.count_negative,
        count_positive_session_card=v_usercard.count_positive_session,
        count_negative_session_card=v_usercard.count_negative_session,
        count_session_card=v_usercard.count_session,
        count_positive_usercard=v_usercard.count_positive,
        count_negative_usercard=v_usercard.count_negative,
        count_usercard=v_usercard.count,
        acc_user=0 if v_user.count == 0 else v_user.count_positive / v_user.count,
        acc_card=0 if v_card.count == 0 else v_card.count_positive / v_card.count,
        acc_usercard=0 if v_usercard.count == 0 else v_usercard.count_positive / v_usercard.count,
        acc_session_user=0 if v_user.count_session == 0 else v_user.count_positive_session / v_user.count_session,
        acc_session_card=0 if v_usercard.count_session == 0 else v_usercard.count_positive_session / v_usercard.count_session,
        delta=delta,  # hrs
        delta_previous=previous_delta,  # hrs
        usercard_previous_study_response=v_usercard.previous_study_response or False,
        usercard_delta_session=delta_session,  # mins
        usercard_delta_previous_session=previous_delta_session,  # mins
        usercard_previous_study_response_session=v_usercard.previous_study_response_session or False,
        leitner_box=v_usercard.leitner_box or 0,
        sm2_efactor=v_usercard.sm2_efactor or 0,
        sm2_interval=v_usercard.sm2_interval or 0,
        sm2_repetition=v_usercard.sm2_repetition or 0,
        delta_to_leitner_scheduled_date=delta_to_leitner_scheduled_date,
        delta_to_sm2_scheduled_date=delta_to_sm2_scheduled_date,
        repetition_model=json.loads(v_user.parameters)['repetition_model'],
        correct_on_first_try=v_usercard.correct_on_first_try or False,
        correct_on_first_try_session=v_usercard.correct_on_first_try or False,  # TODO
        elapsed_milliseconds=elapsed_milliseconds,
        utc_datetime=date.astimezone(pytz.utc),
        utc_date=date.astimezone(pytz.utc).date(),
    )


@dataclass(frozen=True)
class RetentionInput:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    retention_features: Optional[List[float]] = None
    label: Optional[int] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class RetentionDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir: str,
        fold: str,
        tokenizer,
        overwrite_cached_data: bool = False,
        overwrite_retention_features_df: bool = False,
    ):
        if (
            os.path.exists(f'{data_dir}/cached_train_new_card')
            and os.path.exists(f'{data_dir}/cached_train_old_card')
            and os.path.exists(f'{data_dir}/cached_test_new_card')
            and os.path.exists(f'{data_dir}/cached_test_old_card')
        ) and not overwrite_cached_data:
            inputs = {
                'train_new_card': torch.load(f'{data_dir}/cached_train_new_card'),
                'train_old_card': torch.load(f'{data_dir}/cached_train_old_card'),
                'test_new_card': torch.load(f'{data_dir}/cached_test_new_card'),
                'test_old_card': torch.load(f'{data_dir}/cached_test_old_card'),
            }
        else:
            # TODO
            pass

        self.inputs = inputs[fold]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx) -> RetentionInput:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.inputs[idx]


def retention_data_collator(
    inputs: List[RetentionInput]
) -> Dict[str, torch.Tensor]:
    # In this method we'll make the assumption that all `inputs` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    batch = {}
    first = inputs[0]

    if hasattr(first, 'label') and first.label is not None:
        labels = torch.tensor([f.label for f in inputs], dtype=torch.float)
        batch['labels'] = labels

    if hasattr(first, 'retention_features') and first.retention_features is not None:
        retention_features = torch.tensor([f.retention_features for f in inputs], dtype=torch.float)
        batch['retention_features'] = retention_features

    # Handling of all other possible attributes.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in vars(first).items():
        if (
            k not in ('label', 'retention_features')
            and v is not None
            and not isinstance(v, str)
        ):
            batch[k] = torch.tensor([getattr(f, k) for f in inputs], dtype=torch.long)

    return batch


if __name__ == '__main__':
    data_dir = f'{settings.DATA_DIR}/retention_phase2'
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = RetentionDataset(
        data_dir=data_dir,
        fold='train_new_card',
        tokenizer=tokenizer
    )
    print(len(train_dataset))
    # df = get_retention_features_df()
