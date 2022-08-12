#!/usr/bin/env python
# coding: utf-8

import os
import json
import pytz
import dataclasses
import multiprocessing
import pandas as pd
import numpy as np
from datetime import date, datetime
from pydantic import BaseModel
from typing import Optional, List, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.orm import Session

import torch
from transformers import DistilBertTokenizerFast

from karl.db.session import SessionLocal, engine
from karl.config import settings
from karl.models import User, Card, UserCardSnapshot, UserSnapshot, CardSnapshot
from karl.schemas import VUserCard, VUser, VCard


class RetentionFeaturesSchema(BaseModel):
    '''
    Essentially all you can tell about an individual study record.
    '''
    user_id: str
    card_id: str
    card_text: str
    is_new_fact: bool
    user_n_study_positive: int
    user_n_study_negative: int
    user_n_study_total: int
    card_n_study_positive: int
    card_n_study_negative: int
    card_n_study_total: int
    usercard_n_study_positive: int
    usercard_n_study_negative: int
    usercard_n_study_total: int
    acc_user: float
    acc_card: float
    acc_usercard: float
    usercard_delta: int
    usercard_delta_previous: int
    usercard_previous_study_response: bool
    leitner_box: int
    sm2_efactor: float
    sm2_interval: float
    sm2_repetition: int
    delta_to_leitner_scheduled_date: int
    delta_to_sm2_scheduled_date: int
    repetition_model: str
    elapsed_milliseconds: int
    correct_on_first_try: Optional[bool]
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
    elapsed_milliseconds: int = 0,
) -> RetentionFeaturesSchema:
    if v_usercard.previous_study_date is not None:
        usercard_delta = (date - v_usercard.previous_study_date).total_seconds() // 3600
    else:
        usercard_delta = 0
    if v_usercard.leitner_scheduled_date is not None:
        delta_to_leitner_scheduled_date = (v_usercard.leitner_scheduled_date - date).total_seconds() // 3600
    else:
        delta_to_leitner_scheduled_date = 0
    if v_usercard.sm2_scheduled_date is not None:
        delta_to_sm2_scheduled_date = (v_usercard.sm2_scheduled_date - date).total_seconds() // 3600
    else:
        delta_to_sm2_scheduled_date = 0
    return RetentionFeaturesSchema(
        user_id=v_usercard.user_id,
        card_id=v_usercard.card_id,
        card_text=card_text,
        is_new_fact=(v_usercard.correct_on_first_try is None),
        user_n_study_positive=v_user.count_positive,
        user_n_study_negative=v_user.count_negative,
        user_n_study_total=v_user.count,
        card_n_study_positive=v_card.count_positive,
        card_n_study_negative=v_card.count_negative,
        card_n_study_total=v_card.count,
        usercard_n_study_positive=v_usercard.count_positive,
        usercard_n_study_negative=v_usercard.count_negative,
        usercard_n_study_total=v_usercard.count,
        acc_user=0 if v_user.count == 0 else v_user.count_positive / v_user.count,
        acc_card=0 if v_card.count == 0 else v_card.count_positive / v_card.count,
        acc_usercard=0 if v_usercard.count == 0 else v_usercard.count_positive / v_usercard.count,
        usercard_delta=usercard_delta or 0,
        usercard_delta_previous=v_usercard.previous_delta or 0,
        usercard_previous_study_response=v_usercard.previous_study_response or False,
        leitner_box=v_usercard.leitner_box or 0,
        sm2_efactor=v_usercard.sm2_efactor or 0,
        sm2_interval=v_usercard.sm2_interval or 0,
        sm2_repetition=v_usercard.sm2_repetition or 0,
        delta_to_leitner_scheduled_date=delta_to_leitner_scheduled_date,
        delta_to_sm2_scheduled_date=delta_to_sm2_scheduled_date,
        repetition_model=json.loads(v_user.parameters)['repetition_model'],
        correct_on_first_try=v_usercard.correct_on_first_try or False,
        elapsed_milliseconds=elapsed_milliseconds,
        utc_datetime=date.astimezone(pytz.utc),
        utc_date=date.astimezone(pytz.utc).date(),
    )


def _get_user_features(
    user_id: str,
    session: Session = SessionLocal()
):
    user = session.query(User).get(user_id)
    record_ids, features, labels = [], [], []
    for record in user.records:
        if record.response is None:
            continue
        v_user = session.query(UserSnapshot).get(record.id)
        v_card = session.query(CardSnapshot).get(record.id)
        v_usercard = session.query(UserCardSnapshot).get(record.id)
        if v_user is None or v_card is None or v_usercard is None:
            continue
        v_user = VUser(**v_user.__dict__)
        v_card = VCard(**v_card.__dict__)
        v_usercard = VUserCard(**v_usercard.__dict__)
        elapsed_milliseconds = record.elapsed_milliseconds_text + record.elapsed_milliseconds_answer
        features.append(vectors_to_features(v_usercard, v_user, v_card, record.date, record.card.text, elapsed_milliseconds))
        labels.append(record.response)
        record_ids.append(record.id)
    session.close()
    return record_ids, features, labels


def get_retention_features_df(overwrite: bool = False):
    df_path = f'{settings.DATA_DIR}/retention_features.h5'
    if not overwrite and os.path.exists(df_path):
        df = pd.read_hdf(df_path, 'df')
    else:
        session = SessionLocal()
        # gather features
        futures = []
        executor = ProcessPoolExecutor(
            mp_context=multiprocessing.get_context(settings.MP_CONTEXT),
            initializer=engine.dispose,
        )
        for user in session.query(User):
            if not user.id.isdigit() or len(user.records) == 0:
                continue
            futures.append(executor.submit(_get_user_features, user.id))

        record_ids, features, labels = [], [], []
        for future in futures:
            f0, f1, f2 = future.result()
            record_ids.extend(f0)
            features.extend(f1)
            labels.extend(f2)

        card_to_deck_id = {}
        card_to_deck_name = {}
        for card in session.query(Card):
            card_to_deck_id[card.id] = card.deck_id
            card_to_deck_name[card.id] = card.deck_name

        df = []
        for record_id, feature, label in zip(record_ids, features, labels):
            row = feature.__dict__
            row['response'] = label
            row['record_id'] = record_id
            row['deck_id'] = card_to_deck_id.get(row['card_id'], None)
            row['deck_name'] = card_to_deck_name.get(row['card_id'], None)
            df.append(row)

        df = pd.DataFrame(df)
        df['n_minutes_spent'] = df.groupby('user_id')['elapsed_milliseconds'].cumsum() // 60000
        df.to_hdf(df_path, key='df', mode='w')
        session.close()

    return df


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
            # gather features
            print('gather features')
            df_all = get_retention_features_df(overwrite_retention_features_df)

            # separate new and old cards
            df_new_card = df_all[df_all.is_new_fact == True]  # noqa: E712
            df_old_card = df_all[df_all.is_new_fact == False]  # noqa: E712

            # within each fold, take the first 75% as training data and the rest as test data
            df_by_fold = {
                'train_new_card': df_new_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[:int(x.user_id.size * 0.75)]),
                'test_new_card': df_new_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[int(x.user_id.size * 0.75):]),
                'train_old_card': df_old_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[:int(x.user_id.size * 0.75)]),
                'test_old_card': df_old_card.groupby('user_id', as_index=False).apply(lambda x: x.iloc[int(x.user_id.size * 0.75):]),
            }

            # create token encodings
            print('token encodings')
            encodings_by_fold = {
                fold: tokenizer(df.card_text.tolist(), truncation=True, padding=True)
                for fold, df in df_by_fold.items()
            }

            # collect manually crafted features
            print('normalize')
            ndarray_by_fold = {
                fold: df[feature_fields].to_numpy(dtype=np.float64)
                for fold, df in df_by_fold.items()
                if fold in ['train_old_card', 'test_old_card']
            }

            # normalize manually crafted features
            mean = np.mean(ndarray_by_fold['train_old_card'], axis=0)
            std = np.std(ndarray_by_fold['train_old_card'], axis=0)
            torch.save(mean, f'{data_dir}/cached_mean')
            torch.save(std, f'{data_dir}/cached_std')
            ndarray_by_fold['train_old_card'] = (ndarray_by_fold['train_old_card'] - mean) / std
            ndarray_by_fold['test_old_card'] = (ndarray_by_fold['test_old_card'] - mean) / std
            for i, field in enumerate(feature_fields):
                print(field, '%.2f' % mean[i], '%.2f' % std[i])

            # put everything together and save in cache
            inputs = {}
            for foold, df in df_by_fold.items():
                inputs[foold] = []
                for i, row in enumerate(df.itertuples(index=False)):
                    example = {k: v[i] for k, v in encodings_by_fold[foold].items()}
                    example['label'] = int(row.response)
                    if foold in ndarray_by_fold:
                        example['retention_features'] = ndarray_by_fold[foold][i]
                    inputs[foold].append(RetentionInput(**example))
                cached_inputs_file = f'{data_dir}/cached_{foold}'
                print(f"Saving features into cached file {cached_inputs_file}")
                torch.save(inputs[foold], cached_inputs_file)

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
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # train_dataset = RetentionDataset(
    #     data_dir=settings.DATA_DIR,
    #     fold='train_new_card',
    #     tokenizer=tokenizer
    # )
    # print(len(train_dataset))
    df = get_retention_features_df()
