import os
import numpy as np
import multiprocessing
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.orm import Session

import torch
from transformers import DistilBertTokenizerFast

from karl.models import User, Card, UserCardSnapshotV2, UserSnapshotV2, CardSnapshotV2
from karl.db.session import SessionLocal, engine
from karl.config import settings
from karl.schemas import VUserCard, VUser, VCard
from .data import (  # noqa: F401
    RetentionInput,
    feature_fields,
)


def _get_user_features(
    user_id: str,
    session: Session = SessionLocal()
):
    user = session.query(User).get(user_id)
    record_ids, features, labels = [], [], []
    for record in user.study_records:
        if record.label is None:
            continue
        v_user = session.query(UserSnapshotV2).get(record.id)
        v_card = session.query(CardSnapshotV2).get(record.id)
        v_usercard = session.query(UserCardSnapshotV2).get(record.id)
        if v_user is None or v_card is None or v_usercard is None:
            continue
        v_user = VUser(**v_user.__dict__)
        v_card = VCard(**v_card.__dict__)
        v_usercard = VUserCard(**v_usercard.__dict__)

        elapsed_milliseconds = record.elapsed_milliseconds_text + record.elapsed_milliseconds_answer
        features.append(vectors_to_features(v_usercard, v_user, v_card, record.date, record.card.text, record.card.answer, elapsed_milliseconds))
        labels.append(record.label)
        record_ids.append(record.id)
    session.close()
    return record_ids, features, labels


def get_retention_features_df(overwrite: bool = False):
    df_path = f'{settings.CODE_DIR}/retention_features_phase2.h5'
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
        cnt_non_empty_user = 0
        for user in session.query(User):
            if not user.id.isdigit() or len(user.study_records) == 0:
                continue
            cnt_non_empty_user += 1
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
            row['label'] = label
            row['record_id'] = record_id
            row['deck_id'] = card_to_deck_id.get(row['card_id'], None)
            row['deck_name'] = card_to_deck_name.get(row['card_id'], None)
            df.append(row)

        df = pd.DataFrame(df)
        df['n_minutes_spent'] = df.groupby('user_id')['elapsed_milliseconds'].cumsum() // 60000
        df.to_hdf(df_path, key='df', mode='w')
        session.close()

    return df


if __name__ == '__main__':
    data_dir = f'{settings.DATA_DIR}/retention_phase2'
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # gather features
    print('gather features')
    df_all = get_retention_features_df()

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
            example['label'] = int(row.label)
            if foold in ndarray_by_fold:
                example['retention_features'] = ndarray_by_fold[foold][i]
            inputs[foold].append(RetentionInput(**example))
        cached_inputs_file = f'{data_dir}/cached_{foold}'
        print(f"Saving features into cached file {cached_inputs_file}")
        torch.save(inputs[foold], cached_inputs_file)
