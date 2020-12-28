import os
import json
import dataclasses
import multiprocessing
import numpy as np
from pydantic import BaseModel
from typing import Optional, List, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.orm import Session

import torch
from transformers import DistilBertTokenizerFast

from karl.db.session import SessionLocal, engine
from karl.config import settings
from karl.models import User, Card, UserFeatureVector, CardFeatureVector, UserCardFeatureVector


class RetentionFeaturesSchema(BaseModel):
    user_id: str
    card_id: str
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


@dataclass(frozen=True)
class RetentionInput:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    retention_features: Optional[List[float]] = None
    label: Optional[int] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def _get_user_features(
    user_id: str,
    session: Session = SessionLocal()
):
    user = session.query(User).get(user_id)
    features, labels = [], []
    for ith_record, record in enumerate(user.records):
        if record.response is None:
            continue
        v_user = session.query(UserFeatureVector).get(record.id)
        v_card = session.query(CardFeatureVector).get(record.id)
        v_usercard = session.query(UserCardFeatureVector).get(record.id)
        if v_user is None or v_card is None or v_usercard is None:
            continue
        if v_usercard.previous_study_date is not None:
            usercard_delta = (record.date - v_usercard.previous_study_date).total_seconds()
        else:
            usercard_delta = 0
        if v_usercard.leitner_scheduled_date is not None:
            delta_to_leitner_scheduled_date = (v_usercard.leitner_scheduled_date - record.date).total_seconds()
        else:
            delta_to_leitner_scheduled_date = 0
        if v_usercard.sm2_scheduled_date is not None:
            delta_to_sm2_scheduled_date = (v_usercard.sm2_scheduled_date - record.date).total_seconds()
        else:
            delta_to_sm2_scheduled_date = 0
        labels.append(record.response)
        features.append(RetentionFeaturesSchema(
            user_id=record.user_id,
            card_id=record.card_id,
            is_new_fact=record.is_new_fact,
            user_n_study_positive=v_user.n_study_positive,
            user_n_study_negative=v_user.n_study_negative,
            user_n_study_total=v_user.n_study_total,
            card_n_study_positive=v_card.n_study_positive,
            card_n_study_negative=v_card.n_study_negative,
            card_n_study_total=v_card.n_study_total,
            usercard_n_study_positive=v_usercard.n_study_positive,
            usercard_n_study_negative=v_usercard.n_study_negative,
            usercard_n_study_total=v_usercard.n_study_total,
            acc_user=0 if v_user.n_study_total == 0 else v_user.n_study_positive / v_user.n_study_total,
            acc_card=0 if v_card.n_study_total == 0 else v_card.n_study_positive / v_card.n_study_total,
            acc_usercard=0 if v_usercard.n_study_total == 0 else v_usercard.n_study_positive / v_usercard.n_study_total,
            usercard_delta=usercard_delta or 0,
            usercard_delta_previous=v_usercard.previous_delta or 0,
            usercard_previous_study_response=v_usercard.previous_study_response or False,
            leitner_box=v_usercard.leitner_box or 0,
            sm2_efactor=v_usercard.sm2_efactor or 0,
            sm2_interval=v_usercard.sm2_interval or 0,
            sm2_repetition=v_usercard.sm2_repetition or 0,
            delta_to_leitner_scheduled_date=delta_to_leitner_scheduled_date,
            delta_to_sm2_scheduled_date=delta_to_sm2_scheduled_date,
        ))
    session.close()
    return features, labels


class RetentionDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir: str,
        fold: str,
        tokenizer,
    ):
        if os.path.exists(f'{data_dir}/cached_train_new_card') and \
                os.path.exists(f'{data_dir}/cached_train_old_card') and \
                os.path.exists(f'{data_dir}/cached_test_new_card') and \
                os.path.exists(f'{data_dir}/cached_test_old_card'):
            inputs = {
                'train_new_card': torch.load(f'{data_dir}/cached_train_new_card'),
                'train_old_card': torch.load(f'{data_dir}/cached_train_old_card'),
                'test_new_card': torch.load(f'{data_dir}/cached_test_new_card'),
                'test_old_card': torch.load(f'{data_dir}/cached_test_old_card'),
            }
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

            train_features, test_features = [], []
            train_labels, test_labels = [], []
            for future in futures:
                f1, f2, f3, f4 = future.result()
                train_features.extend(f1)
                train_labels.extend(f2)
                test_features.extend(f3)
                test_labels.extend(f4)

            folds = [
                'train_new_card',
                'train_old_card',
                'test_new_card',
                'test_old_card',
            ]
            features = {f: [] for f in folds}
            labels = {f: [] for f in folds}
            card_ids = {f: [] for f in folds}
            feature_keys = [
                key for key in RetentionFeaturesSchema.__fields__.keys()
                if key not in ['user_id', 'card_id']
            ]
            for x, y in zip(train_features, train_labels):
                f = 'train_new_card' if x.is_new_fact else 'train_old_card'
                features[f].append([x.__dict__[key] for key in feature_keys])
                labels[f].append(y)
                card_ids[f].append(x.card_id)
            for x, y in zip(test_features, test_labels):
                f = 'test_new_card' if x.is_new_fact else 'test_old_card'
                features[f].append([x.__dict__[key] for key in feature_keys])
                labels[f].append(y)
                card_ids[f].append(x.card_id)

            # normalize
            for f in folds:
                features[f] = np.array(features[f])
            mean_new = features['train_new_card'].mean(axis=0)
            std_new = features['train_new_card'].std(axis=0)
            mean_old = features['train_old_card'].mean(axis=0)
            std_old = features['train_old_card'].std(axis=0)
            features['train_new_card'] = (features['train_new_card'] - mean_new) / std_new
            features['test_new_card'] = (features['test_new_card'] - mean_new) / std_new
            features['train_old_card'] = (features['train_old_card'] - mean_old) / std_old
            features['test_old_card'] = (features['test_old_card'] - mean_old) / std_old
            print('features["train_new_card"].shape', features['train_new_card'].shape)
            print('features["train_old_card"].shape', features['train_old_card'].shape)

            # create card encoding
            card_texts = []
            card_id_to_index = {}
            for card in session.query(Card):
                card_id_to_index[card.id] = len(card_id_to_index)
                card_texts.append(card.text)
            card_encodings = tokenizer(card_texts, truncation=True, padding=True)
            cards = {f: [card_id_to_index[card_id] for card_id in card_ids[f]] for f in folds}

            # put everything together and cache
            for f in folds:
                inputs = []
                for i, label in enumerate(labels[f]):
                    example = {k: v[cards[f][i]] for k, v in card_encodings.items()}
                    example['retention_features'] = features[f][i]
                    example['label'] = label
                    inputs.append(RetentionInput(**example))
                cached_inputs_file = f'{data_dir}/cached_{f}'
                print(f"Saving features into cached file {cached_inputs_file}")
                torch.save(inputs, cached_inputs_file)
            session.close()
        self.inputs = inputs[fold]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx) -> RetentionInput:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.inputs[idx]


def retention_data_collator(
    features: List[RetentionInput]
) -> Dict[str, torch.Tensor]:
    # In this method we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    first = features[0]

    if hasattr(first, "label") and first.label is not None:
        labels = torch.tensor([f.label for f in features], dtype=torch.long)
        batch = {"labels": labels}
    else:
        batch = {}

    # Handling of all other possible attributes.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in vars(first).items():
        if (
            k not in ('label', 'retention_features')
            and v is not None
            and not isinstance(v, str)
        ):
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        elif k == 'retention_features':
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.float)

    return batch


if __name__ == '__main__':
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = RetentionDataset(
        data_dir=settings.DATA_DIR,
        fold='train',
        tokenizer=tokenizer
    )
    print(len(train_dataset))
