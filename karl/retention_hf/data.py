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
    delta_to_leitner_scheduled_date: int
    delta_to_sm2_scheduled_date: int
    # sm2_efactor: float


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
    train_retention_features, test_retention_features = [], []
    train_labels, test_labels = [], []
    n_records = len(user.records)
    n_train_records = int(n_records * 0.75)
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
        features = RetentionFeaturesSchema(
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
            delta_to_leitner_scheduled_date=delta_to_leitner_scheduled_date,
            delta_to_sm2_scheduled_date=delta_to_sm2_scheduled_date,
        )
        if ith_record < n_train_records:
            train_retention_features.append(features)
            train_labels.append(record.response)
        else:
            test_retention_features.append(features)
            test_labels.append(record.response)
    session.close()
    return train_retention_features, train_labels, test_retention_features, test_labels


class RetentionDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir: str,
        fold: str,
        tokenizer,
    ):
        cached_features_file = f'{data_dir}/cached_{fold}'
        if os.path.exists(cached_features_file):
            self.features = torch.load(cached_features_file)
            print(f"Loading features from cached file {cached_features_file}")
            return

        session = SessionLocal()

        # create card encoding
        card_ids, card_texts = [], []
        for card in session.query(Card):
            card_ids.append(card.id)
            card_texts.append(card.text)

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

        feature_keys = [
            key for key in RetentionFeaturesSchema.__fields__.keys()
            if key not in ['user_id', 'card_id']
        ]
        train_features_np = np.array([
            [x.__dict__[key] for key in feature_keys]
            for x in train_features
        ])
        test_features_np = np.array([
            [x.__dict__[key] for key in feature_keys]
            for x in test_features
        ])
        mean = train_features_np.std(axis=0)
        std = train_features_np.std(axis=0)
        retention_features = {
            'train': (train_features_np - mean) / std,
            'test': (test_features_np - mean) / std,
        }
        print('retention_features["train"].shape', retention_features['train'].shape)

        labels = {
            'train': train_labels,
            'test': test_labels,
        }
        print('len(labels["train"])', len(labels['train']))

        card_encodings = tokenizer(card_texts, truncation=True, padding=True)
        # card ids are strings, we create a mapping from card_id to index in encodings
        card_id_to_index = {cid: i for i, cid in enumerate(card_ids)}
        cards = {
            'train': [card_id_to_index[x.card_id] for x in train_features],
            'test': [card_id_to_index[x.card_id] for x in test_features],
        }

        for _fold in ['train', 'test']:
            features = []
            for i, label in enumerate(labels[_fold]):
                inputs = {k: v[cards[_fold][i]] for k, v in card_encodings.items()}
                inputs['retention_features'] = retention_features[_fold][i]
                inputs['label'] = label
                features.append(RetentionInput(**inputs))
            cached_features_file = f'{data_dir}/cached_{_fold}'
            print(f"Saving features into cached file {cached_features_file}")
            torch.save(features, cached_features_file)
            if _fold == fold:
                self.features = features

        session.close()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx) -> RetentionInput:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx]


def retention_data_collator(
    features: List[RetentionInput]
) -> Dict[str, torch.Tensor]:
    # In this method we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    first = features[0]

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if hasattr(first, "label") and first.label is not None:
        labels = torch.tensor([f.label for f in features], dtype=torch.long)
        batch = {"labels": labels}
    elif hasattr(first, "label_ids") and first.label_ids is not None:
        labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        batch = {"labels": labels}
    else:
        batch = {}

    # Handling of all other possible attributes.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in vars(first).items():
        if (
            k not in ('label', 'label_ids', 'retention_features')
            and v is not None
            and not isinstance(v, str)
        ):
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        elif k == 'retention_features':
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.float)

    return batch


if __name__ == '__main__':
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = RetentionDataset(settings.DATA_DIR, 'train', tokenizer)
    print(len(train_dataset))
    test_dataset = RetentionDataset(settings.DATA_DIR, 'train', tokenizer)
    print(len(test_dataset))
