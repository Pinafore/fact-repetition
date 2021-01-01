import os
import json
import pytz
import dataclasses
import multiprocessing
import numpy as np
from datetime import date, datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.orm import Session

import torch
from transformers import DistilBertTokenizerFast

from karl.db.session import SessionLocal, engine
from karl.config import settings
from karl.models import User, Card, \
    UserFeatureVector, CardFeatureVector, UserCardFeatureVector, \
    CurrUserFeatureVector, CurrCardFeatureVector, CurrUserCardFeatureVector


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
    utc_date: date


feature_fields = [
    field_name for field_name, field_info in RetentionFeaturesSchema.__fields__.items()
    if field_info.type_ in [int, float, bool]
    and field_name != 'is_new_fact'
]


def vectors_to_features(
    v_usercard: Union[UserCardFeatureVector, CurrUserCardFeatureVector],
    v_user: Union[UserFeatureVector, CurrUserFeatureVector],
    v_card: Union[CardFeatureVector, CurrCardFeatureVector],
    date: datetime,
    card_text: str,
    elapsed_milliseconds: int = 0,
) -> RetentionFeaturesSchema:
    # NOTE this function should be a replica of `_get_user_features` in `retention_hf.data`
    if v_usercard.previous_study_date is not None:
        usercard_delta = (date - v_usercard.previous_study_date).total_seconds()
    else:
        usercard_delta = 0
    if v_usercard.leitner_scheduled_date is not None:
        delta_to_leitner_scheduled_date = (v_usercard.leitner_scheduled_date - date).total_seconds()
    else:
        delta_to_leitner_scheduled_date = 0
    if v_usercard.sm2_scheduled_date is not None:
        delta_to_sm2_scheduled_date = (v_usercard.sm2_scheduled_date - date).total_seconds()
    else:
        delta_to_sm2_scheduled_date = 0
    return RetentionFeaturesSchema(
        user_id=v_usercard.user_id,
        card_id=v_usercard.card_id,
        card_text=card_text,
        is_new_fact=(v_usercard.correct_on_first_try is None),
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
        repetition_model=json.loads(v_user.parameters)['repetition_model'],
        correct_on_first_try=v_usercard.correct_on_first_try or False,
        elapsed_milliseconds=elapsed_milliseconds,
        utc_date=date.astimezone(pytz.utc).date(),
    )


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
        elapsed_milliseconds = record.elapsed_milliseconds_text + record.elapsed_milliseconds_answer
        features.append(vectors_to_features(v_usercard, v_user, v_card, record.date, record.card.text, elapsed_milliseconds))
        labels.append(record.response)
    session.close()
    return features, labels


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

            folds = [
                'train_new_card',
                'train_old_card',
                'test_new_card',
                'test_old_card',
            ]
            features = {f: [] for f in folds}
            labels = {f: [] for f in folds}
            card_ids = {f: [] for f in folds}

            for future in futures:
                user_features, user_labels = future.result()
                user_features_new, user_features_old = [], []
                user_labels_new, user_labels_old = [], []
                user_card_ids_new, user_card_ids_old = [], []
                for x, y in zip(user_features, user_labels):
                    if x.is_new_fact:
                        # user_features_new.append([x.__dict__[field] for field in feature_fields if field != 'correct_on_first_try'])
                        user_labels_new.append(y)
                        user_card_ids_new.append(x.card_id)
                    else:
                        user_features_old.append([x.__dict__[field] for field in feature_fields])
                        user_labels_old.append(y)
                        user_card_ids_old.append(x.card_id)
                # the first 3/4 of examples goes to train, the rest goes to test
                n_train_new = int(len(user_labels_new) * 0.75)
                n_train_old = int(len(user_labels_old) * 0.75)
                features['train_new_card'].extend(user_features_new[:n_train_new])
                features['train_old_card'].extend(user_features_old[:n_train_old])
                features['test_new_card'].extend(user_features_new[n_train_new:])
                features['test_old_card'].extend(user_features_old[n_train_old:])
                labels['train_new_card'].extend(user_labels_new[:n_train_new])
                labels['train_old_card'].extend(user_labels_old[:n_train_old])
                labels['test_new_card'].extend(user_labels_new[n_train_new:])
                labels['test_old_card'].extend(user_labels_old[n_train_old:])
                card_ids['train_new_card'].extend(user_card_ids_new[:n_train_new])
                card_ids['train_old_card'].extend(user_card_ids_old[:n_train_old])
                card_ids['test_new_card'].extend(user_card_ids_new[n_train_new:])
                card_ids['test_old_card'].extend(user_card_ids_old[n_train_old:])

            # normalize
            for f in folds:
                features[f] = np.array(features[f])
            # mean_new = features['train_new_card'].mean(axis=0)
            # std_new = features['train_new_card'].std(axis=0)
            mean_old = features['train_old_card'].mean(axis=0)
            std_old = features['train_old_card'].std(axis=0)
            # features['train_new_card'] = (features['train_new_card'] - mean_new) / std_new
            # features['test_new_card'] = (features['test_new_card'] - mean_new) / std_new
            features['train_old_card'] = (features['train_old_card'] - mean_old) / std_old
            features['test_old_card'] = (features['test_old_card'] - mean_old) / std_old
            print('features["train_new_card"].shape', features['train_new_card'].shape)
            print('features["train_old_card"].shape', features['train_old_card'].shape)

            for i, field in enumerate(feature_fields):
                print(field, '%.2f' % mean_old[i], '%.2f' % std_old[i])

            self.mean = mean_old
            self.std = std_old
            torch.save(self.mean, f'{data_dir}/cached_mean')
            torch.save(self.std, f'{data_dir}/cached_std')

            # create card encoding
            card_texts = []
            card_id_to_index = {}
            for card in session.query(Card):
                card_id_to_index[card.id] = len(card_id_to_index)
                card_texts.append(card.text)
            card_encodings = tokenizer(card_texts, truncation=True, padding=True)
            cards = {f: [card_id_to_index[card_id] for card_id in card_ids[f]] for f in folds}

            # put everything together and cache
            inputs = {}
            for f in folds:
                inputs[f] = []
                for i, label in enumerate(labels[f]):
                    example = {k: v[cards[f][i]] for k, v in card_encodings.items()}
                    example['label'] = label
                    if 'new' not in f:
                        example['retention_features'] = features[f][i]
                    inputs[f].append(RetentionInput(**example))
                cached_inputs_file = f'{data_dir}/cached_{f}'
                print(f"Saving features into cached file {cached_inputs_file}")
                torch.save(inputs[f], cached_inputs_file)
            session.close()
        self.inputs = inputs[fold]
        self.mean = torch.load(f'{data_dir}/cached_mean')
        self.std = torch.load(f'{data_dir}/cached_std')

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
        fold='train_new_card',
        tokenizer=tokenizer
    )
    print(len(train_dataset))
