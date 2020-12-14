# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from transformers.data.metrics import simple_accuracy
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    DistilBertTokenizer,
    EvalPrediction,
)

from karl.retention_utils import (
    RetentionDataArguments,
    RetentionDataset,
    RetentionDistilBertConfig,
    RetentionInputFeatures,
    DistilBertRetentionModel,
    retention_data_collator,
)

from karl.new_util import User, Fact

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    retention_feature_size: Optional[int] = field(
        default=None, metadata={"help": "Size of extra retention features"}
    )


def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}


def train():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments,
                               TrainingArguments,
                               RetentionDataArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = RetentionDistilBertConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,
        retention_feature_size=model_args.retention_feature_size,
        finetuning_task='retention',
        cache_dir=model_args.cache_dir,
    )
    tokenizer = DistilBertTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = DistilBertRetentionModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    # labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids, labels=labels)
    # loss, logits = outputs[:2]

    train_dataset = (
        RetentionDataset(data_args,
                         fold='train',
                         tokenizer=tokenizer)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        RetentionDataset(data_args,
                         fold='test',
                         tokenizer=tokenizer)
        if training_args.do_eval
        else None
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=retention_data_collator,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(
            training_args.output_dir, "eval_results_retention.txt"
        )
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results retention *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))


class HFRetentionModel:

    def __init__(self, config_dir: str = 'configs/hf_config.json'):
        parser = HfArgumentParser((ModelArguments,
                                   TrainingArguments,
                                   RetentionDataArguments))

        model_args, training_args, data_args = parser.parse_json_file(
            json_file='configs/hf_config.json')

        self.tokenizer = DistilBertTokenizer.from_pretrained(training_args.output_dir)
        self.model = DistilBertRetentionModel.from_pretrained(training_args.output_dir)
        self.model = self.model.to(training_args.device)
        self.model.eval()

        self.data_args = data_args

    def predict(self, user: User, facts: List[Fact], date=None) -> np.ndarray:
        if date is None:
            date = datetime.now()
        # user_count_correct
        # user_count_wrong
        # user_count_total
        # user_average_overall_accuracy
        # user_average_question_accuracy
        # user_previous_result
        # user_gap_from_previous
        # question_average_overall_accuracy
        # question_count_total
        # question_count_correct
        # question_count_wrong
        # bias
        retention_features_list = []
        question_text_list = []
        for fact in facts:
            uq_correct = user.count_correct_before.get(fact.fact_id, 0)
            uq_wrong = user.count_wrong_before.get(fact.fact_id, 0)
            uq_total = uq_correct + uq_wrong
            if fact.fact_id in user.previous_study:
                prev_date, prev_response = user.previous_study[fact.fact_id]
            else:
                prev_date = date
            retention_features_list.append([
                uq_correct,  # user_count_correct
                uq_wrong,  # user_count_wrong
                uq_total,  # user_count_total
                0 if len(user.results) == 0 else np.mean(user.results),  # user_average_oveuracy
                0 if uq_total == 0 else uq_correct / uq_total,  # user_average_question_acc
                0 if len(user.results) == 0 else user.results[-1],  # user_previous_result
                (date - prev_date).seconds / (60 * 60),  # user_gap_from_previous
                0 if len(fact.results) == 0 else np.mean(fact.results),  # question_average_accuracy
                len(fact.results),  # question_count_total
                sum(fact.results),  # question_count_correct
                len(fact.results) - sum(fact.results),  # question_count_wrong
                1  # bias
            ])
            question_text_list.append(fact.text)

        if self.data_args.max_seq_length is None:
            max_length = self.tokenizer.max_len
        else:
            max_length = self.data_args.max_seq_length

        batch_size = len(question_text_list)
        batch_encoding = self.tokenizer.batch_encode_plus(
            question_text_list,
            max_length=max_length,
            pad_to_max_length=True,
            truncation=True,
        )

        features = []
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in batch_encoding.items()}
            # x = ((xs[i] - mean) / std).astype(float)
            # inputs['retention_features'] = x.tolist()
            inputs['retention_features'] = retention_features_list[i]
            features.append(RetentionInputFeatures(**inputs))

        inputs = retention_data_collator(features)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            predictions = self.model(**inputs)
        scores = F.softmax(predictions[0], dim=-1).detach().cpu().numpy()
        scores = scores[:, 1]  # take the probability of recall
        return scores

    def predict_one(self, user: User, fact: Fact) -> float:
        '''recall probability'''
        return self.predict(user, [fact])[0]  # batch size is 1


def test_wrapper():
    model = HFRetentionModel('configs/hf_configs.json')

    user = User(
        user_id='user 1',
        previous_study={'fact 1': (datetime.now(), True)},
        leitner_box={'fact 1': 2},
        leitner_scheduled_date={'fact 2': datetime.now()},
        sm2_efactor={'fact 1': 0.5},
        sm2_interval={'fact 1': 6},
        sm2_repetition={'fact 1': 10},
        sm2_scheduled_date={'fact 2': datetime.now()},
        results=[True, False, True],
        count_correct_before={'fact 1': 1},
        count_wrong_before={'fact 1': 3}
    )

    fact = Fact(
        fact_id='fact 1',
        text='This is the question text',
        answer='Answer Text III',
        category='WORLD',
        qrep=np.array([1, 2, 3, 4]),
        skill=np.array([0.1, 0.2, 0.3, 0.4]),
        results=[True, False, True, True]
    )

    print(model.predict(user, [fact, fact, fact]))
    print(model.predict_one(user, fact))


if __name__ == "__main__":
    train()
