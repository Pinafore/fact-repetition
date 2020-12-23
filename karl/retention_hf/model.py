import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from transformers import (
    DistilBertConfig,
    DistilBertTokenizerFast,
    DistilBertModel,
    DistilBertPreTrainedModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from transformers.data.metrics import simple_accuracy

from karl.config import settings
from karl.retention_hf.data import RetentionInput, RetentionFeaturesSchema
from karl.retention_hf.data import RetentionDataset, retention_data_collator


class DistilBertRetentionModel(DistilBertPreTrainedModel):

    def __init__(
        self,
        config,
        retention_feature_size: int,
        num_labels: int,
        **kwargs,
    ):
        super().__init__(config)
        self.num_labels = num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.pre_classifier = nn.Linear(config.dim + retention_feature_size, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        retention_features=None,
        output_attentions=None,
        labels=None,
    ):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        if retention_features is not None:
            pooled_output = torch.cat((pooled_output, retention_features), axis=1)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}


if __name__ == '__main__':
    feature_keys = [
        key for key in RetentionFeaturesSchema.__fields__.keys()
        if key not in ['user_id', 'card_id', 'label']
    ]
    model = DistilBertRetentionModel(
        config=DistilBertConfig(),
        retention_feature_size=len(feature_keys),
        num_labels=2,
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = RetentionDataset(settings.DATA_DIR, 'train', tokenizer)
    test_dataset = RetentionDataset(settings.DATA_DIR, 'test', tokenizer)
    training_args = TrainingArguments(
        output_dir=f'{settings.CODE_DIR}/output/retention_hf',
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-05,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=retention_data_collator,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)

    print("*** Evaluate ***")

    result = trainer.evaluate(eval_dataset=test_dataset)

    output_eval_file = os.path.join(
        training_args.output_dir, 'eval_results_retention.txt'
    )
    with open(output_eval_file, "w") as writer:
        print("***** Eval results retention *****")
        for key, value in result.items():
            print("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))
