import json
import click
import os
import pandas as pd
from allennlp.predictors.predictor import Predictor
from allennlp.data import DatasetReader, Instance
from allennlp.common.util import JsonDict
from fact.models.bert_baseline import KarlModel
from fact.datasets.qanta import QantaReader
from tqdm import tqdm


KARL_PREDICTOR = 'karl_predictor'


@Predictor.register(KARL_PREDICTOR)
class KarlPredictor(Predictor):
    def __init__(self, model: KarlModel, dataset_reader: QantaReader):
        super().__init__(model, dataset_reader)
        self._model.output_embedding = True
    
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(
            json_dict['text'],
            json_dict['user_id'],
            json_dict['question_id'],
            user_accuracy=json_dict.get('user_accuracy'),
            user_buzzratio=json_dict.get('user_buzzratio'),
            user_count=json_dict.get('user_count'),
            question_accuracy=json_dict.get('question_accuracy'),
            question_buzzratio=json_dict.get('question_buzzratio'),
            question_count=json_dict.get('question_count'),
            times_seen=json_dict.get('times_seen'),
            times_seen_correct=json_dict.get('times_seen_correct'),
            times_seen_wrong=json_dict.get('times_seen_wrong'),
            label=json_dict.get('label')
        )

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        pred = self.predict_instance(instance)
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        pred['all_labels'] = all_labels
        return pred


@click.group()
def main():
    pass


@main.command()
@click.argument('archive_path')
@click.argument('out_path')

def main(archive_path, out_path):
    predictor = KarlPredictor.from_path(archive_path, predictor_name=KARL_PREDICTOR)
    pred = user_predictions()
    with open(out_path, 'w+') as f:
        json.dump({'predictions': preds, 'questions': questions}, f)

def user_predictions():
    with open('data/withanswer.question.json') as f:
        question_data = json.load(f)
    with open('data/dev.record.json') as f:
        dev_record = json.load(f)
    question_dict = {q['qid']: q for q in question_data}
    uid_count = uid.size().reset_index(name = 'qcount')
    uids = uid_count.sort_values(by=['qcount'], ascending=[False])[0:20][uid].valus # array
    questions = []
    for uid in uids:
        question = dev_df.loc[dev_df['uid'] == uid]['qid'].values
        for qid in question:
            text = question_dict[qid]['text']
            questions.append({'text': question_dict[qid]['text'], 'uid': uid, 'qid': qid})
    preds = []
    counter = 0
    for q in tqdm(questions):
        counter += 1
        if counter > 5000:
            break
        pred = predictor.predict_json({'text': q['text'], 'user_id': '', 'question_id': ''})
        pred['answer'] = q['answer']
        preds.append(pred)
    return preds
    
    

def extract_embeddings():
    with open('data/withanswer.question.json') as f:
        question_data = json.load(f)
    with open('data/train.record.json') as f:
        dev_record = json.load(f)
    question_dict = {q['qid']: q for q in question_data}
    dev_df = pd.DataFrame(dev_record)
    unique_qid = dev_df['qid'].unique()
    # questions = [
    #     'Name this capital of idaho',
    #     'This physicist was known for being a talented violinist',
    # ]
    # for extract embeddings only
    questions = []
    answers = []
    for qid in tqdm(unique_qid):
        questions.append({'text': question_dict[qid]['text'], 'answer': question_dict[qid]['answer']})

    preds = []
    counter = 0
    for q in tqdm(questions):
        counter += 1
        # if counter > 5000:
        #     break
        # Since user_id/question_id don't exist, these are set to null values
        # and counts are set to zero as well
        # We should probably change the api a little to be able to manually pass counts
        # Output has keys: q_rep, logits, probs
        pred = predictor.predict_json({'text': q['text'], 'user_id': '', 'question_id': ''})
        pred['answer'] = q['answer']
        preds.append(pred)
    return preds

if __name__ == '__main__':
    main()
    