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
    preds, questions = user_predictions(predictor)
    with open(out_path, 'w+') as f:
        json.dump({'predictions': preds, 'questions': questions}, f)

def user_predictions(predictor):
    predictor._model.output_embedding = False
    # two user answer 10 questions, uid_1_science: answer only science questions; uid_7: answer questions from 7 categories
    uid_1_science = 'fb4f77ccf021b67a5a024e1e511678a90adcb7e2'
    uid_7 = 'adfb33027392ad8065317b827a835bf24d0f2dc2'
    uids = [uid_1_science, uid_7]
    with open('data/withanswer.question.json') as f:
        question_data = json.load(f)
    question_dict = {q['qid']: q for q in question_data}
    # get random 20 qids
    random_qids = random.sample(list(question_df['qid'].values), 20)
    questions = []
    for uid in uids:
        # qid_list = list(record_df.loc[record_df['uid'] == uid]['qid'].values) # get the history of the user
        for qid in random_qids:
            text = question_dict[qid]['text']
            questions.append({'text': question_dict[qid]['text'], 'uid': uid, 'qid': qid, 'answer': question_dict[qid]['answer'], 'category': question_dict[qid]['category']})
    preds = []
    for q in questions:
        pred = predictor.predict_json({'text': q['text'], 'user_id': q['uid'], 'question_id': q['qid']})
        pred['uid'] = q['uid']
        pred['qid'] = q['qid']
        preds.append(pred)
    return preds, questions

def user_predictions(predictor):
    predictor._model.output_embedding = False
    with open('data/withanswer.question.json') as f:
        question_data = json.load(f)
    with open('data/dev.record.json') as f:
        dev_record = json.load(f)
    question_dict = {q['qid']: q for q in question_data}
    dev_df = pd.DataFrame(dev_record)
    uid_df = dev_df.groupby('uid')
    uid_count = uid_df.size().reset_index(name = 'qcount')
    uids = uid_count.sort_values(by=['qcount'], ascending=[False])[0:20]['uid'].values # array
    questions = []
    for uid in uids:
        qid_list = [record['qid'] for record in dev_record if record['uid'] == uid]
        num = 0
        for qid in qid_list:
            num += 1
            text = question_dict[qid]['text']
            questions.append({'text': question_dict[qid]['text'], 'uid': uid, 'qid': qid, 'answer': question_dict[qid]['answer']})
            if num >= 50:
                break
    preds = []
    counter = 0
    for q in tqdm(questions):
        # counter += 1
        # if counter > 5000:
        #     break
        pred = predictor.predict_json({'text': q['text'], 'user_id': q['uid'], 'question_id': q['qid']})
        pred['uid'] = q['uid']
        pred['qid'] = q['qid']
        pred['answer'] = q['answer']
        preds.append(pred)
    return preds, questions
    
    

def extract_embeddings(predictor):
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
    return preds, questions

if __name__ == '__main__':
    main()
    