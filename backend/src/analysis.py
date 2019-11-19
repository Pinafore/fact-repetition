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



def main():
    with open('data/withanswer.question.json') as f:
        question_data = json.load(f)
    with open('record.json') as f:
        record_data = json.load(f)
    with open('preds_2users.json') as f:
        preds = json.load(f)
    question_df = pd.DataFrame(question_data)
    record_df = pd.DataFrame(record_data)
    uid_1_science = 'fb4f77ccf021b67a5a024e1e511678a90adcb7e2'
    uid_7 = 'adfb33027392ad8065317b827a835bf24d0f2dc2'
    uids = [uid_1_science, uid_7]
    outputs = ""
    outputs += "========RECOR========D\n"
    for uid in uids:
        output += uid + '\n'
        qid_array = record_df.loc[record_df['uid'] == uid]['qid'].values # get the history of the user
        for qid in qid_array:
            category = question_df.loc[question_df['qid'] == qid]['category'].values[0]
            answer = question_df.loc[question_df['qid'] == qid]['answer'].values[0]
            outputs += category + '\t\t' + answer + '\n'
    outputs += "========PREDICTIONS========="
    for uid in uids:
        output += uid + '\n\n\n'
        for pred in preds['predictions']:
            if pred['uid'] == uid:
                qid = pred['qid']
                if pred['probs'][0] <= pred['probs'][1]:
                    guess = False
                else:
                    guess = True
                output = 'qid' + qid + '\t' + question_dict[qid]['category'] + '\t' + 'answer: ' + question_dict[qid]['answer'] + '\t' + 'guess: ' + str(guess) + '\n'
                # output += question_dict[qid]['text']
                outputs += output
    
    with open('user_analysis.txt', 'w+') as f:
        f.write(outputs)

if __name__ == '__main__':
    main()
    