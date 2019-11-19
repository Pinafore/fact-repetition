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

uids = ['8b7a4410f6290a0c9a50ef3bb1fc70769938db33',
 '04e9b820b50006a7573c29410911deb3db661565',
 '00f3cd9e2c518d24d3e71852e80f54045c0ed66a',
 '64db935668c3f0656f22b6e16c58803b34f507d7',
 '8281221a4063d4f463625f4492b0b7bfa6446655',
 '573778da91441db3d0af7cf311adee07a55ff99a',
 '1b962b4b8b9a1c9d3e79090ecec070c76c1839e5',
 'df4e7058889867ce37e0791141e9748aaca53817',
 'ecd0a1786f53771045841281d8c19af2c5622523',
 '491cadb3defe26a64c362226b0f659b6211ff9b0',
 'f71c7255f375b061a125cb923ea70a1906e30828',
 'b69afcbad4a03ae5d5d856922fc0a357594d254e',
 '69ca9eccf15947778e56cee0e1ec0a9a599c06eb',
 'aa2fcfe16944c37c179d98e73972c887f558b580',
 '8a5f30d4645c150f3748d67408dd8e47641fd741',
 '9556e21b4516b214e8bbf7701f84691430c39743',
 'b406b714399febe033ed83cc4aaebed35e0f39eb',
 'd38ff4112165b1d36e83f70cae0a7da9704d0b65',
 '2de57808a73a55e20ea469db3876234526a1c6b5',
 '074b4b8f7cf1a99d13bcc1e6c5beac34fd37716c']

def main():
    with open('data/withanswer.question.json') as f:
        question_data = json.load(f)
    with open('data/train.record.json') as f:
        train_record = json.load(f)
    with open('data/dev.record.json') as f:
        dev_record = json.load(f)
    with open('preds_50.json') as f:
        preds = json.load(f)
    question_dict = {q['qid']: q for q in question_data}
    train_df = pd.DataFrame(train_record)
    dev_df = pd.DataFrame(dev_record)

    outputs = ""
    for uid in uids:
        outputs += "TRAIN RECORD\n"
        for record in train_record:
            if record['uid'] == uid:
                qid = record['qid']
                output = 'uid: ' +  uid + '\t' + 'qid: ' + qid + '\t' + 'ruling: ' + str(record['ruling']) + '\t\t\t\t' + 'answer: ' + question_dict[qid]['answer'] + '\n'
                outputs += output
        outputs += "DEV RECORD\n"
        for pred in preds['predictions']:
            if pred['uid'] == uid:
                qid = pred['qid']
                if pred['probs'][0] <= pred['probs'][1]:
                    guess = False
                else:
                    guess = True
                for record in dev_record:
                    if record['uid'] == uid and record['qid'] == qid:
                        ruling = record['ruling']
                        break
                output = 'uid: ' + uid + '\t' + 'qid: ' + qid + '\t' + 'ruling: ' + str(ruling) + '\t' + 'guess: ' + str(guess) + '\t' + 'answer: ' + question_dict[qid]['answer'] + '\n'
                outputs += output
    
    with open('user_analysis.txt', 'w+') as f:
        f.write(outputs)

if __name__ == '__main__':
    main()
    