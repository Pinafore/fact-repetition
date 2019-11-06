import torch
import json
import time
from pytorch_pretrained_bert import BertTokenizer, BertModel
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from tqdm import tqdm
# import seaborn as sns

QUESTION_FILE = 'preds.json'

with open(QUESTION_FILE) as f:
    question_data = json.load(f)

import csv

with open('./output.tsv', 'w+') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in tqdm(range(len(question_data['predictions']))):
        tsv_writer.writerow(question_data['predictions'][i]['q_rep'])

print(i)
# answers = []
# print(answers)

with open('./output_label.tsv', 'w+') as out_file:
    for i in tqdm(range(len(question_data['predictions']))):
    # tsv_writer = csv.writer(out_file, delimiter='\n')
    # tsv_writer.writerow(answers)
        if '\n' in question_data['predictions'][i]['answer']:
            out_file.write(question_data['predictions'][i]['answer'].split()[0] + '\n')
        else:
            out_file.write(question_data['predictions'][i]['answer'] + '\n')

print(i)