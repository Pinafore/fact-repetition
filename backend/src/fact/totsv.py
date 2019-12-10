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

def main():
    QUESTION_FILE = 'preds_dev_embeddings.json'

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
        out_file.write('answer\tquestion_accuracy\n')
        for i in tqdm(range(len(question_data['predictions']))):
        # tsv_writer = csv.writer(out_file, delimiter='\n')
        # tsv_writer.writerow(answers)
            if '\n' in question_data['predictions'][i]['answer']:
                out_file.write(question_data['predictions'][i]['answer'].split()[0] + '\t')
            else:
                out_file.write(question_data['predictions'][i]['answer'] + '\t')
            out_file.write(str(question_data['predictions'][i]['accuracy']) + '\n')

    print(i)


if __name__ == '__main__':
    main()