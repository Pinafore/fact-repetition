import json
import pandas as pd
import numpy as np
from tqdm import tqdm

def main():
    QUESTION_FILE = 'preds_dev_embeddings.json'
    with open(QUESTION_FILE) as f:
        question_data = json.load(f)

    import csv

    with open('./output_plot2.tsv', 'w+') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in tqdm(range(len(question_data['predictions']))):
            tsv_writer.writerow(question_data['predictions'][i]['q_rep'])

    with open('./output_label_plot2.tsv', 'w+') as out_file:
        out_file.write('split\n')
        for i in tqdm(range(len(question_data['predictions']))):
        # tsv_writer = csv.writer(out_file, delimiter='\n')
        # tsv_writer.writerow(answers)
            out_file.write(str(question_data['questions'][i]['split']) + '\n')    

if __name__ == '__main__':
    main()