import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

def main():
    QUESTION_FILE = 'preds_dev_embeddings1.json'
    with open(QUESTION_FILE) as f:
        question_data = json.load(f)

    e_df = pd.DataFrame(question_data['predictions'])
    q_df = pd.DataFrame(question_data['questions'])
    df = pd.concat([e_df['q_rep'], q_df['answer']], axis=1)
    g = df.groupby(['answer'])
    answers = list(dict(list(g)))
    q_reps = []
    answer_emb = []
    for answer in tqdm(answers):
        gp = g.get_group(answer)
        q_rep_list = []
        for i in range(len(gp)):
            q_rep_list.append(gp.iloc[[i]]['q_rep'].values[0])
        q_rep = np.mean(np.array(q_rep_list), axis = 0)
        q_reps.append(q_rep)
        answer_emb.append({'answer': answer, 'q_rep': q_rep})
    kmeans = KMeans(n_clusters=5, random_state = 0).fit(q_reps)
    labels = kmeans.labels_

    print('Finish kmeans.')

    import csv

    with open('./output_plot1.tsv', 'w+') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in tqdm(range(len(q_reps))):
            tsv_writer.writerow(q_reps[i])

    with open('./output_label_plot1.tsv', 'w+') as out_file:
        out_file.write('answer\tkmeans_label\n')
        for i in tqdm(range(len(answers))):
        # tsv_writer = csv.writer(out_file, delimiter='\n')
        # tsv_writer.writerow(answers)
            out_file.write(answers[i] + '\t' + str(labels[i]) + '\n')    

if __name__ == '__main__':
    main()