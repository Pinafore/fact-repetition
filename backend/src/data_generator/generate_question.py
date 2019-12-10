import json
import pandas as pd 
from tqdm import tqdm
import math

with open('qanta.question.json') as f:
    qanta_question = json.load(f)

qanta_df = pd.DataFrame(qanta_question)
with open('record.matched.deduplicated.json') as f:
    record_data = json.load(f)

record_df = pd.DataFrame(record_data)
qid_list = list(record_df['qid'].unique()) 
with open('train.record.json') as f:
    train_record = json.load(f)

train_df = pd.DataFrame(train_record)

avg = train_df['ruling'].mean() # avg accuracy
accuracy_series = train_df.groupby(['qid']).mean()['ruling']
value = []
for i in range(len(accuracy_series)):
    value.append(accuracy_series[i])
qid =  list(accuracy_series.keys())
question_accuracy = dict(zip(qid, value))

matched_question = []
for question in tqdm(qanta_question):
    question['answer'] = question['page']
    del question['page']
    qid = question['qid']
    if qid in qid_list:
        if qid in question_accuracy:
            question['accuracy'] = question_accuracy[qid]
        else:
            question['accuracy'] = avg
        matched_question.append(question)

print(len(qanta_question))
print(len(matched_question))
with open('matched.question.json', 'w+') as f:
    json.dump(matched_question, f)

