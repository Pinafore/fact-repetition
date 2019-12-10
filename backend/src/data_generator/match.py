import json
import pandas as pd 
from tqdm import tqdm
import math

with open('qanta.question.json') as f:
    qanta_question = json.load(f)

qanta_df = pd.DataFrame(qanta_question)
qid_list = list(qanta_df['qid'].unique())

with open('record.deduplicated.json') as f:
    record_data = json.load(f)

matched_record = []
for record in tqdm(record_data):
    if record['qid'] in qid_list:
        matched_record.append(record)

print(len(record_data))
print(len(matched_record))
with open('record.matched.deduplicated.json', 'w+') as f:
    json.dump(matched_record, f)

with open('record.matched.deduplicated.json') as f:
    matched_record = json.load(f)

train = []
dev = []
test = []

num = len(matched_record)
for i in range(num):
    if i <= math.floor(0.7 * num):
        train.append(matched_record[i])
    elif i <= math.floor(0.8 * num):
        dev.append(matched_record[i])
    else:
        test.append(matched_record[i])

with open('train.record.json', 'w+') as f:
    json.dump(train, f)
with open('dev.record.json', 'w+') as f:
    json.dump(dev, f)
with open('test.record.json', 'w+') as f:
    json.dump(test, f)