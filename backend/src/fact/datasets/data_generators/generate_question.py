import json
import pandas as pd
from tqdm import tqdm

RECORD_FILE = './protobowl-042818.log'
NEW_QUESTION_FILE = './question.json'

record_data = []
with open(RECORD_FILE, encoding="utf8") as f:
    for record in tqdm(f):
        record_data.append(json.loads(record))

question_data = []
for record in tqdm(record_data):
    question_data.append({'qid': record['object']['qid'], 'text': record['object']['question_text'], 'category': record['object']['question_info']['category'], 'difficulty': record['object']['question_info']['difficulty']})

df = pd.DataFrame(question_data)
unique_df = df.drop_duplicates()

# print("len of question.", len(new_df))
# print("len of qid", len(df['qid'].unique()))


with open(NEW_QUESTION_FILE, 'w+') as f:
    json.dump(unique_df.to_dict('records'), f)
