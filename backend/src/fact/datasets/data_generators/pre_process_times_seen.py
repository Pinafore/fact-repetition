import json
import time
import pandas as pd
import numpy as np

start_time = time.time()
TRAIN_RECORD = 'train.record.json'
TIMES_SEEN = 'times_seen.json'
TIMES_SEEN_CORRECT = 'times_seen_correct.json'
TIMES_SEEN_WRONG = 'times_seen_wrong.json'

with open(TRAIN_RECORD) as f:
    train_record = json.load(f)

df = pd.DataFrame(train_record)

ruling_df = pd.DataFrame({'count' : df.groupby(['uid', 'qid', 'ruling']).size()}).reset_index()
ruling_count_df = pd.pivot_table(ruling_df, values='count', index=['uid', 'qid'], columns=['ruling'], aggfunc=np.sum).fillna(0)
feature = {}
for i in range(len(ruling_count_df.index)):
    feature[str(ruling_count_df.index[i])] = ruling_count_df[0][i] + ruling_count_df[1][i]
with open(TIMES_SEEN, 'w+') as f:
    json.dump(feature, f)

print("1")

ruling_df = pd.DataFrame({'count' : df.groupby(['uid', 'qid', 'ruling']).size()}).reset_index()
ruling_count_df = pd.pivot_table(ruling_df, values='count', index=['uid', 'qid'], columns=['ruling'], aggfunc=np.sum).fillna(0)
feature = {}
for i in range(len(ruling_count_df.index)):
    feature[str(ruling_count_df.index[i])] = ruling_count_df[1][i] # "False  True"
with open(TIMES_SEEN_CORRECT, 'w+') as f:
    json.dump(feature, f)

print("1")

ruling_df = pd.DataFrame({'count' : df.groupby(['uid', 'qid', 'ruling']).size()}).reset_index()
ruling_count_df = pd.pivot_table(ruling_df, values='count', index=['uid', 'qid'], columns=['ruling'], aggfunc=np.sum).fillna(0)
feature = {}
for i in range(len(ruling_count_df.index)):
    feature[str(ruling_count_df.index[i])] = ruling_count_df[0][i]
with open(TIMES_SEEN_WRONG, 'w+') as f:
    json.dump(feature, f)

print("1")