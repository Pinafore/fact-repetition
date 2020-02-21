import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

TRAIN_RECORD = 'train.record.json'
DEV_RECORD = 'dev.record.json'
# TEST_RECORD = 'test.record.json'

start_time = time.time()

with open(TRAIN_RECORD) as f:
    train_record = json.load(f)
with open(DEV_RECORD) as f:
    dev_record = json.load(f)
# with open(TEST_RECORD) as f:
#     test_record = json.load(f)

train_df = pd.DataFrame(train_record)
dev_df = pd.DataFrame(dev_record)
# test_df = pd.DataFrame(test_record)

# =========== most frequent baseline ====================
train_ruling_df = pd.DataFrame({'count' : train_df.groupby(['ruling']).size()}).reset_index()
# print("Train most frequent baseline accuracy: ", train_ruling_df['count'][1] / (train_ruling_df['count'][0] + train_ruling_df['count'][1]))
dev_ruling_df = pd.DataFrame({'count' : dev_df.groupby(['ruling']).size()}).reset_index()
# test_ruling_df = pd.DataFrame({'count' : test_df.groupby(['ruling']).size()}).reset_index()
print("Dev most frequent baseline accuracy: ", dev_ruling_df['count'][1] / (dev_ruling_df['count'][0] + dev_ruling_df['count'][1]))


# =========== most frequent user baseline ====================
temp = pd.DataFrame({'count' : train_df.groupby(['uid', 'ruling'] ).size()}).reset_index()
train_uid_ruling_df = pd.pivot_table(temp, values='count', index=['uid'], columns=['ruling'], aggfunc=np.sum).fillna(0)
temp = pd.DataFrame({'count' : dev_df.groupby(['uid', 'ruling'] ).size()}).reset_index()
dev_uid_ruling_df = pd.pivot_table(temp, values='count', index=['uid'], columns=['ruling'], aggfunc=np.sum).fillna(0)
train_uid_list = train_uid_ruling_df.index.tolist()
dev_uid_list = dev_uid_ruling_df.index.tolist()
# train_uid_ruling_df['guess'] = train_uid_ruling_df[1] >= train_uid_ruling_df[0]   
# temp = train_uid_ruling_df.groupby(['Guess']).size().reset_index(name='count') 
guess_correct = 0
guess_wrong = 0
for i in tqdm(range(len(dev_uid_list))):
    uid = dev_uid_list[i]
    dev_correct = dev_uid_ruling_df[1][i]
    dev_wrong = dev_uid_ruling_df[0][i]
    if uid not in train_uid_list:
        guess_correct += dev_correct
        guess_wrong += dev_wrong
    else:
        index = train_uid_list.index(uid)
        if train_uid_ruling_df[1][index] >= train_uid_ruling_df[0][index]:
            guess_correct += dev_correct
            guess_wrong += dev_wrong
        else:
            guess_correct += dev_wrong
            guess_wrong += dev_correct
print("Dev most frequent user baseline accuracy: ", guess_correct / (guess_correct + guess_wrong))



temp = pd.DataFrame({'count' : train_df.groupby(['qid', 'ruling'] ).size()}).reset_index()
train_qid_ruling_df = pd.pivot_table(temp, values='count', index=['qid'], columns=['ruling'], aggfunc=np.sum).fillna(0)
temp = pd.DataFrame({'count' : dev_df.groupby(['qid', 'ruling'] ).size()}).reset_index()
dev_qid_ruling_df = pd.pivot_table(temp, values='count', index=['qid'], columns=['ruling'], aggfunc=np.sum).fillna(0)
train_qid_list = train_qid_ruling_df.index.tolist()
dev_qid_list = dev_qid_ruling_df.index.tolist()
# train_qid_ruling_df['guess'] = train_qid_ruling_df[1] >= train_qid_ruling_df[0]   
# temp = train_qid_ruling_df.groupby(['Guess']).size().reset_index(name='count') 
guess_correct = 0
guess_wrong = 0
for i in tqdm(range(len(dev_qid_list))):
    qid = dev_qid_list[i]
    dev_correct = dev_qid_ruling_df[1][i]
    dev_wrong = dev_qid_ruling_df[0][i]
    if qid not in train_qid_list:
        guess_correct += dev_correct
        guess_wrong += dev_wrong
    else:
        index = train_qid_list.index(qid)
        if train_qid_ruling_df[1][index] >= train_qid_ruling_df[0][index]:
            guess_correct += dev_correct
            guess_wrong += dev_wrong
        else:
            guess_correct += dev_wrong
            guess_wrong += dev_correct

print("Dev most frequent question baseline accuracy: ", guess_correct / (guess_correct + guess_wrong))