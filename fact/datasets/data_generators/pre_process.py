import json
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def uid_encoding(df):
    uid_list = list(df.groupby('uid').groups.keys())
    uid_list = [[uid] for uid in uid_list]
    uid_enc = OneHotEncoder(handle_unknown='ignore')
    uid_enc.fit(uid_list)
    return uid_enc

def qid_encoding(df):
    qid_list = list(df.groupby('qid').groups.keys())
    qid_list = [[qid] for qid in qid_list]
    qid_enc = OneHotEncoder(handle_unknown='ignore')
    qid_enc.fit(qid_list)
    return qid_enc

start_time = time.time()
TRAIN_RECORD = 'train.record.json'
QUESTION_FILE = 'question.json'
QUESTION_UID = 'question.uid.json'
with open(TRAIN_RECORD) as f:
    train_record = json.load(f)
with open(QUESTION_FILE) as f:
    question_data = json.load(f)

train_df = pd.DataFrame(train_record)
question_df = pd.DataFrame(question_data)


qid_array = train_df.sort_values(by ='qid')['qid'].unique()
uid_array = train_df.sort_values(by ='uid')['uid'].unique()


np.savetxt('qid_array.txt', qid_array, delimiter=',', fmt = '%s')   
np.savetxt('uid_array.txt', uid_array, delimiter=',', fmt = '%s')  