import json
import time
from tqdm import tqdm
import pandas as pd
import math
import datetime

# MONTH = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', \
#      'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', \
#          'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
# def convert_date(row):
#     date = row['date']
#     month = date[4:7]
#     day = date[8:10]
#     year = date[11:15]
#     month = MONTH[month]
#     return month + '/' + day + '/' + year

MONTH = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, \
     'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, \
         'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def less_than_1hour(last_date, cur_date):
    last_month = MONTH[last_date[4:7]]
    cur_month = MONTH[cur_date[4:7]]
    last_day = int(last_date[8:10])
    cur_day = int(cur_date[8:10])
    last_year = int(last_date[11:15])
    cur_year = int(cur_date[11:15])
    last_hour = int(last_date[16:18])
    cur_hour = int(cur_date[16:18])
    last_min = int(last_date[19:21])
    cur_min = int(cur_date[19:21])
    last_sec = int(last_date[22:24])
    cur_sec = int(cur_date[22:24])
    last_date = datetime.datetime(last_year, last_month, last_day, last_hour, last_min, last_sec)
    cur_date = datetime.datetime(cur_year, cur_month, cur_day, cur_hour, cur_min, cur_sec)
    # print(last_date, cur_date, (cur_date - last_date).seconds, (cur_date - last_date).seconds <= 3600)
    return (cur_date - last_date).seconds <= 3600


with open('record.json') as f:
    record_data = json.load(f)

print("Number of record: ", len(record_data))
record_data = [record for record in record_data if record['qid'] != 'question_id' and record['uid'] != 'offline' and record['ruling'] != 'prompt'] # remove noise
record_df = pd.DataFrame(record_data) # convert to df
uqid_count_df = record_df.groupby(['uid', 'qid']).size().reset_index(name = 'uqcount').sort_values(by=['uqcount'], ascending=[False]) # groupby uid, qid, sort by count
uqid_df = record_df
# uqid_df['date_only'] = uqid_df.apply(lambda row: convert_date(row), axis=1) # replace date format
uqid_df.sort_values(by=['uid', 'qid'], inplace=True)
df_list = []
df_list.append(uqid_df.iloc[[0]])
#math.floor(len(uqid_df)/4)
delete = 0
last = 0
for i in tqdm(range(1, math.floor(len(uqid_df)/4))):
    # print(i, ': ', end = '\t')
    last_uid = uqid_df.iloc[[i-1]]['uid'].values[0]
    last_qid = uqid_df.iloc[[i-1]]['qid'].values[0]
    uid = uqid_df.iloc[[i]]['uid'].values[0]
    qid = uqid_df.iloc[[i]]['qid'].values[0]
    # print(uid, ' ', last_uid, ' ', qid, ' ', last_qid, ' ', qid == last_qid and uid == last_uid)
    if qid == last_qid and uid == last_uid:
        last_date = uqid_df.iloc[[last]]['date'].values[0]
        cur_date = uqid_df.iloc[[i]]['date'].values[0]
        # print(last_date, cur_date, less_than_2hours(last_date, cur_date))
        if less_than_1hour(last_date, cur_date):
            # print('=====delete=============')
            delete += 1
            continue
    df_list.append(uqid_df.iloc[[i]])
    last = i

print(math.floor(len(uqid_df)*3/4) - math.floor(len(uqid_df)/2))
print(delete)
new_df = pd.concat(df_list)
print(len(new_df))
df_json = new_df.to_dict(orient='records')

with open('record.deduplicated1.json', 'w+') as f:
    json.dump(df_json, f)



