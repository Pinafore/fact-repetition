import json
import pandas as pd
from tqdm import tqdm

# TRAIN_DATA_FILE = 'qanta.train.json'
TRAIN_DATA_FILE = 'qanta.train.2018.04.18.json'
TRAIN_DATA_OUTPUT = '../cleaned_datasets/question.train.json'
# DEV_DATA_FILE = 'qanta.dev.json'
DEV_DATA_FILE = 'qanta.dev.2018.04.18.json'
DEV_DATA_OUTPUT = '../cleaned_datasets/question.dev.json'
# TEST_DATA_FILE = 'qanta.test.json'
TEST_DATA_FILE = 'qanta.test.2018.04.18.json'
TEST_DATA_OUTPUT = '../cleaned_datasets/question.test.json'

QUESTION_DATA_FILE = [TRAIN_DATA_FILE, DEV_DATA_FILE, TEST_DATA_FILE]
QUESTION_DATA_OUTPUT = [TRAIN_DATA_OUTPUT, DEV_DATA_OUTPUT, TEST_DATA_OUTPUT]


PROTOBOWL_FILE = 'protobowl-042818.log'
RECORD_OUTPUT = '../cleaned_datasets/record.json'


# ============QUESTION DATASET=============
for i in range(len(QUESTION_DATA_FILE)):
    print("Start processing: ", QUESTION_DATA_FILE[i])
    with open(QUESTION_DATA_FILE[i]) as f:
        question_data = json.load(f)

    #keep only question info
    question_data = question_data['questions']    

    for q in tqdm(question_data):

        #rename proto_id to qid
        q['qid'] = q['proto_id']
        del q['proto_id']

        #delete unnecessary info 
        del q['answer']
        del q['first_sentence']
        del q['tokenizations']
        del q['tournament']
        del q['year']
        del q['qdb_id']         #all null
        del q['dataset']        #all protobowl
        del q['qanta_id']       #0-n useless compared with proto_id
        del q['gameplay']       #all true
        del q['fold']

    #store to an external file
    with open(QUESTION_DATA_OUTPUT[i], 'w+') as f:
        json.dump(question_data, f)
    print("Finish processing: ", QUESTION_DATA_FILE[i])


# ============ANSWERING RECORD=============
print("Start processing: ", PROTOBOWL_FILE)
record_data = []
with open(PROTOBOWL_FILE, encoding="utf8") as f:
    for record in tqdm(f):
        record_data.append(json.loads(record))

print("Number of records before processing: ", len(record_data))

# check the counts of some fields
# pb_df = pd.DataFrame(pb_data)
# print(pb_df['duration'].value_counts())

#delete unnecessary info
for record in tqdm(record_data):
    del record['action']
    record['uid'] = record['object']['user']['id']
    record['qid'] = record['object']['qid']
    record['ruling'] = record['object']['ruling']
    record['buzz_ratio'] = record['object']['time_elapsed'] / record['object']['time_remaining']
    del record['object']

# delete ruling:promt and uid or qid: offline
record_data_cleaned = [record for record in record_data if record['ruling'] != 'prompt' \
                         and record['uid'] != 'offline' and record['qid'] != 'offline']

print("Number of records after processing: ", len(record_data_cleaned))

#store to an external file
with open(RECORD_OUTPUT, 'w+') as f:
    json.dump(record_data_cleaned, f)
print("Finish processing: ", PROTOBOWL_FILE)