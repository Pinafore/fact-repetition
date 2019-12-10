import json
import time
from tqdm import tqdm

TRAIN_QUESTION_FILE = './qanta.train.2018.04.18.json'
DEV_QUESTION_FILE = './qanta.dev.2018.04.18.son'
TEST_QUESTION_FILE = './qanta.train.2018.04.18'
OUT_PUT_FILE = './qanta.question.json'

FILE_NAME = [TRAIN_QUESTION_FILE, DEV_QUESTION_FILE, TEST_QUESTION_FILE]

start_time = time.time()
question_data_list = []

for i in range(len(FILE_NAME)):
    print("Start processing: ", QUESTION_DATA_FILE[i])
    with open(FILE_NAME[i]) as f:
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
        del q['tournament']
        del q['year']
        del q['qdb_id']         #all null
        del q['dataset']        #all protobowl
        del q['qanta_id']       #0-n useless compared with proto_id
        del q['gameplay']       #all true
        del q['fold']

    question_data_list.append(question_data)

with open(OUT_PUT_FILE, 'w+') as f:
    json.dump(question_data_list, f)
print("Finish storing question dataset.")

print("Total time: --- %s seconds ---" % (time.time() - start_time))