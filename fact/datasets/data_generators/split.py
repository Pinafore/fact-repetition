import json
import time
from tqdm import tqdm

REPEATED_PATH = './repeated_records/'
NONREPEATED_PATH = './non_repeated_records/'
REPEATED_RECORD_WITH_UID_QID_COUNT_FILE = 'uid_qid_count.repeated.record.json'
NON_LESS_FREQUENT_USER_REPEATED_RECORD_FILE = 'non_less_frequent_user.repeated.record.json'
NON_LESS_FREQUENT_QUESTION_REPEATED_RECORD_FILE = 'non_less_frequent_question.repeated.record.json'
NON_LESS_FREQUENT_USER_AND_QUESTION_REPEATED_RECORD_FILE = 'non_less_frequent_user_and_question.repeated.record.json'
NONREPEATED_RECORD_WITH_UID_QID_COUNT_FILE = 'uid_qid_count.nonrepeated.record.json'
NON_LESS_FREQUENT_USER_NONREPEATED_RECORD_FILE = 'non_less_frequent_user.nonrepeated.record.json'
NON_LESS_FREQUENT_QUESTION_NONREPEATED_RECORD_FILE = 'non_less_frequent_question.nonrepeated.record.json'
NON_LESS_FREQUENT_USER_AND_QUESTION_NONREPEATED_RECORD_FILE = 'non_less_frequent_user_and_question.nonrepeated.record.json'


FILE_PATH = [REPEATED_PATH, REPEATED_PATH, REPEATED_PATH, REPEATED_PATH, \
            NONREPEATED_PATH, NONREPEATED_PATH, NONREPEATED_PATH, NONREPEATED_PATH]
FILE_NAME = [REPEATED_RECORD_WITH_UID_QID_COUNT_FILE, NON_LESS_FREQUENT_USER_REPEATED_RECORD_FILE, \
            NON_LESS_FREQUENT_QUESTION_REPEATED_RECORD_FILE, NON_LESS_FREQUENT_USER_AND_QUESTION_REPEATED_RECORD_FILE, \
            NONREPEATED_RECORD_WITH_UID_QID_COUNT_FILE, NON_LESS_FREQUENT_USER_NONREPEATED_RECORD_FILE, \
            NON_LESS_FREQUENT_QUESTION_NONREPEATED_RECORD_FILE, NON_LESS_FREQUENT_USER_AND_QUESTION_NONREPEATED_RECORD_FILE]

start_time = time.time()

for i in range(len(FILE_NAME)):
    # load the dataset
    with open(FILE_PATH[i] + FILE_NAME[i]) as f:
        record_data = json.load(f)

    # calculate the number for train/dev/test
    N = len(record_data)
    train_num = round(N * 0.7)
    dev_num = round(N * 0.2)
    test_num = N - train_num - dev_num
    print("Total number of record: %d. Train: %d, Dev: %d, Test: %d" % (N, train_num, dev_num, test_num))

    # split to train/de/test
    train_data = []
    dev_data = []
    test_data = []

    for j in tqdm(range(N)):
        if j < train_num:
            train_data.append(record_data[j])
        elif j < train_num + dev_num:
            dev_data.append(record_data[j])
        else:
            test_data.append(record_data[j])

    print("Successfually Split the %s into train: %d, dev: %d, test:%d. " % (
        FILE_PATH[i] + FILE_NAME[i], len(train_data), len(dev_data), len(test_data)))

    # store train/dev/test set to file
    with open(FILE_PATH[i] + 'train.' + FILE_NAME[i], 'w+') as f:
        json.dump(train_data, f)
    with open(FILE_PATH[i] + 'dev.' + FILE_NAME[i], 'w+') as f:
        json.dump(dev_data, f)
    with open(FILE_PATH[i] + 'test.' + FILE_NAME[i], 'w+') as f:
        json.dump(test_data, f)
    print("Finish storing splited dataset for %s.", FILE_PATH[i] + FILE_NAME[i])
    print("After storing : --- %s seconds ---" % (time.time() - start_time))

print("Total time: --- %s seconds ---" % (time.time() - start_time))