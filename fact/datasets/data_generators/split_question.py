import json
import time
from tqdm import tqdm

QUESTION_FILE = './question.json'
TRAIN_QUESTION = './train.question.json'
DEV_QUESTION = './dev.question.json'
TEST_QUESTION = './test.question.json'

start_time = time.time()


with open(QUESTION_FILE) as f:
    question_data = json.load(f)

# calculate the number for train/dev/test
N = len(question_data)
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
        train_data.append(question_data[j])
    elif j < train_num + dev_num:
        dev_data.append(question_data[j])
    else:
        test_data.append(question_data[j])

print("Successfually Split the %s into train: %d, dev: %d, test:%d. " % (
    QUESTION_FILE, len(train_data), len(dev_data), len(test_data)))

# store train/dev/test set to file
with open(TRAIN_QUESTION, 'w+') as f:
    json.dump(train_data, f)
with open(DEV_QUESTION, 'w+') as f:
    json.dump(dev_data, f)
with open(TEST_QUESTION, 'w+') as f:
    json.dump(test_data, f)
print("Finish storing splited dataset for %s.", QUESTION_FILE)
