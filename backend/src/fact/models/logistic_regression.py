import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import json
import torch
from tqdm import tqdm
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

TRAIN_RECORD = './train.record.json'
DEV_RECORD = './dev.record.json'
TEST_RECORD = './test.record.json'
TRAIN_QUESTION = './train.question.json'
DEV_QUESTION = './dev.question.json'
TEST_QUESTION = './test.record.json'
QUESTION_FILE = './question.json'
TIMES_SEEN = 'times_seen.json'
TIMES_SEEN_CORRECT = 'times_seen_correct.json'
TIMES_SEEN_WRONG = 'times_seen_wrong.json'


def accuracy_per_user(df):
    avg = df['ruling'].mean()
    accuracy_series = df.groupby(['uid']).mean()['ruling']
    value = []
    for i in range(len(accuracy_series)):
        value.append(accuracy_series[i])
    uid =  list(accuracy_series.keys())
    feature = dict(zip(uid, value))
    feature['<UKN>'] = avg
    return feature
    # numpy.savetxt("./features/accuracy_per_user", feature)

def average_buzz_ratio_per_user(df):
    avg = df['buzz_ratio'].mean()
    buzz_ratio_series = df.groupby(['uid']).mean()['buzz_ratio']
    value = []
    for i in range(len(buzz_ratio_series)):
        value.append(buzz_ratio_series[i])
    uid =  list(buzz_ratio_series.keys())
    feature = dict(zip(uid, value))
    feature['<UKN>'] = avg
    return feature

def accuracy_per_question(df):
    avg = df['ruling'].mean()
    accuracy_series = df.groupby(['qid']).mean()['ruling']
    value = []
    for i in range(len(accuracy_series)):
        value.append(accuracy_series[i])
    qid =  list(accuracy_series.keys())
    feature = dict(zip(qid, value))
    feature['<UKN>'] = avg
    return feature

def average_buzz_ratio_per_question(df):
    avg = df['buzz_ratio'].mean()
    buzz_ratio_series = df.groupby(['qid']).mean()['buzz_ratio']
    value = []
    for i in range(len(buzz_ratio_series)):
        value.append(buzz_ratio_series[i])
    qid =  list(buzz_ratio_series.keys())
    feature = dict(zip(qid, value))
    feature['<UKN>'] = avg
    return feature

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

def uid_count(df):
    uid = list(df.groupby('uid').groups.keys())
    uid_count_series = df.groupby(['uid']).mean()['uid_count']
    value = []
    for i in range(len(uid_count_series)):
        value.append(uid_count_series[i])
    feature = dict(zip(uid, value))
    feature['<UKN>'] = 0
    return feature

def qid_count(df):
    qid = list(df.groupby('qid').groups.keys())
    qid_count_series = df.groupby(['qid']).mean()['qid_count']
    value = []
    for i in range(len(qid_count_series)):
        value.append(qid_count_series[i])
    feature = dict(zip(qid, value))
    feature['<UKN>'] = 0
    return feature


def category_subcategory_difficulty():
    with open(QUESTION_FILE) as f:
        question_data = json.load(f)
    df = pd.DataFrame(question_data)
    print("question len: ", len(df.groupby('qid')))
    enc = OneHotEncoder(handle_unknown='ignore')
    category_list = list(df.groupby('category').groups.keys())
    subcategory_list = list(df.groupby('subcategory').groups.keys())
    difficulty_list = list(df.groupby('difficulty').groups.keys())
    category_subcategory_difficulty_list = []
    sorted_len = [len(category_list), len(subcategory_list), len(difficulty_list)]
    sorted_len.sort()
    for i in range(sorted_len[-1]):
        if i < sorted_len[0]:
            category_subcategory_difficulty_list.append([category_list[i], subcategory_list[i], difficulty_list[i]])
        elif i < sorted_len[1]:
            category_subcategory_difficulty_list.append([category_list[i], subcategory_list[i], difficulty_list[0]])
        else:
            category_subcategory_difficulty_list.append([category_list[0], subcategory_list[i], difficulty_list[0]])
    enc.fit(category_subcategory_difficulty_list)
    return enc

def bag_of_words():
    with open(QUESTION_FILE) as f:
        question_data = json.load(f)
    text_list = [record['text'] for record in question_data]
    enc = CountVectorizer()
    enc.fit(text_list)
    return enc


def bert_pretrained_embeddings(file_name):
    with open(file_name) as f:
        question_data = json.load(f)
        text_list = [record['text'] for record in question_data]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    text_embedding_list = []
    for text in text_list:
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # segments_ids.to(device)
        # tokens_tensor.to(device)
        # segments_tensors.to(device)
        # model.to(device)
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
        token_embeddings = []
        batch_i = 0
        for token_i in range(len(tokenized_text)):
            hidden_layers = []
            for layer_i in range(len(encoded_layers)):
                vec = encoded_layers[layer_i][batch_i][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
        sentence_embedding = torch.mean(encoded_layers[len(encoded_layers) - 1], 1)
        # print("vector: ", sentence_embedding[0].shape[0])
        # print("vector: ", sentence_embedding[0])
        text_embedding_list.append(sentence_embedding[0].tolist())
    return text_embedding_list


start_time = time.time()

# load the dataset
with open(TRAIN_RECORD) as f:
    train_record_data = json.load(f)
with open(DEV_RECORD) as f:
    dev_record_data = json.load(f)
with open(TEST_RECORD) as f:
    test_record_data = json.load(f)
with open(QUESTION_FILE) as f:
    question_data = json.load(f)
with open(TIMES_SEEN) as f:
    times_seen_feature = json.load(f)
with open(TIMES_SEEN_CORRECT) as f:
    times_seen_correct_feature = json.load(f)
with open(TIMES_SEEN_WRONG) as f:
    times_seen_wrong_feature = json.load(f)
    
# qid_array = open(QID_FILE, 'r').read().split('\n')[0:-2] # remove question_id index and empty element
# uid_array = open(UID_FILE, 'r').read().split('\n')[0:-1] # remove empty element
# num_qid = len(qid_array)
# num_uid = len(uid_array)

train_num = len(train_record_data)
dev_num = len(dev_record_data)

df = pd.DataFrame(train_record_data)
accuracy_per_user_feature = accuracy_per_user(df)
accuracy_per_questio_feature = accuracy_per_question(df)
average_buzz_ratio_per_user_feature = average_buzz_ratio_per_user(df)
average_buzz_ratio_per_question_feature = average_buzz_ratio_per_question(df)
qid_enc = qid_encoding(df)
uid_enc = uid_encoding(df)
# print("After encoding: --- %s seconds ---" % (time.time() - start_time))
train_uid_list = [[record['uid']] for record in train_record_data]
train_qid_list = [[record['qid']] for record in train_record_data]
train_uid_feature = uid_enc.transform(train_uid_list)
train_qid_feature = qid_enc.transform(train_qid_list)
uid_count_feature = uid_count(df)
qid_count_feature = qid_count(df)
# category_subcategory_difficulty_enc = category_subcategory_difficulty()
# text_enc = bag_of_words()
# question_dic = {q['qid']: [q['category'], q['subcategory'], q['difficulty']] for q in question_data}
question_dic = {q['qid']: [q['category'], q['difficulty']] for q in question_data}
# text_dic = {q['qid']: [q['text']] for q in question_data}
print("qid length: ", len(list(pd.DataFrame(train_record_data + dev_record_data + test_record_data).groupby('qid').groups.keys())))
# print("After transdorming: --- %s seconds ---" % (time.time() - start_time))
train_x = []
train_y = []
feature_matrix = []
# category_subcategory_difficulty_list = []
text_list = []
uid_onehot_feature = []
qid_onehot_feature = []

for i in tqdm(range(train_num)):
    uid = train_record_data[i]['uid']
    qid = train_record_data[i]['qid']
    feature_vector = []
    if uid in uid_count_feature:
        feature_vector.append(accuracy_per_user_feature[uid])
        feature_vector.append(average_buzz_ratio_per_user_feature[uid])
        feature_vector.append(uid_count_feature[uid])
    else:
        feature_vector.append(accuracy_per_user_feature['<UKN>'])
        feature_vector.append(average_buzz_ratio_per_user_feature['<UKN>'])
        feature_vector.append(uid_count_feature['<UKN>'])

    if qid in qid_count_feature:
        feature_vector.append(accuracy_per_questio_feature[qid])
        feature_vector.append(average_buzz_ratio_per_question_feature[qid])
        feature_vector.append(qid_count_feature[qid])
    else:
        feature_vector.append(accuracy_per_questio_feature['<UKN>'])
        feature_vector.append(average_buzz_ratio_per_question_feature['<UKN>'])
        feature_vector.append(qid_count_feature['<UKN>'])
    feature_matrix.append(feature_vector)
    if str((uid, qid)) in times_seen_feature:
        times_seen = times_seen_feature[str((uid, qid))]
        times_seen_correct = times_seen_correct_feature[str((uid, qid))]
        times_seen_wrong = times_seen_wrong_feature[str((uid, qid))]
    else:
        times_seen = 0
        times_seen_correct = 0
        times_seen_wrong = 0
    feature_vector += [times_seen, times_seen_correct, times_seen_wrong]
    # if qid in question_dic:
    #     category_subcategory_difficulty_list.append([question_dic[qid][0], question_dic[qid][1], question_dic[qid][2]])
    # else:
    #     category_subcategory_difficulty_list.append(['<UKN>', '<UKN>', '<UKN>'])
    # if qid in text_dic:
    #     text_list.append([text_dic[qid]])
    # else:
    #     text_list.append(['<UKN>'])

feature_matrix = sp.csr_matrix(feature_matrix)
# category_subcategory_difficulty_feature = category_subcategory_difficulty_enc.transform(category_subcategory_difficulty_list)
# text_feature = text_enc.transform(text_list)
print("len of train_uid_feature: ", train_uid_feature.shape)
print("len of train_qid_feature: ", train_qid_feature.shape)
print("len of feature_matrix: ", feature_matrix.shape)
# print("len of category_subcategory_difficulty_feature: ", category_subcategory_difficulty_feature.shape)
# train_x = sp.hstack((train_uid_feature, train_qid_feature, feature_matrix, times_seen_feature,  \
#                         category_subcategory_difficulty_feature), format='csr')
train_text_embedding_list = bert_pretrained_embeddings(TRAIN_QUESTION)
train_text_embedding_matrix = sp.csr_matrix(train_text_embedding_list)
train_x = sp.hstack((train_uid_feature, train_qid_feature, feature_matrix, train_text_embedding_matrix), format='csr')
train_y = np.array(train_y)


dev_x = []
dev_y = []
feature_matrix = []
# category_subcategory_difficulty_list = []
text_list = []
dev_uid_list = [[record['uid']] for record in dev_record_data]
dev_qid_list = [[record['qid']] for record in dev_record_data]
dev_uid_feature = uid_enc.transform(dev_uid_list)
dev_qid_feature = qid_enc.transform(dev_qid_list)
for i in tqdm(range(dev_num)):
    uid = train_record_data[i]['uid']
    qid = dev_record_data[i]['qid']
    feature_vector = []

    if uid in uid_count_feature:
        feature_vector.append(accuracy_per_user_feature[uid])
        feature_vector.append(average_buzz_ratio_per_user_feature[uid])
        feature_vector.append(uid_count_feature[uid])
    else:
        feature_vector.append(accuracy_per_user_feature['<UKN>'])
        feature_vector.append(average_buzz_ratio_per_user_feature['<UKN>'])
        feature_vector.append(uid_count_feature['<UKN>'])
    if qid in qid_count_feature:
        feature_vector.append(accuracy_per_questio_feature[qid])
        feature_vector.append(average_buzz_ratio_per_question_feature[qid])
        feature_vector.append(qid_count_feature[qid])
    else:
        feature_vector.append(accuracy_per_questio_feature['<UKN>'])
        feature_vector.append(average_buzz_ratio_per_question_feature['<UKN>'])
        feature_vector.append(qid_count_feature['<UKN>'])
    feature_matrix.append(feature_vector)
    if str((uid, qid)) in times_seen_feature:
        times_seen = times_seen_feature[str((uid, qid))]
        times_seen_correct = times_seen_correct_feature[str((uid, qid))]
        times_seen_wrong = times_seen_wrong_feature[str((uid, qid))]
    else:
        times_seen = 0
        times_seen_correct = 0
        times_seen_wrong = 0
    feature_vector += [times_seen, times_seen_correct, times_seen_wrong]
    # if qid in question_dic:
    #     category_subcategory_difficulty_list.append([question_dic[qid][0], question_dic[qid][1], question_dic[qid][2]])
    # else:
    #     category_subcategory_difficulty_list.append(['<UKN>', '<UKN>', '<UKN>'])
    # if qid in text_dic:
    #     text_list.append([text_dic[qid]])
    # else:
    #     text_list.append(['<UKN>'])

    if dev_record_data[i]['ruling']:
        dev_y.append(1)
    else:
        dev_y.append(0)
feature_matrix = sp.csr_matrix(feature_matrix)
# category_subcategory_difficulty_feature = category_subcategory_difficulty_enc.transform(category_subcategory_difficulty_list)
text_feature = text_enc.transform(text_list)
# dev_x = sp.hstack((dev_uid_feature, dev_qid_feature, feature_matrix, times_seen_feature, \
#                     category_subcategory_difficulty_feature), format='csr')
dev_text_embedding_list = bert_pretrained_embeddings(DEV_QUESTION)
dev_text_embedding_matrix = sp.csr_matrix(dev_text_embedding_list)
dev_x = sp.hstack((dev_uid_feature, dev_qid_feature, feature_matrix, dev_text_embedding_matrix), format='csr')
dev_y = np.array(dev_y)

# print("before selection: ", train_x.shape, "dev: ", dev_x.shape)
# sel_var = VarianceThreshold(threshold=(.99 * (1 - .99)))
# train_x = sel.fit_transform(train_x)
# dev_x = sel.transform(dev_x)
# print("After selection: ", train_x.shape, "dev: ", dev_x.shape)
# sel_chi = SelectKBest(chi2, k=3)
# train_x = sel_chi.fit_transform(train_x, train_y)
# dev_x = sel_chi.transform(dev_x)

# X = np.array([[0.5, 0.6], [0.7, 0.5], [0.5, 0.5], [0.2, 0.2]])
# y = np.array([0, 0, 1, 1])

clf1 = LogisticRegression(penalty = 'l2', C = 0.00002, solver='lbfgs', max_iter=1000, random_state=123)
print('train_x', type(train_x), 'size', train_x.shape)
print('train_y', type(train_y), 'size', train_y.shape)
print('train_x[0]', type(train_x[0]), 'size', train_x[0].shape)
clf1.fit(train_x, train_y)
probas = clf1.predict_proba(train_x)
y_pred = [0 if pr[0] > pr [1] else 1 for pr in probas]
accuracy = accuracy_score(train_y, y_pred)
print("Accuracy (train): %0.2f%% " % (accuracy * 100))
probas = clf1.predict_proba(dev_x)
y_pred = [0 if pr[0] > pr [1] else 1 for pr in probas]
# print("probas", probas)
print("weights", clf1.coef_[0])
weight_index = clf1.coef_[0].argsort()
for i in range(50):
    print(weight_index[-(i+1)], '\t', clf1.coef_[0][weight_index[-(i+1)]])
print("bias", clf1.intercept_ )
accuracy = accuracy_score(dev_y, y_pred)
print("Accuracy (dev): %0.2f%% " % (accuracy * 100))

