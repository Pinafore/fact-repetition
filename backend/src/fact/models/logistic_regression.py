import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import json
from tqdm import tqdm
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
QUESTION_FILE = './question.json'

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

def times_seen(df):
    cumcount_array = df.groupby(['uid', 'qid']).cumcount().values
    feature = [[cumcount] for cumcount in cumcount_array]
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
    vectorizer = CountVectorizer()
    text_feature = vectorizer.fit_transform(text_list)
    qid = [record['qid'] for record in question_data]
    feature = dict(zip(uid, text_feature))
    return feature

start_time = time.time()

for k in range(len(FILE_NAME)):
    if k > 0:
        break
    # load the dataset
    with open(FILE_PATH[k] + 'train.' + FILE_NAME[k]) as f:
        train_record_data = json.load(f)
    with open(FILE_PATH[k] + 'dev.' + FILE_NAME[k]) as f:
        dev_record_data = json.load(f)
    with open(FILE_PATH[k] + 'test.' + FILE_NAME[k]) as f:
        test_record_data = json.load(f)
    with open(QUESTION_FILE) as f:
        question_data = json.load(f)
        
    test_num = len(train_record_data)
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
    times_seen_feature = times_seen(df)
    category_subcategory_difficulty_enc = category_subcategory_difficulty()
    # count_vec = bag_of_words()
    question_dic = {q['qid']: [q['category'], q['subcategory'], q['difficulty']] for q in question_data}
    print("qid length: ", len(list(pd.DataFrame(train_record_data + dev_record_data + test_record_data).groupby('qid').groups.keys())))
    # print("After transdorming: --- %s seconds ---" % (time.time() - start_time))
    train_x = []
    train_y = []
    feature_matrix = []
    category_subcategory_difficulty_list = []
    text_list = []
    for i in tqdm(range(test_num)):
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
        if qid in question_dic:
            category_subcategory_difficulty_list.append([question_dic[qid][0], question_dic[qid][1], question_dic[qid][2]])
        else:
            category_subcategory_difficulty_list.append(['<UKN>', '<UKN>', '<UKN>'])
        if train_record_data[i]['ruling']:
            train_y.append(1)
        else:
            train_y.append(0)
    feature_matrix = sp.csr_matrix(feature_matrix)
    category_subcategory_difficulty_feature = category_subcategory_difficulty_enc.transform(category_subcategory_difficulty_list)
    train_x = sp.hstack((train_uid_feature, train_qid_feature, feature_matrix, times_seen_feature,  \
                            category_subcategory_difficulty_feature), format='csr')
    train_y = np.array(train_y)


    dev_x = []
    dev_y = []
    feature_matrix = []
    category_subcategory_difficulty_list = []
    dev_uid_list = [[record['uid']] for record in dev_record_data]
    dev_qid_list = [[record['qid']] for record in dev_record_data]
    dev_uid_feature = uid_enc.transform(dev_uid_list)
    dev_qid_feature = qid_enc.transform(dev_qid_list)
    times_seen_feature = times_seen(pd.concat([pd.DataFrame(train_record_data), pd.DataFrame(dev_record_data)]))[len(train_record_data):]
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
        if qid in question_dic:
            category_subcategory_difficulty_list.append([question_dic[qid][0], question_dic[qid][1], question_dic[qid][2]])
        else:
            category_subcategory_difficulty_list.append(['<UKN>', '<UKN>', '<UKN>'])

        if dev_record_data[i]['ruling']:
            dev_y.append(1)
        else:
            dev_y.append(0)
    feature_matrix = sp.csr_matrix(feature_matrix)
    category_subcategory_difficulty_feature = category_subcategory_difficulty_enc.transform(category_subcategory_difficulty_list)
    dev_x = sp.hstack((dev_uid_feature, dev_qid_feature, feature_matrix, times_seen_feature, \
                        category_subcategory_difficulty_feature), format='csr')
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

    clf1 = LogisticRegression(penalty = 'l2', C = 0.01, solver='lbfgs', max_iter=1000, random_state=123)
# print('train_x', type(train_x), 'size', train_x.shape)
# print('train_y', type(train_y), 'size', train_y.shape)
# print('train_x[0]', type(train_x[0]), 'size', train_x[0].shape)
    clf1.fit(train_x, train_y)
    probas = clf1.predict_proba(train_x)
    y_pred = [0 if pr[0] > pr [1] else 1 for pr in probas]
    accuracy = accuracy_score(train_y, y_pred)
    print("Accuracy (train): %0.2f%% " % (accuracy * 100))
    probas = clf1.predict_proba(dev_x)
    y_pred = [0 if pr[0] > pr [1] else 1 for pr in probas]
    # print("probas", probas)
    print("weights", clf1.coef_)
    print("bias", clf1.intercept_ )
    accuracy = accuracy_score(dev_y, y_pred)
    print("Accuracy (dev): %0.2f%% " % (accuracy * 100))

