# download preprocessed data
import os
import wget
import pathlib

DATA_DIR = 'data/'
S3_DIR = 'https://pinafore-us-west-2.s3-us-west-2.amazonaws.com/karl/'

pathlib.Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

files = [
    'protobowl/protobowl-042818.log.train.h5',
    'protobowl/protobowl-042818.log.test.h5',
    'protobowl/x_train.npy',
    'protobowl/y_train.npy',
    'protobowl/x_test.npy',
    'protobowl/y_test.npy',
    # 'protobowl/protobowl-042818.log.h5',
    'protobowl/protobowl-042818.log.questions.pkl',
    'diagnostic_questions.pkl',
]

for f in files:
    if not os.path.exists(DATA_DIR + f):
        print('downloading {} from s3'.format(f))
        wget.download(S3_DIR + f, DATA_DIR + f)
        print()


DATA_DIR = 'output/'
S3_DIR = 'https://pinafore-us-west-2.s3-us-west-2.amazonaws.com/karl/'
files = [
    'retention_hf_distilbert_old_card.zip',
    'retention_hf_distilbert_new_card.zip',
]

for f in files:
    if not os.path.exists(DATA_DIR + f):
        print('downloading {} from s3'.format(f))
        wget.download(S3_DIR + f, DATA_DIR + f)
        print()
