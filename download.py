# download preprocessed data
import os
import wget
import pathlib

DATA_DIR = 'data/protobowl/'
S3_DIR = 'https://pinafore-us-west-2.s3-us-west-2.amazonaws.com/karl/protobowl/'

pathlib.Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

if not os.path.exists('data'):
    os.mkdir('data')

files = [
    'protobowl-042818.log.train.h5',
    'protobowl-042818.log.test.h5',
    'x_train.npy',
    'y_train.npy',
    'x_test.npy',
    'y_test.npy',
    # 'protobowl-042818.log.h5',
    'protobowl-042818.log.questions.pkl',
]

for f in files:
    if not os.path.exists(DATA_DIR + f):
        print('downloading {} from s3'.format(f))
        wget.download(S3_DIR + f, DATA_DIR + f)
        print()
