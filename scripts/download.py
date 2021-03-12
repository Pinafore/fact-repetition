# download preprocessed data
import os
import wget
import pathlib

DATA_DIR = 'data/protobowl/'
S3_DIR = 'https://pinafore-us-west-2.s3-us-west-2.amazonaws.com/karl/protobowl/'

pathlib.Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

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


DATA_DIR = 'output/'
S3_DIR = 'https://pinafore-us-west-2.s3-us-west-2.amazonaws.com/karl/'
print('downloading {} from s3'.format(f))
f = 'retention_hf_distilbert_new_card.zip'
wget.download(S3_DIR + f, DATA_DIR + f)
f = 'retention_hf_distilbert_old_card.zip'
wget.download(S3_DIR + f, DATA_DIR + f)
