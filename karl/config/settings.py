# ROOT_DIR = '/Users/shifeng/workspace'
ROOT_DIR = '/fs/clip-quiz/shifeng'
CODE_DIR = f'{ROOT_DIR}/karl-dev'
DATA_DIR = f'{CODE_DIR}/data'
DB_HOST = f'{ROOT_DIR}/karl-db/run'
SQLALCHEMY_DATABASE_URL = f'postgresql+psycopg2://shifeng@localhost:5433/karl-prod?host={DB_HOST}'
# SQLALCHEMY_DATABASE_URL = 'postgresql+psycopg2://shifeng@localhost:5433/karl-prod'
