CODE_DIR = '/fs/clip-quiz/shifeng/karl-dev'
DATA_DIR = f'{CODE_DIR}/data'
DB_HOST = '/scratch0/shifeng/karl-db/run'
SQLALCHEMY_DATABASE_URL = f'postgresql+psycopg2://shifeng@localhost:5433/karl-prod?host={DB_HOST}'
STABLE_DATABASE_URL = 'postgresql+psycopg2://shifeng@4.tcp.ngrok.io:17010/karl-prod'