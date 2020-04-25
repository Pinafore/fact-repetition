#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd

category_whitelist = [
    'SCIENCE',
    'LITERATURE',
    'AMERICAN HISTORY',
    'SPORTS',
    'HISTORY',
    'WORLD GEOGRAPHY',
    'BUSINESS & INDUSTRY',
    'ART',
    'TRANSPORTATION',
    'COLLEGES & UNIVERSITIES',
    'RELIGION',
    'ANIMALS',
    'AUTHORS',
    'GEOGRAPHY',
    'BODIES OF WATER',
    'ISLANDS',
    'BOOKS & AUTHORS',
    'FOOD',
    'OPERA',
    'LANGUAGES',
    'SHAKESPEARE',
    'CLASSICAL MUSIC',
    'PEOPLE',
    'TELEVISION',
    'MYTHOLOGY',
    'MUSIC',
    'BIOLOGY',
    'MUSEUMS',
    'NONFICTION',
    'FICTIONAL CHARACTERS',
    'COMPOSERS',
    'ASTRONOMY',
    'POETS & POETRY',
    'ARCHITECTURE',
    'ORGANIZATIONS',
    'MOUNTAINS',
    'ZOOLOGY',
    'FASHION',
    'ANATOMY',
    'CHEMISTRY',
    'WORLD LEADERS',
    'SCIENTISTS',
    'AWARDS',
    'ARTISTS',
    'POP CULTURE',
    'GOVERNMENT & POLITICS',
    'HEALTH & MEDICINE',
    'TECHNOLOGY',
    'PHYSICS',
    'SCULPTURE',
    'WEATHER',
    'LANDMARKS',
    'INVENTORS',
    'GEOLOGY',
    'COOKING',
    'POLITICIANS',
    'ARCHAEOLOGY',
    'PHILOSOPHY',
    'ACTRESSES',
    'NOVELS',
    'MATH',
    'MOVIES',
    'GAMES',
    'ECONOMICS',
    'INVENTIONS',
    'PSYCHOLOGY',
]

DATA_DIR = '/fs/clip-quiz/shifeng/karl/data/'

# load jeopardy data
with open(DATA_DIR + 'jeopardy_358974_questions_20190612.pkl', 'rb') as f:
    questions_df = pickle.load(f)
with open(DATA_DIR + 'jeopardy_310326_question_player_pairs_20190612.pkl', 'rb') as f:
    records_df = pickle.load(f)

# merge question_df and records_df into one
questions_df['karl_id'] = questions_df.index  # what KARL db uses as ID
records_df = questions_df.set_index('question_id').join(records_df.set_index('question_id'))
records_df['question_id'] = records_df.index
# get number of records per question
s = records_df['question_id'].value_counts()
counts_df = pd.DataFrame({'question_id': s.index, 'count': s.values})
records_df = records_df.join(counts_df.set_index('question_id'))
# filter questions with no more than one records
records_df = records_df[records_df['count'] > 1]
# rank questions within each category by number of records
records_df_grouped = records_df.groupby('category')
ranking_dict = {}
for category in category_whitelist:
    group = records_df_grouped.get_group(category)
    ranking_dict.update(group['count'].rank(ascending=False, method='min').to_dict())
records_df['category_ranking'] = records_df['question_id'].apply(lambda x: ranking_dict.get(x, None))
# remove questions ranked lower than top 20
# records_df = records_df[records_df['category_ranking'] <= 30]

records_grouped = records_df.drop('question_id', axis=1).groupby('question_id')
flashcards = []
# if True:
#     question_id = '1014-6-5'
#     records_group = records_grouped.get_group(question_id)
for question_id, records_group in records_grouped:
    question = records_group.iloc[0]
    flashcards.append({
        'text': question['text'],
        'answer': question['answer'],
        'record_id': question_id,
        'card_id': int(question['karl_id']),
        'category': question['category'],
    })

print("extracted {} cards".format(len(flashcards)))
print("with {} records".format(len(records_df)))

with open(DATA_DIR + 'diagnostic_questions.pkl', 'wb') as f:
    pickle.dump(flashcards, f)
with open(DATA_DIR + 'diagnostic_records.pkl', 'wb') as f:
    pickle.dump(records_df, f)
