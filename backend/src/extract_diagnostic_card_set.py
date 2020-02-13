import json
import pickle
import requests
import collections
import pandas as pd
import numpy as np

n_questions_per_category = 20

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

# load jeopardy data
jeopardy_questions_df = pickle.load(open('data/jeopardy_358974_questions_20190612.pkl', 'rb'))
jeopardy_records_df = pickle.load(open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb'))

jeopardy_questions_dict = jeopardy_questions_df.to_dict()
jeopardy_questions = {}
for i in jeopardy_questions_dict['clue'].keys():
    jeopardy_questions[jeopardy_questions_dict['questionid'][i]] = {
        'text': jeopardy_questions_dict['clue'][i],
        'answer': str(jeopardy_questions_dict['answer'][i]),
    }
    
category_mapping = {}
for row in jeopardy_questions_df.itertuples():
    category_mapping[row.questionid] = row.category
jeopardy_records_df['category'] = jeopardy_records_df['question_id'].apply(lambda x: category_mapping.get(x, None))

# # most popular categories
# category_blacklist = ['POTPOURRI', 'BEFORE & AFTER,']
# category_counter = collections.Counter(jeopardy_questions_df.category.tolist())
# top_categories = [x[0] for x in category_counter.most_common()]
# top_categories = [x for x in top_categories if x != 'POTPOURRI'][:n_categories]
# print(top_categories)

top_categories = category_whitelist
print('# categories', len(top_categories))

# most popular questions within each category
selected_qids = []
for category in top_categories:
    df = jeopardy_records_df[jeopardy_records_df.category == category]
    question_counter = collections.Counter(df.question_id)
    selected_qids += [x[0] for x in question_counter.most_common(n_questions_per_category)]

# filter records and get fake label
jeopardy_records_filtered_df = jeopardy_records_df[jeopardy_records_df.question_id.isin(selected_qids)]
jeopardy_set_df = jeopardy_records_filtered_df[['question_id', 'correct']]
jeopardy_set_df['correct'] = jeopardy_set_df['correct'].apply(lambda x: int(x) / 2 + 0.5)
jeopardy_set_df = jeopardy_set_df.groupby('question_id').mean()
jeopardy_set_df['correct'] = jeopardy_set_df['correct'].apply(lambda x: 1 if x > 0.6 else -1)
    
# create set of flashcards from study record
max_cards = 1000
flashcards = []
for row in jeopardy_set_df.itertuples():
    if len(flashcards) >= max_cards:
        break
    qid = row.Index
    flashcards.append({
        'text': jeopardy_questions[qid]['text'],
        'user_id': 'diagnosis',
        'question_id': qid,
        'label': 'correct' if row.correct == 1 else 'wrong',
        'answer': jeopardy_questions[qid]['answer'],
        'category': category_mapping[qid],
        })
print('# flashcards', len(flashcards))
print()

with open('diagnostic_card_set.pkl', 'wb') as f:
    pickle.dump(flashcards, f)
