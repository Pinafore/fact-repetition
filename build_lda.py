import os
import time
import json
import pickle
import argparse
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def get_texts(data_type: str) -> List[str]:
    texts = []
    if data_type in ['quizbowl', 'all']:
        with open('data/withanswer.question.json', 'r') as f:
            qs = json.load(f)
        texts += [x['text'] for x in qs]
    if data_type in ['jeopardy', 'all']:
        with open('data/jeopardy_358974_questions_20190612.pkl', 'rb') as f:
            df = pickle.load(f)
            df = df.dropna()
        texts += df.text.to_list()
    if data_type in ['diagnostic']:
        with open('data/diagnostic_questions.pkl', 'rb') as f:
            cards = pickle.load(f)
        texts += [card['text'] for card in cards]
    return texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Data to train LDA on",
                        choices=['quizbowl', 'jeopardy', 'diagnostic', 'all'],
                        required=True)
    parser.add_argument("--max_df", type=float, default=0.80)
    parser.add_argument("--min_df", type=float, default=3)
    parser.add_argument("--vocab", dest='max_features', type=int, required=True)
    parser.add_argument("--topic", dest='n_components', type=int, required=True)
    args = parser.parse_args()
    print(vars(args))
    print()

    root_dir = '/fs/clip-scratch/shifeng/karl/checkpoints/'
    model_name= '{}_{}_{}_{}'.format(
        args.data, args.max_features, args.n_components, time.time())
    model_dir = os.path.join(root_dir, model_name)
    os.mkdir(model_dir)
    print(model_dir)
    print()

    texts = get_texts(args.data)
    vectorizer = CountVectorizer(max_df=args.max_df,
                                 min_df=args.min_df,
                                 max_features=args.max_features,
                                 stop_words='english')
    lda = LatentDirichletAllocation(n_components=args.n_components,
                                    max_iter=100,
                                    learning_method='online',
                                    evaluate_every=3,
                                    verbose=1,
                                    random_state=0)
    lda.fit(vectorizer.fit_transform(texts))

    with open(os.path.join(model_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(model_dir, 'lda.pkl'), 'wb') as f:
        pickle.dump(lda, f)
    
    print()
    feature_names = vectorizer.get_feature_names()
    for topic in lda.components_:
        topic = topic.argsort()[:-11:-1]
        print(' '.join([feature_names[i] for i in topic]))
        print()
