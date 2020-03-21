import os
import time
import json
import pickle
import argparse
from typing import List
from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as SklearnLDA

import gensim
from gensim.models import LdaModel as GensimLDA
# from gensim.models import CoherenceModel


import en_core_web_lg
nlp = en_core_web_lg.load()


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

def process(doc):
    tokens = []
    entity = ''
    proper_noun = ''
    for token in doc:
        if token.ent_iob == 1:
            # middle of entity
            if not token.is_punct:
                entity += ' ' + token.text
            continue
        elif token.ent_iob == 3:
            # end of entity
            entity = token.text
            continue
        # not entity
        if len(entity) > 0:
            tokens.append(entity)
            entity = ''

        if token.pos_ == 'PROPN':
            if len(proper_noun) > 0:
                proper_noun += ' ' + token.text
            else:
                proper_noun = token.text
            continue
        elif len(proper_noun) > 0:
            tokens.append(proper_noun)
            proper_noun = ''

        if token.is_stop or token.is_punct or token.is_digit or not token.is_alpha:
            continue
        else:
            tokens.append(token.lemma_.lower())
    return tokens


def build_mallet(texts, args, model_dir):
    # TODO
    pass


def build_gensim(texts, args, model_dir):
    nlp.add_pipe(process, name='process', last=True)
    processed_docs = list(nlp.pipe(texts, batch_size=2500, n_threads=16))
    vocab = gensim.corpora.Dictionary(processed_docs)
    vocab.filter_extremes(no_below=args.min_df, no_above=args.max_df)
    corpus = [vocab.doc2bow(doc) for doc in processed_docs]

    print('Number of unique tokens: %d' % len(vocab))
    print('Number of documents: %d' % len(corpus))

    _ = vocab[0]
    id2word = vocab.id2token

    lda = GensimLDA(
        corpus=corpus,
        id2word=id2word,
        chunksize=2000,
        alpha='auto',
        eta='auto',
        iterations=400,
        num_topics=args.n_topics,
        passes=20,
        eval_every=None
    )

    lda.save(os.path.join(model_dir, 'model'))

    top_topics = lda.top_topics(corpus)  # , num_words=20)

    avg_topic_coherence = sum([t[1] for t in top_topics]) / args.n_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    pprint(top_topics)


def build_sklearn(texts, args, model_dir):
    vectorizer = CountVectorizer(max_df=args.max_df,
                                 min_df=args.min_df,
                                 max_features=args.n_vocab,
                                 stop_words='english')
    lda = SklearnLDA(n_components=args.n_topics,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Data to train LDA on',
                        choices=['quizbowl', 'jeopardy', 'diagnostic', 'all'],
                        required=True)
    parser.add_argument('--model', choices=['sklearn', 'gensim'])
    parser.add_argument('--max_df', type=float, default=0.80)
    parser.add_argument('--min_df', type=float, default=3)
    parser.add_argument('--n_vocab', type=int, required=True)
    parser.add_argument('--n_topics', type=int, required=True)
    args = parser.parse_args()
    print(vars(args))
    print()

    root_dir = '/fs/clip-scratch/shifeng/karl/checkpoints/'
    model_name = '{}_{}_{}_{}'.format(args.model, args.data, args.n_topics, time.time())
    model_dir = os.path.join(root_dir, model_name)
    os.mkdir(model_dir)
    print(model_dir)
    print()

    texts = get_texts(args.data)

    if args.model == 'sklearn':
        build_sklearn(texts, args, model_dir)
    elif args.model == 'gensim':
        build_gensim(texts, args, model_dir)
