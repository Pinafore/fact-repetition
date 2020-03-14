import os
import pickle
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.fields import Schema, ID, TEXT

from util import Params


class Scheduler(ABC):

    @abstractmethod
    def reset(self):
        # reset model and return init state
        pass

    @abstractmethod
    def predict(self, cards: List[dict]) -> List[dict]:
        # estimate difficulty of cards
        pass

    @abstractmethod
    def predict_one(self, card: dict) -> float:
        # estimate difficulty of cards
        pass

    @abstractmethod
    def embed(self, cards: List[dict]) -> List[dict]:
        # embed card in topic space. return augmented cards
        pass

    @abstractmethod
    def set_params(self, params: dict):
        pass

    @abstractmethod
    def schedule(self, cards: List[dict]) -> Tuple[List[int], List[int], str]:
        # order, ranking, rationale
        pass

    @abstractmethod
    def update(self, cards: List[dict]):
        pass


# find this in util.py
# class Params(TypedDict):
#     n_components: int = 20
#     qrep: float = 0.1
#     prob: float = 0.7
#     category: float = 0.3
#     leitner: float = 1.0
#     sm2: float = 1.0
#     step_correct: float = 0.5
#     step_wrong: float = 0.05
#     step_qrep: float = 0.3
#     vectorizer: str = 'checkpoints/tf_vectorizer.pkl'
#     lda: str = 'checkpoints/lda.pkl'
#     whoosh_index: str = 'whoosh_index'


class MovingAvgScheduler(Scheduler):

    def __init__(self, params: Params = None):
        if params is None:
            params = Params()
        self.params = params
        # TODO change to logger
        print('loading question and records...')
        with open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb') as f:
            records_df = pickle.load(f)
        with open('data/jeopardy_358974_questions_20190612.pkl', 'rb') as f:
            questions_df = pickle.load(f)
        self.karl_to_question_id = questions_df.to_dict()['question_id']

        # augment question_df with record stats
        print('merging dfs...')
        df = questions_df.set_index('question_id').join(records_df.set_index('question_id'))
        df = df[df['correct'].notna()]
        df['correct'] = df['correct'].apply(lambda x: 1 if x == 1 else 0)
        df_grouped = df.reset_index()[['question_id', 'correct']].groupby('question_id')
        dict_correct_mean = df_grouped.mean().to_dict()['correct']
        dict_records_cnt = df_grouped.count().to_dict()['correct']
        questions_df['prob'] = questions_df['question_id'].apply(lambda x: dict_correct_mean.get(x, None))
        questions_df['count'] = questions_df['question_id'].apply(lambda x: dict_records_cnt.get(x, None))
        self.questions_df = questions_df
        self.question_id_set = set(self.questions_df['question_id'])

        # setup whoosh for text search
        if not os.path.exists(self.params.whoosh_index):
            print('building whoosh...')
            self.build_whoosh()
        self.ix = open_dir(self.params.whoosh_index)

        diagnostic_file = 'data/diagnostic_questions.pkl'
        # setup LDA
        if not (os.path.exists(self.params.vectorizer) and os.path.exists(self.params.lda)):
            with open(diagnostic_file, 'rb') as f:
                diagnostic_cards = pickle.load(f)
            print('building lda...')
            self.build_lda(diagnostic_cards)
        with open(self.params.vectorizer, 'rb') as f:
            self.tf_vectorizer = pickle.load(f)
        with open(self.params.lda, 'rb') as f:
            self.lda = pickle.load(f)

        # estimate initial state
        with open(diagnostic_file, 'rb') as f:
            diagnostic_cards = pickle.load(f)
        self.reset(self.estimate_avg(diagnostic_cards))

        # setup Leitner and SM2
        self.leitner = Leitner()
        self.sm2 = SM2()

        print('scheduler ready')

    def estimate_avg(self, cards: List[dict]) -> List[float]:
        # estimate the average acccuracy for each component
        # use for initializing user estimate
        estimate_file = 'data/diagnostic_avg_estimate.txt'
        if os.path.exists(estimate_file):
            with open(estimate_file) as f:
                return [float(x) for x in f.readlines()]

        texts = [x['text'] for x in cards]
        qreps = self.lda.transform(self.tf_vectorizer.transform(texts))
        estimates = [[] for _ in range(self.params.n_components)]
        for card, qrep in zip(cards, qreps):
            topic_idx = np.argmax(qrep)
            prob = self.predict_one(card)
            estimates[topic_idx].append(prob)
        estimates = [np.mean(x) for x in estimates]
        with open('data/diagnostic_avg_estimate.txt', 'w') as f:
            for e in estimates:
                f.write(str(e) + '\n')
        return estimates

    def reset(self, prob: Optional[List[float]]):
        # round number since start
        self.round_num = 0
        # topical representation
        k = self.params.n_components
        self.qrep = np.array([1 / k for _ in range(k)])
        # average difficulty (user accuracy) of questions seen so far
        if prob is None:
            prob = [0.5 for _ in range(k)]
        self.prob = prob
        self.category = None

        # cache of embeddings
        self.qrep_cache = dict()
        # cache of card probability
        self.prob_cache = dict()
        # cache of previous study round number
        self.prev_cache = dict()

    def build_lda(self, cards: List[dict]):
        texts = [card['text'] for card in cards]
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=50000,
                                        stop_words='english')
        tf = tf_vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=self.params.n_components,
                                        max_iter=5, learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda.fit(tf)
        # tf_feature_names = tf_vectorizer.get_feature_names()
        with open(self.params.vectorizer, 'wb') as f:
            pickle.dump(tf_vectorizer, f)
        with open(self.params.lda, 'wb') as f:
            pickle.dump(lda, f)

    def build_whoosh(self):
        if not os.path.exists(self.params.whoosh_index):
            os.mkdir(self.params.whoosh_index)
        schema = Schema(
            question_id=ID(stored=True),
            text=TEXT(stored=True),
            answer=TEXT(stored=True)
        )
        ix = create_in(self.params.whoosh_index, schema)
        writer = ix.writer()

        for idx, q in tqdm(self.questions_df.iterrows()):
            writer.add_document(
                question_id=q['question_id'],
                text=q['text'],
                answer=q['answer']
            )
        writer.commit()

    def embed(self, cards: List[dict]) -> List[dict]:
        to_be_embedded = []
        for i, card in enumerate(cards):
            qid = card['question_id']
            if 'qrep' in card:
                self.qrep_cache[qid] = card['qrep']
            elif qid in self.qrep_cache:
                cards[i]['qrep'] = self.qrep_cache[qid]
            else:
                to_be_embedded.append(card)
        if len(to_be_embedded) == 0:
            return cards
        self._embed(to_be_embedded)
        for i, card in enumerate(cards):
            if 'qrep' not in card:
                cards[i]['qrep'] = self.qrep_cache[card['question_id']]
        return cards

    def _embed(self, cards: List[dict]):
        texts = [x['text'] for x in cards]
        qreps = self.lda.transform(self.tf_vectorizer.transform(texts))
        self.qrep_cache.update({x['question_id']: q for x, q in zip(cards, qreps)})

    def embed_one(self, card: dict) -> np.ndarray:
        if 'qrep' in card:
            self.qrep_cache[card['question_id']] = card['qrep']
            return card['qrep']
        if card['question_id'] in self.qrep_cache:
            return self.qrep_cache[card['question_id']]
        qrep = self.lda.transform(self.tf_vectorizer.transform([card['text']]))
        self.qrep_cache[card['question_id']] = qrep
        return qrep

    def predict(self, cards: List[dict]) -> List[dict]:
        for i, card in enumerate(cards):
            cards[i]['prob'] = self.predict_one(card)
        return cards

    def retrieve(self, card: dict) -> Tuple[List[dict], List[float]]:
        record_id = self.karl_to_question_id[int(card['question_id'])]

        # 1. try to find in records with gameid-catnum-level
        if record_id in self.question_id_set:
            hits = self.questions_df[self.questions_df.question_id == record_id]
            if len(hits) > 0:
                cards = [card.to_dict() for idx, card in hits.iterrows()]
                return cards, [1 / len(hits) for _ in range(len(hits))]

        # 2. do text search
        with self.ix.searcher() as searcher:
            query = QueryParser("text", self.ix.schema).parse(card['text'])
            hits = searcher.search(query)
            hits = [x for x in hits if x['answer'] == card['answer']]
            if len(hits) > 0:
                scores = [x.score for x in hits]
                ssum = np.sum(scores)
                scores = [x / ssum for x in scores]
                cards = [self.questions_df[self.questions_df['question_id'] == x['question_id']].iloc[0] for x in hits]
                return cards, scores

        # 3. not found
        return [], []

    def predict_one(self, card: dict) -> float:
        if 'prob' in card:
            self.prob_cache[card['question_id']] = card['prob']
            return card['prob']
        if card['question_id'] in self.prob_cache:
            return self.prob_cache[card['question_id']]
        prob = self._predict_one(card)
        self.prob_cache[card['question_id']] = prob
        return prob

    def _predict_one(self, card: dict) -> float:
        # 1. find same or similar card in records
        cards, scores = self.retrieve(card)
        if len(cards) > 0:
            return np.dot([x['prob'] for x in cards], scores)
        # 2. use model to predict
        # TODO
        return 0

    def set_params(self, params: dict):
        self.params.__dict__.update(params)

    def dist_rep(self, card: dict) -> float:
        # distance penalty due to repetition
        prev = self.prev_cache.get(card['question_id'], self.round_num)
        return self.round_num - prev

    def dist_category(self, card: dict) -> float:
        if self.category is None:
            return 0
        return int(card['category'] != self.category)

    def dist_prob(self, card: dict) -> float:
        if 'qrep' not in card:
            card['qrep'] = self.embed_one(card)
        topic_idx = np.argmax(card['qrep'])
        d = card['prob'] - self.prob[topic_idx]
        # penalize easier questions
        d *= 1 if d < 0 else 10
        return abs(d)

    def dist_qrep(self, card: dict) -> float:
        def cosine_distance(a, b):
            return 1 - (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
        return cosine_distance(self.qrep, card['qrep'])

    def dist_leitner(self, card: dict) -> float:
        # distance to leitner scheduled time
        t = self.leitner.scheduled_time.get(card['question_id'], None)
        return 0 if t is None else (t - datetime.now()).days

    def dist_sm2(self, card: dict) -> float:
        # distance to sm2 scheduled time
        t = self.sm2.scheduled_time.get(card['question_id'], None)
        return 0 if t is None else (t - datetime.now()).days

    def score_one(self, card: dict) -> Dict[str, float]:
        # compute distance metric of given card to current state
        # dist = topic repr dist + topical difficulty dist + recency + leitner score
        # NOTE lower is better
        card['qrep'] = self.embed_one(card)
        card['prob'] = self.predict_one(card)
        return {
            'qrep': self.dist_qrep(card),
            'prob': self.dist_prob(card),
            'category': self.dist_category(card),
            'leitner': self.dist_leitner(card),
            'sm2': self.dist_sm2(card),
        }

    def score(self, cards: List[dict]) -> List[float]:
        cards = self.embed(self.predict(cards))
        return [self.score_one(card) for card in cards]

    def schedule(self, cards: List[dict]) -> Tuple[List[int], List[int], str]:
        if len(cards) == 0:
            return [], [], ''

        scores = self.score(cards)
        scores_summed = [
            sum([self.params[key] * value for key, value in ss.items()])
            for ss in scores
        ]
        for i, card in enumerate(cards):
            # NOTE lower is better
            cards[i]['score'] = scores_summed[i]
            cards[i]['scores'] = scores[i]
        reverse_index = {x['question_id']: i for i, x in enumerate(cards)}
        cards_sorted = sorted(cards, key=lambda x: x['score'])
        # sorted indices
        order = [reverse_index[x['question_id']] for x in cards_sorted]
        ranking = [order.index(i) for i, _ in enumerate(cards)]

        card = cards_sorted[0]
        info = card['scores']
        topic_idx = np.argmax(card['qrep'])
        info.update({
            'sum': card['score'],
            'user': self.prob[topic_idx],
            'card': card['prob'],
            'topic': topic_idx,
        })
        rationale = '\n'.join([
            '{}: {:.3f}'.format(key, value)
            for key, value in info.items()
        ])
        return order, ranking, rationale

    def update(self, cards: List[dict]):
        cards = self.embed(self.predict(cards))
        for card in cards:
            topic_idx = np.argmax(card['qrep'])
            if card['label'] == 'correct':
                a = self.params['step_correct']
                self.prob[topic_idx] = a * card['prob'] + (1 - a) * self.prob[topic_idx]
            else:
                self.prob[topic_idx] += self.params['step_wrong']
            self.prob[topic_idx] = min(1.0, self.prob[topic_idx])
            b = self.params['step_qrep']
            self.qrep = b * card['qrep'] + (1 - b) * self.qrep
            self.category = card['category']
        # TODO update Leitner and SM2


class Leitner:

    def __init__(self):
        self.scheduled_time = dict()
        self.card_to_box = dict()
        increment_days = [0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 9999]
        self.increment_days = {i: x for i, x in enumerate(increment_days)}

    def update(self, card: dict):
        qid = card['question_id']
        curr_box = self.card_to_box.get(qid, None)
        if curr_box is None:
            self.card_to_box[qid] = 1
            curr_box = 0
        new_box = curr_box + (1 if card['label'] == 'correct' else -1)
        new_box = max(min(new_box, 9), 0)
        self.card_to_box[qid] = new_box
        interval = timedelta(days=self.increment_days[new_box])
        print(new_box, interval)
        self.scheduled_time[qid] = datetime.now() + interval


class SM2:

    def __init__(self):
        self.scheduled_time = dict()
        self.study = dict()

    def get_quality_from_response(self, response: str) -> int:
        return 4 if response == 'correct' else 1

    def update(self, card: dict):
        date_studied = datetime.now()
        qid = card['question_id']
        quality = self.get_quality_from_response(card['label'])

        if card['label'] != 'correct':
            self.scheduled_time[qid] = date_studied
            self.study[qid] = (2.5, 0, 0)
            # return self.study[qid], self.scheduled_time[qid]

        if qid not in self.study:
            self.scheduled_time[qid] = date_studied
            self.study[qid] = (2.5, 1, 1)
            # return self.study[qid], self.scheduled_time[qid]

        e_factor, repetition, interval = self.study[qid]
        e_factor = max(1.3, e_factor + 0.1 - (5.0 - quality) * (0.08 + (5.0 - quality) * 0.02))
        repetition += 1

        if repetition == 1:
            interval = 1
        elif repetition == 2:
            interval = 6
        else:
            interval *= e_factor

        self.scheduled_time[qid] = datetime.now() + timedelta(days=interval)
        self.study[qid] = (e_factor, repetition, interval)
