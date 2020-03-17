import os
import json
import pickle
import numpy as np
import logging
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.fields import Schema, ID, TEXT

from util import Params, Card, User, History
from db import SchedulerDB

logger = logging.getLogger('scheduler')


# find this in util.py
# class Params(TypedDict):
#     n_components: int = 20
#     qrep: float = 0.1
#     skill: float = 0.7
#     category: float = 0.3
#     leitner: float = 1.0
#     sm2: float = 1.0
#     step_correct: float = 0.5
#     step_wrong: float = 0.05
#     step_qrep: float = 0.3
#     vectorizer: str = 'checkpoints/tf_vectorizer.pkl'
#     lda: str = 'checkpoints/lda.pkl'
#     whoosh_index: str = 'whoosh_index'


class MovingAvgScheduler:

    def __init__(self, params: Params = None, db_filename='db.sqlite'):
        if params is None:
            params = Params()
        self.params = params
        self.db_filename = db_filename
        self.db = SchedulerDB(db_filename)

        logger.info('loading question and records...')
        with open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb') as f:
            records_df = pickle.load(f)
        with open('data/jeopardy_358974_questions_20190612.pkl', 'rb') as f:
            questions_df = pickle.load(f)
        self.karl_to_question_id = questions_df.to_dict()['question_id']

        # augment question_df with record stats
        logger.info('merging dfs...')
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
            logger.info('building whoosh...')
            self.build_whoosh()
        self.ix = open_dir(self.params.whoosh_index)

        diagnostic_file = 'data/diagnostic_questions.pkl'
        # setup LDA
        if not (os.path.exists(self.params.vectorizer) and os.path.exists(self.params.lda)):
            with open(diagnostic_file, 'rb') as f:
                diagnostic_cards = pickle.load(f)
            logger.info('building lda...')
            self.build_lda(diagnostic_cards)
        with open(self.params.vectorizer, 'rb') as f:
            self.tf_vectorizer = pickle.load(f)
        with open(self.params.lda, 'rb') as f:
            self.lda = pickle.load(f)

        logger.info('scheduler ready')

    def estimate_avg(self) -> np.ndarray:
        # estimate the average acccuracy for each component
        # use for initializing user estimate
        with open('data/diagnostic_questions.pkl', 'rb') as f:
            cards = pickle.load(f)
        estimate_file = 'data/diagnostic_avg_estimate.txt'
        if os.path.exists(estimate_file):
            with open(estimate_file) as f:
                return np.array([float(x) for x in f.readlines()])
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
        return np.array(estimates)

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

    def get_card(self, card: dict) -> Card:
        '''get card from db, insert if new'''
        # retrieve from db if exists
        c = self.db.get_card(card['question_id'])
        if c is not None:
            return c
        # create new card and insert to db
        qrep = self.embed_one(card)
        skill = np.zeros_like(qrep)
        skill[np.argmax(qrep)] = 1
        skill *= self.predict_one(card)
        new_card = Card(
            card_id=card['question_id'],
            text=card['text'],
            answer=card['answer'],
            qrep=qrep,
            skill=skill,
            category=card['category'],
            date=datetime.now()
        )
        self.db.add_card(new_card)
        return new_card

    def get_cards(self, cards: List[dict]) -> List[Card]:
        '''get cards from db, insert if new
        optimize speed by batch embedding
        '''
        to_be_embedded = []
        for i, card in enumerate(cards):
            card_id = card['question_id']
            cc = self.db.get_card(card_id)
            if cc is None:
                cards[i]['index'] = i
                to_be_embedded.append(card)
            else:
                cards[i] = cc
        if len(to_be_embedded) == 0:
            return cards

        texts = [x['text'] for x in to_be_embedded]
        qreps = self.lda.transform(self.tf_vectorizer.transform(texts))
        to_be_embedded = self.predict(to_be_embedded)

        for i, c in enumerate(to_be_embedded):
            skill = np.zeros_like(qreps[i])
            skill[np.argmax(qreps[i])] = 1
            skill *= c['prob']
            new_card = Card(
                card_id=c['question_id'],
                text=c['text'],
                answer=c['answer'],
                qrep=qreps[i],
                skill=skill,
                category=c['category'],
                date=datetime.now()
            )
            self.db.add_card(new_card)
            assert cards[c['index']]['question_id'] == new_card.card_id
            cards[c['index']] = new_card
        return cards

    def get_user(self, user_id: str) -> User:
        '''get user from DB, insert if new'''
        # retrieve from db if exists
        u = self.db.get_user(user_id)
        if u is not None:
            return u
        # create new user and insert to db
        k = self.params.n_components
        new_user = User(
            user_id=user_id,
            qrep=np.array([1 / k for _ in range(k)]),
            skill=self.estimate_avg(),
            category='HISTORY',
        )
        self.db.add_user(new_user)
        return new_user

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
        # 1. find same or similar card in records
        cards, scores = self.retrieve(card)
        if len(cards) > 0:
            return np.dot([x['prob'] for x in cards], scores)
        # TODO 2. use model to predict
        return 0

    def predict(self, cards: List[dict]) -> List[dict]:
        for i, card in enumerate(cards):
            cards[i]['prob'] = self.predict_one(card)
        return cards

    def embed_one(self, card: dict) -> np.ndarray:
        return self.lda.transform(self.tf_vectorizer.transform([card['text']]))[0]

    def set_params(self, params: dict):
        self.params.__dict__.update(params)

    def dist_category(self, user: User, card: Card) -> float:
        if user.category is None:
            return 0
        return float(card.category.lower() != user.category.lower())

    def dist_skill(self, user: User, card: Card) -> float:
        topic_idx = np.argmax(card.qrep)
        d = card.skill[topic_idx] - user.skill[topic_idx]
        # penalize easier questions
        d *= 1 if d < 0 else 10
        return abs(d)

    def dist_qrep(self, user: User, card: Card) -> float:
        def cosine_distance(a, b):
            return 1 - (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
        return cosine_distance(user.qrep, card.qrep)

    def dist_leitner(self, user: User, card: Card) -> float:
        t = user.leitner_scheduled_time.get(card.card_id, None)
        return 0 if t is None else (t - datetime.now()).days

    def dist_sm2(self, user: User, card: Card) -> float:
        t = user.sm2_scheduled_time.get(card.card_id, None)
        return 0 if t is None else (t - datetime.now()).days

    def score(self, user: User, cards: List[Card]) -> List[float]:
        return [{
            'qrep': self.dist_qrep(user, card),
            'skill': self.dist_skill(user, card),
            'category': self.dist_category(user, card),
            'leitner': self.dist_leitner(user, card),
            'sm2': self.dist_sm2(user, card),
        } for card in cards]

    def schedule(self, cards: List[dict]) -> Tuple[List[int], List[int], str]:
        if len(cards) == 0:
            return [], [], ''

        # a schedule request might contain that of many users
        # first group cards by user
        # card index for each user
        user_card_index_mapping = defaultdict(list)
        for i, c in enumerate(cards):
            user_card_index_mapping[c['user_id']].append(i)
        all_cards = self.get_cards(cards)
        users = [self.get_user(x) for x in user_card_index_mapping.keys()]
        # TODO assuming single user here
        assert len(users) == 1

        # for user in users:
        user = users[0]
        cards = [all_cards[i] for i in user_card_index_mapping[user.user_id]]

        scores = self.score(user, cards)
        scores_summed = [
            sum([self.params.__dict__[key] * value for key, value in ss.items()])
            for ss in scores
        ]
        order = np.argsort(scores_summed).tolist()
        ranking = [order.index(i) for i, _ in enumerate(cards)]

        # create rationale
        index_selected = order[0]
        card_selected = cards[index_selected]
        topic_idx = np.argmax(card_selected.qrep)
        info = scores[index_selected]
        info.update({
            'sum': scores_summed[index_selected],
            'user': user.skill[topic_idx],
            'card': card_selected.skill[topic_idx],
            'topic': topic_idx,
        })
        rationale = '\n'.join([
            '{}: {:.3f}'.format(key, value)
            for key, value in info.items()
        ])

        # add temporary history, to be completed by a call to `update`
        temp_history_id = json.dumps({'user_id': user.user_id,
                                      'card_id': card_selected.card_id})
        history = History(
            history_id=temp_history_id,
            user_id=user.user_id,
            card_id=card_selected.card_id,
            response='PLACEHOLDER',
            judgement='PLACEHOLDER',
            user_snapshot=user.to_snapshot(),
            scheduler_snapshot=json.dumps(self.params.__dict__),
            card_ids=json.dumps([x.card_id for x in cards]),
            scheduler_output=json.dumps({
                'order': order,
                'ranking': ranking,
                'rationale': rationale}),
            date=datetime.now())
        self.db.add_history(history)
        return order, ranking, rationale

    def update(self, cards: List[dict]):
        # a schedule request might contain that of many users
        # first group cards by user
        # card index for each user
        user_card_index_mapping = defaultdict(list)
        responses, history_ids = [], []
        for i, c in enumerate(cards):
            user_card_index_mapping[c['user_id']].append(i)
            responses.append(c['label'])
            history_ids.append(c['history_id'])
        cards = self.get_cards(cards)
        users = [self.get_user(x) for x in user_card_index_mapping.keys()]
        # TODO assuming single user here
        assert len(users) == 1

        # for user in users:
        user = users[0]
        for i in user_card_index_mapping[user.user_id]:
            card, response, history_id = cards[i], responses[i], history_ids[i]

            # update qrep
            # TODO better, discounted update of topical representation
            b = self.params.step_qrep
            user.qrep = b * card.qrep + (1 - b) * user.qrep

            # update skill
            # TODO better, discounted moving average for skill estimate
            if response == 'correct':
                a = self.params.step_correct
                user.skill = a * card.skill + (1 - a) * user.skill
            else:
                user.skill[np.argmax(card.qrep)] += self.params['step_wrong']
            user.skill = np.clip(user.skill, a_min=0.0, a_max=1.0)

            user.category = card.category
            user.last_study_time[card.card_id] = datetime.now()

            self.leitner_update(user, card, response)
            self.sm2_update(user, card, response)

            temp_history_id = json.dumps({'user_id': user.user_id,
                                          'card_id': card.card_id})
            history = self.db.get_history(temp_history_id)
            history.__dict__.update({
                'history_id': history_id,
                'response': response,
                'judgement': response,
            })
            self.db.update_history(temp_history_id, history)
        self.db.update_user(user)

    def leitner_update(self, user: User, card: Card, response: str):
        days = [None, 0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 9999]
        increment_days = {i: x for i, x in enumerate(days)}

        # boxes: 1 ~ 10
        cur_box = user.leitner_box.get(card.card_id, None)
        if cur_box is None:
            cur_box = 1
        new_box = cur_box + (1 if response == 'correct' else -1)
        new_box = max(min(new_box, 10), 1)
        user.leitner_box[card.card_id] = new_box
        interval = timedelta(days=increment_days[new_box])
        # TODO should use review time instead of now() here
        # using review time leads to spacing between cards in the same box
        date_studied = datetime.now()
        user.leitner_scheduled_time[card.card_id] = date_studied + interval

    def get_quality_from_response(self, response: str) -> int:
        return 4 if response == 'correct' else 1

    def sm2_update(self, user: User, card: Card, response: str):
        # TODO should use review time instead of now()
        date_studied = datetime.now()

        e_f = user.sm2_efactor.get(card.card_id, 2.5)
        inv = user.sm2_interval.get(card.card_id, 1)
        rep = user.sm2_repetition.get(card.card_id, 0) + 1

        q = self.get_quality_from_response(response)
        e_f = max(1.3, e_f + 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02))

        if response != 'correct':
            inv = 0
            rep = 0
        else:
            if rep == 1:
                inv = 1
            elif rep == 2:
                inv = 6
            else:
                inv = inv * e_f

        user.sm2_repetition[card.card_id] = rep
        user.sm2_efactor[card.card_id] = e_f
        user.sm2_interval[card.card_id] = inv
        user.sm2_scheduled_time[card.card_id] = date_studied + timedelta(days=inv)
