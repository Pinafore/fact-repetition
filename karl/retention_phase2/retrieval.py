# load HF dataset of all cards (create if it doesn't exist)
# load FAISS index of card text (create if it doesn't exist)

import os
import logging
import torch
import pandas as pd
from shutil import rmtree
from datasets import Dataset, load_from_disk
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

from karl.db.session import SessionLocal
from karl.config import settings
from karl.models import Card, StudyRecord


# create logger with 'scheduler'
logger = logging.getLogger('karl_retriever')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f'{settings.CODE_DIR}/retriever.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


class KARLRetriever:

    def _get_card_dataset(self):
        if not os.path.exists(settings.card_dataset_path):
            # query the database to get all the cards
            logger.info('Creating card dataset')
            session = SessionLocal()
            dataset = []
            for i, card in enumerate(session.query(Card)):
                if self.head is not None and i >= self.head:
                    break
                card = card.__dict__
                card.pop('_sa_instance_state')
                dataset.append(card)
            dataset = Dataset.from_pandas(pd.DataFrame(dataset))
            dataset.save_to_disk(settings.card_dataset_path)
            session.commit()
            session.close()
        else:
            dataset = load_from_disk(settings.card_dataset_path)
        return dataset

    def _get_embedding_dataset(self, gpu=False):
        if not os.path.exists(settings.embedding_dataset_path):
            logger.info('Creating embedding dataset')
            dataset = self._get_card_dataset()

            torch.set_grad_enabled(False)
            encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            dataset = dataset.map(lambda x: {'embeddings': encoder(**tokenizer(x["text"], return_tensors="pt", truncation=True))[0][0].numpy()})
            dataset.save_to_disk(settings.embedding_dataset_path)
        else:
            dataset = load_from_disk(settings.embedding_dataset_path)
        return dataset

    def _get_indexed_dataset(self, gpu=False):
        dataset = self._get_embedding_dataset()
        if not os.path.exists(settings.index_path):
            logger.info('Creating FAISS index')
            dataset.add_faiss_index(column='embeddings')
            dataset.save_faiss_index('embeddings', settings.index_path)
        else:
            dataset.load_faiss_index('embeddings', settings.index_path)
        return dataset

    def __init__(self, head: int = None):
        self.head = head
        self.dataset = self._get_indexed_dataset()
        self.encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    def query(self, text: str, k: int = 5, m: int = 20):
        embedding = self.encoder(**self.tokenizer(text, return_tensors="pt", truncation=True))[0][0].detach().numpy()
        scores, results = self.dataset.get_nearest_examples('embeddings', embedding, k=k)
        n_examples = len(results['text'])
        examples = [{} for _ in range(n_examples)]
        for key, values in results.items():
            for i, value in enumerate(values):
                examples[i][key] = value
        return examples

    def query_for_user(self, text: str, user_id: str, k: int = 5, m: int = 20):
        retrieved_examples = self.query(text, k, m)
        studied_examples = []
        session = SessionLocal()
        for example in retrieved_examples:
            record = session.query(StudyRecord).\
                filter(StudyRecord.user_id == user_id).\
                filter(StudyRecord.card_id == example['id']).\
                order_by(StudyRecord.date.desc()).\
                first()
            if record is not None:
                studied_examples.append(example)
        return studied_examples


if __name__ == '__main__':
    retriever = KARLRetriever(head=100)

    # TODO actually test the retrieval
    cards = retriever._get_card_dataset()
    text = cards[0]
    
    

    rmtree(settings.card_dataset_path)
    rmtree(settings.embedding_dataset_path)
    os.remove(settings.index_path)
