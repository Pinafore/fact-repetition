import numpy as np
import torch
import torch.nn as nn
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from allennlp.data import Vocabulary
# from allennlp.modules.token_embedders import Embedding
# from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# from allennlp.nn.util import get_text_field_mask

from fact.datasets.qanta import * 

train_inst = QantaReader().read(QANTA_TRAIN)  
test_inst = QantaReader().read(QANTA_TEST) 

# def identity_tokenizer(text):
#     return text

# vec = CountVectorizer(tokenizer=identity_tokenizer)
# vocab = Vocabulary.from_instances(train_inst + test_inst)
# print("vocab", vocab)
# token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
#                             embedding_dim=EMBEDDING_DIM)
# embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

# find the list of tokenized text

def load_words(text_list):
    """
    vocabulary building
    Keyword arguments:
    exs: list of input questions-type pairs
    """
    kUNK = '<unk>'
    kPAD = '<pad>'
    words = set()
    word2ind = {kPAD: 0, kUNK: 1}
    ind2word = {0: kPAD, 1: kUNK}
    for tokens in text_list:
        for token in tokens:
            words.add(token)
    words = sorted(words)
    for token in words:
        idx = len(word2ind)
        word2ind[token] = idx
        ind2word[idx] = token
    words = [kPAD, kUNK] + words
    return words, word2ind, ind2word
    
def vectorize(tokens, word2ind):
    """
    vectorize a single example based on the word2ind dict. 
    Keyword arguments:
    exs: list of input questions-type pairs
    ex: tokenized question sentence (list)
    label: type of question sentence
    Output:  vectorized sentence(python list) and label(int)
    e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
    """
    vec_text = [0] * len(tokens)
        # print("==========================", len(ex))

        #### modify the code to vectorize the question text
        #### You should consider the out of vocab(OOV) cases
        #### question_text is already tokenized    
        #### Your code here

    unk = '<unk>'
    for ii in range(len(tokens)):
        if tokens[ii] not in word2ind:
            vec_text[ii] = word2ind.get(unk)
        else:
            vec_text[ii] = word2ind.get(tokens[ii])

        # print("vec_text", vec_text)
    return vec_text



# get the list of tokenized text
train_text_list = []
for inst in tqdm(train_inst):
    token_list = inst.fields['text'].tokens
    tokens = [str(token.text) for token in token_list]
    train_text_list.append(tokens)

test_text_list = []
for inst in tqdm(test_inst):
    token_list = inst.fields['text'].tokens
    tokens = [str(token.text) for token in token_list]
    test_text_list.append(tokens)

#build the vocabulary
vocab, word2ind, ind2word = load_words(train_text_list + test_text_list)

EMBEDDING_DIM = 100
embedder = nn.Embedding(len(vocab), EMBEDDING_DIM, padding_idx=0)


train_x = []
for tokens in tqdm(train_text_list):
    vec_text = vectorize(tokens, word2ind)
    # print("vec_text", vec_text)
    vec_text_tensor = torch.tensor(vec_text)
    text_embed = embedder(vec_text_tensor)
    # print("text_embed", text_embed)
    # print(text_embed.size())
    text_encoded = text_embed.sum(0) / text_embed.shape[0] 
    # print(text_encoded.shape) 
    # print(text_encoded.size())
    # print(text_encoded)
    # text_encoded /= len()
    # embedded_text = embedder(text)
    # mask = get_text_field_mask(text).float()
    # text_vectors = self._seq2vec_encoder(embedded_text, mask=mask)
    question_features = inst.fields['question_features'].array
    user_features = inst.fields['user_features'].array
    feature_vector = np.concatenate((text_encoded.detach().numpy(), question_features, user_features), axis=None)
    train_x.append(feature_vector)
train_x = np.array(train_x)

# print("train_x", train_x[0:5])
train_y = np.array([inst.fields['label'].label for inst in train_inst])
# print("train_y", train_y[0:5])

test_x = []
for tokens in tqdm(test_text_list):
    vec_text = vectorize(tokens, word2ind)
    # print("vec_text", vec_text)
    vec_text_tensor = torch.tensor(vec_text)
    text_embed = embedder(vec_text_tensor)
    # print("text_embed", text_embed)
    # print(text_embed.size())
    text_encoded = text_embed.sum(0) / text_embed.shape[0] 
    # print(text_encoded.shape) 
    # print(text_encoded.size())
    # print(text_encoded)
    # text_encoded /= len()
    # embedded_text = embedder(text)
    # mask = get_text_field_mask(text).float()
    # text_vectors = self._seq2vec_encoder(embedded_text, mask=mask)
    question_features = inst.fields['question_features'].array
    user_features = inst.fields['user_features'].array
    feature_vector = np.concatenate((text_encoded.detach().numpy(), question_features, user_features), axis=None)
    test_x.append(feature_vector)
test_x = np.array(test_x)

test_y = np.array([inst.fields['label'].label for inst in test_inst])
# X = np.array([[0.5, 0.6], [0.7, 0.5], [0.5, 0.5], [0.2, 0.2]])
# y = np.array([0, 0, 1, 1])

clf1 = LogisticRegression(solver='lbfgs', max_iter=100, random_state=123)
print('train_x', type(train_x), 'size', train_x.shape)
print('train_y', type(train_y), 'size', train_y.shape)
print('train_x[0]', type(train_x[0]), 'size', train_x[0].shape)
clf1.fit(train_x, train_y)
probas = clf1.predict_proba(test_x)
y_pred = [0 if pr[0] > pr [1] else 1 for pr in probas]
print("probas", probas)  
accuracy = accuracy_score(test_y, y_pred)
print("Accuracy (train): %0.1f%% " % (accuracy * 100))