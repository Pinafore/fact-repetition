import torch
import json
import time
from pytorch_pretrained_bert import BertTokenizer, BertModel
start_time = time.time()

QUESTION_FILE = './question.json'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(QUESTION_FILE) as f:
    question_data = json.load(f)
    text_list = [record['text'] for record in question_data]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_embedding_list = []
i = 0
for text in text_list:
    i += 1
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model = BertModel.from_pretrained('bert-base-uncased')
    # model.to(device)
    model.eval()
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    token_embeddings = []
    batch_i = 0
    for token_i in range(len(tokenized_text)):
        hidden_layers = []
        for layer_i in range(len(encoded_layers)):
            vec = encoded_layers[layer_i][batch_i][token_i]
            hidden_layers.append(vec)
        token_embeddings.append(hidden_layers)
    sentence_embedding = torch.mean(encoded_layers[len(encoded_layers) - 1], 1)
    print("vector: ", sentence_embedding[0].shape[0])
    # print("vector: ", sentence_embedding[0])
    text_embedding_list.append(sentence_embedding[0].tolist())
    if i >= 50:
        break
print("Total time: --- %s seconds ---" % (time.time() - start_time))