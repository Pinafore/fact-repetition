import torch
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        
config = Config(
    testing=True,
    seed=1,
    batch_size=64,
    lr=3e-4,
    epochs=2,
    hidden_sz=64,
    max_seq_len=100, # necessary to limit memory usage
    max_vocab_size=100000,
)


@Model.register('Basic')
class Basic(Model):
    def __init__(self, 
                word_embeddings: TextFieldEmbedder,
                encoder: None,
                vocab: Vocabulary):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features = None, 
                                            out_features = None)
        self.accuracy = CategoricalAccuracy()
        self.loss = nn.BCEWithLogitsLoss()
    
    # Change these to match the text_to_instance argument names
    def forward(self, tokens: Dict[str, torch.Tensor],
                label: torch.Tensor) -> torch.Tensor:
        # This is where all the modeling stuff goes
        # AllenNLP requires that if there are training labels,
        # that a dictionary with key "loss" be returned.
        # You can stuff anything else in the dictionary that might be helpful
        # later.
        mask = get_text_field_mask(sentence)
        question_embeddings = self.word_embeddings(sentence)
        # user_embeddings = accuracy + buzz_ratio + # of question_answered +lstm on correct&buzzing
        # user_embeddings = user_id_embeddings + #of question played + #of correct answers + # of wrong answers

        embeddings = question_embeddings
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        output["loss"] = self.loss(class_logits, label)
        return output



reader = None
train_dataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/training.txt'))
validation_dataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/validation.txt'))
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

#==========The Embedder==========
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

#==========Train==========
# optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                #   validation_dataset=validation_dataset,
                #   patience=10,
                  num_epochs=1000,
                  cuda_device=cuda_device)
trainer.train()