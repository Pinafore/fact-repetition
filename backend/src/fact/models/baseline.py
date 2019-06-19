import torch
from allennlp.models import Model
from allennlp.data import Vocabulary


@Model.register('baseline')
class Baseline(Model):
    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)
    
    # Change these to match the text_to_instance argument names
    def forward(self, *args, **kwargs):
        # This is where all the modeling stuff goes
        # AllenNLP requires that if there are training labels,
        # that a dictionary with key "loss" be returned.
        # You can stuff anything else in the dictionary that might be helpful
        # later.
    
        return {'loss': torch.ones(1)}
