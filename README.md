# fact-repetition

## Running Allennlp reader

To not depend on the model, this should run without error:
```python
from fact.datasets.qanta import QantaReader, QANTA_TRAIN

instances = QantaReader().read(QANTA_TRAIN)
```

This command will test that the model and data load correctly, then output some statistics
about the dataset.
```bash
$ allennlp dry-run --include-package fact configs/baseline.jsonnet
```

## Training model
```bash
$ allennlp train --include-package fact -s trained-models/baseline configs/baseline.jsonnet
```


## Data
### Question-Oriented Data

The following datasets contain QANTA data extended with protobowl statistics.  

expanded.qanta.dev.2018.04.18.json   
expanded.qanta.train.2018.04.18.json
expanded.qanta.test.2018.04.18.json

Each is the original qanta folds, shrunk to only items that map to protobowl data (train is roughly 78k, dev is roughly 600, test is roughly 700 relative to original ~128k, ~2k, ~4k numbers in original qanta dataset), and with an addition of question_stats that contains:

1) list of all responses to the question (true if correct else false).  (Rulings that were "prompt" were dropped from the data. This only dropped 399 out of 120k unique questions from our dataset.)  
2) overall accuracy calculated from this
3) the point at which the question was answered, converted into a ratio of time elapsed/total time.  (index can be calculated as floor(len(text) * length_per_question);  I can put this explicitly if requested.)  
4) all users that answered this question (with user_ids).

An example question looks like this: 

[{'answer': 'annexation of the Republic of Texas by the United States [or Annexation Treaty of 1845; accept any answer indicating that Texas is joining the United States of America as a state; do not accept any answer indicating the independence of Texas from Mexico]',
  'category': 'History',
  'dataset': 'protobowl',
  'difficulty': 'College',
  'first_sentence': 'Robert Walker argued that failing to take this action would lead to an overflow of Northern insane asylums and British intervention.',
  'fold': 'guessdev',
  'gameplay': True,
  'page': 'Texas_annexation',
  'proto_id': '55414836ea23cc9417e9ba80',
  'qanta_id': 93135,
  'qdb_id': None,
  'question_stats': {'accuracy_bools_per_question': [True,
    True,
    False,
    False,
    True],
   'accuracy_per_question': 0.6,
   'length_per_question': [0.4568, 0.473, 0.4582, 0.3712, 0.4571],
   'users_per_question': ['25b32a7c33fa31f2fffe3777fbd7652cacb75ff5',
    'af31d3ecbf81775f8c7fd95fea185d6a6cbcec72',
    'd97d7fa8989ec141b9d2b640f3f16f7d4bda9bd0',
    'e91d3e1ac0331ae90a55f21a5b6199fd70ad37e0',
    '79b7128c56e41d4a889c6dd4c8d54df955bb2796']},
  'subcategory': 'American',
  'text': "Robert Walker argued that failing to take this action would lead to an overflow of Northern insane asylums and British intervention. Juan Almonte resigned his diplomatic post in indignation over this event. Isaac Van Zandt began discussing this plan with Abel Upshur before Upshur died. Anson Jones proposed this plan, which reduced his own power but reserved the weaker side the right to split into five parts in the future. Five years after it, a payment of 10 million dollars helped the area at issue repay debts. The Regulator-Moderator war was calmed just before this deal was struck. The question of whether the Nueces River became a southern border in this transaction led to war later in James K. Polk's presidency. For 10 points, name this deal that ended the independent Lone Star Republic.",
  'tokenizations': [[0, 132],
   [133, 206],
   [207, 286],
   [287, 425],
   [426, 516],
   [517, 589],
   [590, 723],
   [724, 800]],
  'tournament': 'ACF Regionals',
  'year': 2015},


### User-Oriented Data



## Baselines

