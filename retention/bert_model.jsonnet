function(lr=0.001,
         dropout=0.25,
         extra_hidden_dim=12,
         bert_pooling='cls',
         debug=true,
         bert_finetune=false,
         pytorch_seed=0,
         numpy_seed=0,
         random_seed=0,
         model_name='bert-base-uncased',
         model_name_or_path='bert-base-uncased') {
  pytorch_seed: pytorch_seed,
  numpy_seed: numpy_seed,
  random_seed: random_seed,
  dataset_reader: {
    lazy: false,
    debug: debug,
    type: 'retention',
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: model_name,
      do_lowercase: true,
      start_tokens: [],
      end_tokens: [],
    },
    token_indexers: {
      text: {
        type: 'bert-pretrained',
        pretrained_model: model_name_or_path,
      },
    },
  },
  train_data_path: 'train',
  validation_data_path: 'test',
  model: {
    type: 'bert_retention_model',
    dropout: dropout,
    extra_hidden_dim: extra_hidden_dim,
    bert_pooling: bert_pooling,
    bert_finetune: bert_finetune,
    model_name_or_path: model_name_or_path,
    uid_embedder: {
      token_embedders: {
        uid_tokens: {
        type: 'embedding',
        embedding_dim: 50,
        vocab_namespace: 'uid_tokens'
        },
      }
    },
    qid_embedder: {
      token_embedders: {
        qid_tokens: {
          type: 'embedding',
          embedding_dim: 50,
          vocab_namespace: 'qid_tokens'
        },
      }
    },
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['tokens', 'num_tokens']],
    batch_size: 32,
  },
  trainer: {
    type: 'callback',
    callbacks: [
      {
        type: 'checkpoint',
        checkpointer: { num_serialized_models_to_keep: 1 },
      },
      { type: 'track_metrics', patience: 2, validation_metric: '+accuracy' },
      'validate',
      { type: 'log_to_tensorboard' },
    ],
    optimizer: {
      type: 'adam',
      lr: lr,
    },
    num_epochs: 1,
    cuda_device: 0,
  },
}