local embedding_dim = 100;
local char_out_dim = 100;
local entity_dim = 100;
local max_text_vocab = 150000;
local dropout = 0.25;

local boe = {
  type: 'boe',
  embedding_dim: embedding_dim,
  averaged: true
};
{
    # This needs to be changed to match the actual reader name/etc
    # Example of a reader defn https://github.com/allenai/allennlp/blob/master/training_config/bidaf.jsonnet#L4
    dataset_reader: {
        'type': 'qanta'
    },
    # This gets passed into Reader.read() method
    train_data_path: 'data/train.qanta.record.json',
    validation_data_path: 'data/dev.qanta.record.json',
    model: {
        # This matches the name registered in fact/models/baseline.py
        type: 'baseline',
        dropout: dropout,
        text_field_embedder: {
            token_embedders: {
                tokens: {
                type: 'embedding',
                embedding_dim: embedding_dim,
                num_embeddings: max_text_vocab,
                },
            token_characters: {
                type: 'character_encoding',
                embedding: {
                    embedding_dim: 16,
                },
            dropout: dropout,
            encoder: {
                type: 'cnn',
                embedding_dim: 16,
                num_filters: 100,
                ngram_filter_sizes: [5],
          },
        },
      },
    },
    seq2vec_encoder: boe,
    num_labels: 2,
    },
    iterator: {
        type: 'basic'
    },
    trainer: {
        optimizer: {'type': 'adam'},
        num_serialized_models_to_keep: 2,
        num_epochs: 50,
        patience: 3,
        cuda_device: 0,
        validation_metric: '+accuracy',
    },
}