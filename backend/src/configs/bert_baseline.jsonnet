local embedding_dim = 100;
local char_out_dim = 100;
local entity_dim = 100;
local max_text_vocab = 150000;
local dropout = 0.25;

local boe = {
  type: 'boe',
  embedding_dim: embedding_dim,
  averaged: true,
};
{
    # This needs to be changed to match the actual reader name/etc
    # Example of a reader defn https://github.com/allenai/allennlp/blob/master/training_config/bidaf.jsonnet#L4
    dataset_reader: {
        'type': 'qanta',
    },
    # This gets passed into Reader.read() method
    train_data_path: 'data/train.record.json',
    validation_data_path: 'data/dev.record.json',
    model: {
        # This matches the name registered in fact/models/baseline.py
        type: 'bert_baseline',
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
        num_labels: 2,
    },
    iterator: {
        type: 'bucket',
        sorting_keys: [['tokens', 'num_tokens']],   
        batch_size: 8,
    },
    trainer: {
        optimizer: {'type': 'adam', 'lr': 1e-3},
        num_serialized_models_to_keep: 1,
        num_epochs: 2,
        patience: 2,
        cuda_device: 0,
        validation_metric: '+accuracy',
        learning_rate_scheduler: {
            type: 'reduce_on_plateau',
            factor: 0.5,
            mode: 'max',
            patience: 4,
        },
    },
}