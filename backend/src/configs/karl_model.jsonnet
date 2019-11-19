local embedding_dim = 100;
local char_out_dim = 100;
local hidden_dim = 100;
local entity_dim = 100;
local max_text_vocab = 150000;
local dropout = 0.25;
local bert_model = 'bert-base-uncased';
local bert_dim = 768;

local we_embed = {
    token_embedders: {
        tokens: {
            type: 'embedding',
            embedding_dim: embedding_dim,
            pretrained_file: "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
            vocab_namespace: 'tokens'
        },
    }
};

local we_indexer = {
    tokens: {
        type: "single_id",
        lowercase_tokens: true,
        namespace: 'tokens'
    }
};

local contextualizer = {
    type: "lstm",
    bidirectional: true,
    input_size: embedding_dim,
    hidden_size: hidden_dim,
    num_layers: 1,
    dropout: 0.2
};

local bert_indexer = {
    bert: {
        type: "bert-pretrained",
        pretrained_model: bert_model
    }
};


function(use_bert=true, use_rnn=false, lazy=true) {
    # This needs to be changed to match the actual reader name/etc
    # Example of a reader defn https://github.com/allenai/allennlp/blob/master/training_config/bidaf.jsonnet#L4
    dataset_reader: {
        type: 'qanta',
        lazy: lazy,
        use_rnn: use_rnn,
        use_bert: use_bert,
        token_indexers: if use_bert then bert_indexer else we_indexer,
        tokenizer: if use_bert then {word_splitter: 'bert-basic'} else {word_splitter: 'spacy'}
    },
    # This gets passed into Reader.read() method
    train_data_path: 'data/train.record.json',
    validation_data_path: 'data/dev.record.json',
    model: {
        # This matches the name registered in fact/models/baseline.py
        type: 'karl_model',
        use_bert: use_bert,
        bert_train: true,
        use_rnn: use_rnn,
        dropout: dropout,
        text_field_embedder: if use_rnn then we_embed else null,
        contextualizer: if use_rnn then contextualizer else null,
        bert_model: bert_model,
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
        batch_size: if use_bert then 16 else 64,
    },
    trainer: {
        optimizer: {'type': 'adam', 'lr': 1e-3},
        num_serialized_models_to_keep: 1,
        num_epochs: 4,
        patience: 2,
        cuda_device: 0,
        validation_metric: '+accuracy',
        // learning_rate_scheduler: {
        //     type: 'reduce_on_plateau',
        //     factor: 0.5,
        //     mode: 'max',
        //     patience: 4,
        // },
    },
}