local lazy = true;
local debug = true;
local bert_finetune = false;
local batch_size = 20;
local num_epochs = 4;
local model_name = '/fs/clip-quiz/entilzha/qanta_lm';
{
    "pytorch_seed": 0,
    "numpy_seed": 0,
    "random_seed": 0,
    "train_data_path": 'train',
    "validation_data_path": 'test',
    "dataset_reader": {
        "type": 'retention',
        "lazy": lazy,
        "debug": debug,
        "tokenizer": {
            "type": 'pretrained_transformer',
            "model_name": model_name,
        },
        "token_indexers": {
            "text": {
                "type": 'pretrained_transformer',
                "model_name": model_name,
            },
        },
    },
    "model": {
        "type": 'bert_retention_model',
        "model_name_or_path": model_name,
        "dropout": 0.25,
        "extra_hidden_dim": 12,
        "bert_pooling": 'cls',
        "bert_finetune": bert_finetune,
        "return_embedding": true,
        "uid_embedder": {
            "token_embedders": {
                "uid_tokens": {
                    "type": 'embedding',
                    "embedding_dim": 50,
                    "vocab_namespace": 'uid_tokens'
                },
            }
        },
        "qid_embedder": {
            "token_embedders": {
                "qid_tokens": {
                    "type": 'embedding',
                    "embedding_dim": 50,
                    "vocab_namespace": 'qid_tokens'
                },
            }
        },
    },
    "data_loader": {
        "batch_size": 32
    },
    "trainer": {
        "num_epochs": num_epochs,
        "patience": 2,
        "cuda_device": 0,
        "optimizer": {
          "type": "adam",
          "lr": 0.001,
        },
    }
}
