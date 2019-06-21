{
    # This needs to be changed to match the actual reader name/etc
    # Example of a reader defn https://github.com/allenai/allennlp/blob/master/training_config/bidaf.jsonnet#L4
    dataset_reader: {
        'type': 'qanta'
    },
    # This gets passed into Reader.read() method
    train_data_path: 'data/expanded.qanta.train.2018.04.18.json',
    valdation_data_path: 'data/expanded.qanta.dev.2018.04.18.json',
    model: {
        # This matches the name registered in fact/models/baseline.py
        type: 'baseline'
    },
    iterator: {
        type: 'bucket',
        batch_size: 128
    },
    trainer: {
        optimizer: {'type': 'adam'},
        num_serialized_models_to_keep: 2,
        num_epochs: 50,
        patience: 3,
        cuda_device: 0
    },
}