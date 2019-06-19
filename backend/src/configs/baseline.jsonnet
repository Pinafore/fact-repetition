{
    # This needs to be changed to match the actual reader name/etc
    # Example of a reader defn https://github.com/allenai/allennlp/blob/master/training_config/bidaf.jsonnet#L4
    dataset_reader: {},
    # This gets passed into Reader.read() method
    train_data_path: 'data/expanded.qanta.train.2018.04.18.json',
    valdation_data_path: 'data/expanded.qanta.dev.2018.04.18.json',
    model: {
        # This matches the name registered in fact/models/baseline.py
        type: 'baseline'
    },
}