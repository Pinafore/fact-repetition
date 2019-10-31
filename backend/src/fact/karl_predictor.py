import json
import click
from allennlp.predictors.predictor import Predictor
from allennlp.data import DatasetReader, Instance
from allennlp.common.util import JsonDict
from fact.models.bert_baseline import KarlModel
from fact.datasets.qanta import QantaReader


KARL_PREDICTOR = 'karl_predictor'


@Predictor.register(KARL_PREDICTOR)
class KarlPredictor(Predictor):
    def __init__(self, model: KarlModel, dataset_reader: QantaReader):
        super().__init__(model, dataset_reader)
        self._model.output_embedding = True
    
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(
            json_dict['text'],
            json_dict['user_id'],
            json_dict['question_id'],
            label=json_dict.get('label')
        )


@click.command()
@click.argument('archive_path')
@click.argument('out_path')
def main(archive_path, out_path):
    predictor = KarlPredictor.from_path(archive_path, predictor_name=KARL_PREDICTOR)
    questions = [
        'Name this capital of idaho',
        'This physicist was known for being a talented violinist',
    ]
    preds = []
    for q in questions:
        # Since user_id/question_id don't exist, these are set to null values
        # and counts are set to zero as well
        # We should probably change the api a little to be able to manually pass counts
        # Output has keys: q_rep, logits, probs
        preds.append(predictor.predict_json({'text': q, 'user_id': '', 'question_id': ''}))
    
    with open(out_path, 'w') as f:
        json.dump({'predictions': preds, 'questions': questions}, f)


if __name__ == '__main__':
    main()
