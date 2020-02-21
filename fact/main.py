import click
import os
import _jsonnet

from fact.files import files_cli
from fact.datasets.protobowl import proto_cli


@click.group()
def cli():
    pass


cli.add_command(files_cli)
cli.add_command(proto_cli)


@click.command()
def configure():
    """
    Generate allennlp configuration files
    """
    with open('configs/karl_model.jsonnet') as f:
        karl_conf = f.read()
    
    all_params = {
        'rnn': {'use_rnn': 'true', 'use_bert': 'false'},
        'bert': {'use_rnn': 'false', 'use_bert': 'true'},
    }
    for name, params in all_params.items():
        model_conf = _jsonnet.evaluate_snippet(karl_conf, tla_codes=params)
        with open(os.path.join('configs/generated', name + '.json')) as f:
            f.write(model_conf)


if __name__ == '__main__':
    cli()
