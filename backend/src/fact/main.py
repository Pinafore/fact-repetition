import click

from fact.files import files_cli
from fact.datasets.protobowl import proto_cli


@click.group()
def cli():
    pass


cli.add_command(files_cli)
cli.add_command(proto_cli)


if __name__ == '__main__':
    cli()
