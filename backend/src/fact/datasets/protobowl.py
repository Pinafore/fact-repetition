import click


@click.group(name='proto')
def proto_cli():
    pass


@proto_cli.command()
def process():
    """
    TODO: Put the protobowl condenser code here
    """