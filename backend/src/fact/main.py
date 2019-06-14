import click

from fact.files import files_cli


@click.group()
def cli():
    pass


cli.add_command(files_cli)


if __name__ == '__main__':
    cli()
