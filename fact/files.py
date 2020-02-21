# TODO: Replace the files/commands below with relevant ones
# TODO: Modify script to download missing files, execute cmd, and check checksum
from typing import NamedTuple, Optional
import os
import subprocess
from urllib.request import urlretrieve
import click
from fact.util import get_logger


log = get_logger(__name__)


class File(NamedTuple):
    src: Optional[str]
    dst: str
    md5: Optional[str]
    cmd: str

FILES = [
    File(
        'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip',
        'data/bert/uncased_L-12_H-768_A-12.zip',
        None,
        'cd data/bert && unzip uncased_L-12_H-768_A-12.zip'
    ),
    File(
        'https://msmarco.blob.core.windows.net/msmarco/eval_v2.1_public.json.gz',
        'data/msmarco/eval_v2.1_public.json.gz',
        None,
        'cd data/msmarco && gzip -d eval_v2.1_public.json.gz'
    ),
    File(
        'https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz',
        'data/msmarco/dev_v2.1.json.gz',
        None,
        'cd data/msmarco && gzip -d dev_v2.1.json.gz'
    ),
    File(
        'https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz',
        'data/msmarco/train_v2.1.json.gz',
        None,
        'cd data/msmarco && gzip -d train_v2.1.json.gz'
    )
]


def md5sum(filename):
    return subprocess.run(
        f'md5sum {filename}',
        shell=True,
        stdout=subprocess.PIPE,
        check=True
    ).stdout.decode('utf-8').split()[0]


def verify_checksum(checksum, filename):
    if os.path.exists(filename):
        file_checksum = md5sum(filename)
        if checksum == file_checksum:
            return True
        else:
            return False
    else:
        return False


@click.group(name='files')
def files_cli():
    pass


@files_cli.command()
def download():
    pass


@files_cli.command()
def verify():
    log.info('Checking MD5 Checksums')
    for f in FILES:
        if f.md5 is None:
            log.info(f'Skipping: {f.dst}')
        elif verify_checksum(f.md5, f.dst):
            log.info(f'Verified: {f.dst}')
        else:
            raise ValueError(f'Verification failed for: {f.dst}')