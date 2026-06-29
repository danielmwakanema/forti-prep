import typing

import click

from forti_prep.blob import BlobWriter
from forti_prep.config import Configuration


@click.command()
@click.option(
    "--config",
    type=click.File(),
    default="config.json",
    help="Read config from the given file",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, writable=True),
    default="data",
    help="Write output data to this folder",
)
@click.option("--version", type=int, default=0, help="Upload with this version number")
@click.argument("nc-file", nargs=1)  # File name or opendap url
def cli(config: typing.TextIO, output_dir: str, version: int, nc_file: str):
    cfg: Configuration = Configuration.model_validate_json(config.read())
    writer = BlobWriter(cfg, output_dir)
    writer.write_data(nc_file, version)
