import hashlib
import json
import os
import time
import typing

import click
import numpy as np
import xarray as xr

import nc2forti.convert as convert
from nc2forti.config import Configuration, ParameterConfiguration
from nc2forti.data import Data


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    type=click.File(),
    default="malawi.json",
    help="Read config from the given file",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False),
    help="Read input data from this file",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, writable=True),
    default="data",
    help="Write output data to this folder",
)
@click.option("--version", type=int, default=0, help="Upload with this version number")
# @click.argument('input_netcdf', nargs=1)
def run_once(config: typing.TextIO, input_file: str, output_dir: str, version: int):
    cfg: Configuration = Configuration.model_validate_json(config.read())
    writer = BlobWriter(cfg, output_dir)
    writer.write_data(input_file, cfg.area, version)


@cli.command()
@click.option(
    "--config",
    type=click.File(),
    default="malawi.json",
    help="Read config from the given file",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, writable=True),
    help="Read input data from this folder",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, writable=True),
    default="data",
    help="Write output data to this folder",
)
def run_forever(config: typing.TextIO, input_dir: str, output_dir: str):
    print("Loading configuration...")
    cfg: Configuration = Configuration.schema().loads(config.read())  # type: ignore
    print("Instantiating writer...")
    writer = BlobWriter(cfg, output_dir)

    while True:
        print("Scanning input directory...")
        files = [*os.scandir(input_dir)]

        if len(files):
            print(f"Found {len(files)}...")
            for file in files:
                try:
                    print(f"Attempting to transform file {file.path}...")
                    print("Writing file...")
                    writer.write_data(
                        file.path, cfg.area, int(os.stat(file.path).st_mtime)
                    )
                    print("Deleting file...")
                    os.remove(file.path)
                    print("File transformed successfully.")
                except Exception as error:
                    print(f"Failed to tranform file {file.path} because of {error}")
        else:
            print("Found no files to transform.")
        time.sleep(1)


class BlobWriter:
    def __init__(self, cfg: Configuration, workdir: str):
        self._cfg = cfg
        self._workdir = workdir

    def write_data(self, filename: str, model: str, version: int):
        ds = xr.open_dataset(filename)
        for dim in ds.dims:
            if not dim in ("time", "x", "y") and ds.dims[dim] == 1:
                ds = ds.isel(**{dim: 0}, drop=True)  # type: ignore

        data_dir = self._get_data_dir(ds, model, version)

        self._write_latitude(ds, data_dir)
        self._write_longitude(ds, data_dir)
        self._write_data(ds, self._cfg.parameters, data_dir)
        self._write_complete(model, version)
        self._set_latest(model, version)

    def _set_latest(self, model: str, version: int):
        os.makedirs(os.path.join(self._workdir, "latest"), exist_ok=True)
        filename = os.path.join(self._workdir, "latest", model)
        with open(filename, "w") as f:
            f.write(str(version))

    def _write_complete(self, model: str, version: int):
        filename = os.path.join(self._workdir, model, str(version), "complete.json")
        with open(filename, "w") as f:
            data = {
                "area": model,
                "version": version,
                "time_until_next": 3600000000000,
                "geographic_extent": None,
            }
            json.dump(data, f)

    def _get_data_dir(self, ds: xr.Dataset, model: str, version: int) -> str:
        m = hashlib.md5()
        m.update(ds["y"].values.tobytes())
        m.update(ds["x"].values.tobytes())
        data_dir = os.path.join(self._workdir, model, str(version), m.hexdigest())
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def _open(self, data_dir: str, filename: str, mode="wb") -> typing.IO:
        full_path = os.path.join(data_dir, filename)
        return open(full_path, mode)

    def _write_latitude(self, ds: xr.Dataset, data_dir: str):
        with self._open(data_dir, "latitude") as f:
            values = np.repeat(ds["y"].values, len(ds["x"].values)).astype(np.float32)
            values.tofile(f)

    def _write_longitude(self, ds: xr.Dataset, data_dir: str):
        with self._open(data_dir, "longitude") as f:
            for i in range(len(ds["y"].values)):
                ds["x"].values.astype(np.float32).tofile(f)

    def _write_data(
        self,
        ds: xr.Dataset,
        parameters: typing.Dict[str, ParameterConfiguration],
        data_dir: str,
    ):
        arrays: typing.List[Data] = []

        for param, cfg in parameters.items():
            data = convert.transform(param, ds, parameters[param])
            arrays.append(data)

        raw_data = [d.values for d in arrays]

        with self._open(data_dir, "data") as f:
            for y in range(len(ds["y"])):
                for x in range(len(ds["x"])):
                    for d in raw_data:
                        values = (d[:, y, x] * 10).astype("<i2")
                        values.tofile(f)

        self._write_meta(arrays, data_dir)

    def _write_meta(self, arrays: typing.List[Data], data_dir: str):
        point_count = 0
        parameter_meta = {}
        for a in arrays:
            times = ["0001-01-01T00:00:00Z"]
            if len(a.times) > 0:
                times = [
                    np.datetime_as_string(t, timezone="UTC", unit="s") for t in a.times
                ]
            metadata = {
                "units": a.units,
                "times": times,
                "slice_from": point_count,
                "scale_factor": 0.1,
            }
            point_count += len(times)
            parameter_meta[a.name] = metadata

        base_doc = {"parameters": parameter_meta, "number_of_points": point_count}
        with self._open(data_dir, "meta.json", "w") as f:
            json.dump(base_doc, f)


if __name__ == "__main__":
    cli()
