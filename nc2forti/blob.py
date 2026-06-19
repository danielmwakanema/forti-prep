import hashlib
import json
import os
import typing

import numpy as np
import xarray as xr

from . import convert
from .config import Configuration, ParameterConfiguration
from .data import Data


class BlobWriter:
    def __init__(self, cfg: Configuration, workdir: str):
        self._cfg = cfg
        self._workdir = workdir

    def write_data(self, filename: str, model: str, version: int):
        ds = xr.open_dataset(filename)
        for dim in ds.dims:
            if dim not in ("time", "x", "y") and ds.dims[dim] == 1:
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
            if "latitude" not in ds:
                raise Exception("missing latitude in dataset")
            values = ds["latitude"].values
            values.tofile(f)
            # values = np.repeat(ds["y"].values, len(ds["x"].values)).astype(np.float32)
            # values.tofile(f)

    def _write_longitude(self, ds: xr.Dataset, data_dir: str):
        with self._open(data_dir, "longitude") as f:
            if "longitude" not in ds:
                raise Exception("missing longitude in dataset")
            values = ds["longitude"].values
            values.tofile(f)
            # for i in range(len(ds["y"].values)):
            #     ds["x"].values.astype(np.float32).tofile(f)

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
