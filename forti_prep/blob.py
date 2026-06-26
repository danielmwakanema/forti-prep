import hashlib
import json
import os
import typing

import numpy as np

from .config import Configuration
from .data import Data
from .ncreader import LatLon, NetcdfReader


class BlobWriter:
    def __init__(self, cfg: Configuration, workdir: str):
        self._cfg = cfg
        self._workdir = workdir

    def write_data(self, filename: str, version: int):
        reader = NetcdfReader(filename, self._cfg)

        data_dir = self._get_data_dir(reader, version)

        self._write_latlon(reader.get_latlon(), data_dir)
        self._write_data(reader.get_parameters(), data_dir)
        self._write_complete(version)
        self._set_latest(version)

    def _set_latest(self, version: int):
        os.makedirs(os.path.join(self._workdir, "latest"), exist_ok=True)
        filename = os.path.join(self._workdir, "latest", self._cfg.area)
        with open(filename, "w") as f:
            f.write(str(version))

    def _write_complete(self, version: int):
        filename = os.path.join(
            self._workdir, self._cfg.area, str(version), "complete.json"
        )
        with open(filename, "w") as f:
            data = {
                "area": self._cfg.area,
                "version": version,
                "time_until_next": 3600000000000,
                "geographic_extent": None,
            }
            json.dump(data, f)

    def _get_data_dir(self, reader: NetcdfReader, version: int) -> str:
        m = hashlib.md5()
        m.update(reader._ds[self._cfg.dimensions.y].values.tobytes())
        m.update(reader._ds[self._cfg.dimensions.x].values.tobytes())
        data_dir = os.path.join(
            self._workdir, self._cfg.area, str(version), m.hexdigest()
        )
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def _open(self, data_dir: str, filename: str, mode="wb") -> typing.IO:
        full_path = os.path.join(data_dir, filename)
        return open(full_path, mode)

    def _write_latlon(self, latlon: LatLon, data_dir: str):
        with self._open(data_dir, "latitude") as f:
            latlon.lat.astype(np.float32).tofile(f)
        with self._open(data_dir, "longitude") as f:
            latlon.lon.astype(np.float32).tofile(f)

    def _write_data(self, parameters: list[Data], data_dir: str):
        raw_data = [d.values for d in parameters]
        _, ny, nx = parameters[0].values.shape

        with self._open(data_dir, "data") as f:
            for y in range(ny):
                for x in range(nx):
                    for d in raw_data:
                        values = np.round(d[:, y, x] * 10).astype("<i2")
                        values.tofile(f)

        self._write_meta(parameters, data_dir)

    def _write_meta(self, arrays: list[Data], data_dir: str):
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
