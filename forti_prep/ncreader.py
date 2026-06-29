import numpy as np
import xarray as xr

from . import convert
from .config import Configuration
from .data import Data
from .projection import LatLon, _reproject_to_latlon, find_latlon


class NetcdfReader:
    def __init__(self, nc_file: str, config: Configuration):
        ds = xr.open_dataset(nc_file)

        for dim in ds.dims:
            dim_name = config.dimensions
            if dim not in ("time", dim_name.x, dim_name.y):
                ds = ds.isel(**{dim: 0}, drop=True)  # type: ignore

        self._ds = ds
        self._config = config

    def get_latlon(self) -> LatLon:
        # Config-independent discovery covers the vast majority of CF datasets.
        result = find_latlon(self._ds)
        if result is not None:
            return result

        # Fallback: use configured dimension names when standard_names are absent.
        x_coord = self._ds[self._config.dimensions.x]
        y_coord = self._ds[self._config.dimensions.y]

        grid_mapping_var = next(
            (
                self._ds[v]
                for v in self._ds.data_vars
                if "grid_mapping_name" in self._ds[v].attrs
            ),
            None,
        )

        if grid_mapping_var is None:
            # No projection metadata — treat the coords as longitude/latitude.
            xx, yy = np.meshgrid(x_coord.values, y_coord.values)
            return LatLon(lat=yy, lon=xx)

        return _reproject_to_latlon(x_coord, y_coord, grid_mapping_var)

    def get_parameters(self) -> list[Data]:
        return [
            convert.transform(param, self._ds, cfg)
            for param, cfg in self._config.parameters.items()
        ]
