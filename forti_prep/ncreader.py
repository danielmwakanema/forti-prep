import numpy as np
import xarray as xr

from . import convert
from .config import Configuration
from .data import Data
from .projection import LatLon, _reproject_to_latlon, find_latlon


class NetcdfReader:
    def __init__(self, nc_file: str, config: Configuration):
        ds = xr.open_dataset(nc_file)

        dim_name = config.dimensions
        time_dim = dim_name.time.dimension
        for dim in list(ds.dims):
            if dim in (time_dim, dim_name.x, dim_name.y):
                continue
            # Only collapse genuinely degenerate dims. A non-singleton dim (e.g.
            # WRF's vertical "bottom_top") may still be needed by a conversion
            # method such as max_over_dim, so leave it alone.
            if ds.sizes[dim] == 1:
                ds = ds.isel(**{dim: 0}, drop=True)  # type: ignore

        ds = self._normalize_time(ds, config)

        self._ds = ds
        self._config = config

    def _normalize_time(self, ds: xr.Dataset, config: Configuration) -> xr.Dataset:
        """Rename the configured time dimension/variable to the canonical "time".

        Models that don't use CF-standard naming (e.g. WRF's "Time" dimension with
        values stored in a separate "XTIME" variable) can point at their actual
        names via config instead of needing model-specific reader code.
        """
        time_cfg = config.dimensions.time
        rename = {}
        if time_cfg.dimension != "time":
            rename[time_cfg.dimension] = "time"
        if time_cfg.resolved_variable() != "time":
            rename[time_cfg.resolved_variable()] = "time"
        if rename:
            ds = ds.rename(rename)
        return ds

    def get_latlon(self) -> LatLon:
        if self._config.coordinates is not None:
            return self._get_configured_latlon()

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

    def _get_configured_latlon(self) -> LatLon:
        """Read lat/lon from explicitly-named variables (config.coordinates).

        Used when a model doesn't expose CF standard_names or a grid_mapping
        variable to auto-detect (e.g. WRF's XLAT/XLONG). If the named variables
        still carry the time dimension despite describing a static grid, only
        the first time step is used.
        """
        coords = self._config.coordinates
        assert coords is not None
        lat = self._ds[coords.latitude]
        lon = self._ds[coords.longitude]
        if "time" in lat.dims:
            lat = lat.isel(time=0)
        if "time" in lon.dims:
            lon = lon.isel(time=0)
        return LatLon(lat=lat.values, lon=lon.values)

    def get_parameters(self) -> list[Data]:
        return [
            convert.transform(param, self._ds, cfg)
            for param, cfg in self._config.parameters.items()
        ]
