import typing

import numpy as np
import pyproj
import xarray as xr
from cf_units import Unit

from . import config, weather_symbol
from .data import Data
from .projection import find_latlon


def read_values(
    name: str, ds: xr.Dataset, cfg: config.SimpleParameterConfiguration
) -> Data:
    da = ds[cfg.netcdf_name]

    src_unit_str = da.attrs["units"]
    if src_unit_str in ("kg m**-2", "kg/m^2"):
        src_unit_str = "mm"

    src_unit = Unit(src_unit_str)
    target_unit = Unit(cfg.units)

    values = da.values
    if src_unit != target_unit:
        values = src_unit.convert(values, target_unit)

    # timesteps = {}
    # for idx, t in enumerate(ds.time.values):
    #     timesteps[t] = values[idx]

    return Data(
        name=name,
        # timesteps=timesteps,
        values=values,
        times=ds.time.values,
        units=cfg.units,
    )


def speed(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    x = read_values("", ds, cfg.variables["x"])
    y = read_values("", ds, cfg.variables["y"])

    values = np.sqrt(x.values**2 + y.values**2)
    return Data(
        name=name,
        # timesteps=x.timesteps,
        values=values,
        times=ds.time.values,
        units="m/s",
    )


def from_direction(
    name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration
) -> Data:
    ret = to_direction(name, ds, cfg)
    ret.values = (ret.values + 180) % 360
    return ret


def _grid_rotation_angles(ds: xr.Dataset) -> np.ndarray | float:
    """Return the clockwise angle (radians) from each grid point's y-axis to true north.

    This is non-zero whenever the dataset's projection causes the grid axes to
    diverge from geographic north/east (e.g. Lambert Conformal, Polar Stereographic).
    Returns a scalar 0.0 when no grid-mapping variable is found.
    """
    grid_mapping_var = next(
        (ds[v] for v in ds.data_vars if "grid_mapping_name" in ds[v].attrs), None
    )
    if grid_mapping_var is None:
        return 0.0

    latlon = find_latlon(ds)
    if latlon is None:
        return 0.0

    proj = pyproj.Proj(pyproj.CRS.from_cf(grid_mapping_var.attrs))

    delta = 1e-5  # degrees — small enough for accuracy, large enough to avoid rounding
    x1, y1 = proj(latlon.lon, latlon.lat)
    x2, y2 = proj(latlon.lon, latlon.lat + delta)

    dx = x2 - x1
    dy = y2 - y1
    # arctan2(dx, dy): angle of the true-north displacement vector from the grid y-axis,
    # measured clockwise — i.e. how much the grid is rotated relative to geographic north.
    return np.arctan2(dx, dy)


def to_direction(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    x = read_values("", ds, cfg.variables["x"])
    y = read_values("", ds, cfg.variables["y"])

    alpha = _grid_rotation_angles(ds)  # shape (ny, nx) or scalar 0.0

    values = np.degrees(np.arctan2(y.values, x.values) + alpha)
    return Data(
        name=name,
        values=values % 360,  # force to be in range [0..360)
        times=ds.time.values,
        units="degrees",
    )


def precipitation_amount_1h(
    name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration
) -> Data:
    acc = read_values("", ds, cfg.variables["precipitation_amount_acc"])

    values = np.diff(acc.values, axis=0)
    times = ds.time[1:]

    assert values.shape[0] + 1 == len(ds.time.values)

    return Data(name=name, values=values, times=ds.time.values[1:], units="mm")


def precipitation_amount_6h(
    name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration
) -> Data:

    timestep_count = 6

    acc = read_values("", ds, cfg.variables["precipitation_amount_acc"])
    values = np.diff(acc.values, n=timestep_count, axis=0)

    assert values.shape[0] + timestep_count == len(ds.time.values)

    return Data(
        name=name, values=values, times=ds.time.values[timestep_count:], units="mm"
    )


def symbol_1h(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    precip = read_values("", ds, cfg.variables["precipitation_amount"])
    cloud = read_values("", ds, cfg.variables["cloud_area_fraction"])
    return weather_symbol.get_weather_symbol_1h(precip, cloud)


def symbol_6h(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    precip = read_values("", ds, cfg.variables["precipitation_amount"])
    cloud = read_values("", ds, cfg.variables["cloud_area_fraction"])
    return weather_symbol.get_weather_symbol_6h(precip, cloud)


def symbol_12h(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    precip = read_values("", ds, cfg.variables["precipitation_amount"])
    cloud = read_values("", ds, cfg.variables["cloud_area_fraction"])
    return weather_symbol.get_weather_symbol_12h(precip, cloud)


_conversion_function = typing.Callable[
    [str, xr.Dataset, config.ParameterConfiguration], Data
]

_conversions: typing.Dict[str, _conversion_function] = {
    "vector_to_speed": speed,
    "vector_to_direction_from": from_direction,
    "vector_to_direction_to": to_direction,
    "weather_symbol_1h": symbol_1h,
    "weather_symbol_6h": symbol_6h,
    "weather_symbol_12h": symbol_12h,
    "precipitation_amount_1h": precipitation_amount_1h,
    "precipitation_amount_6h": precipitation_amount_6h,
}


def transform(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    if cfg.conversion_method is None:
        return read_values(name, ds, cfg.variables[name])
    func = _conversions[cfg.conversion_method]
    return func(name, ds, cfg)
