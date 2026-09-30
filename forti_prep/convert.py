import datetime as dt
import typing
import zoneinfo

import numpy as np
import pyproj
import xarray as xr
from cf_units import Unit

from . import config, weather_symbol
from .data import Data
from .projection import find_latlon


def _convert_units(values: np.ndarray, src_unit_str: str, target_unit_str: str) -> np.ndarray:
    if src_unit_str in ("kg m**-2", "kg/m^2"):
        src_unit_str = "mm"
    if src_unit_str in ("", "dimensionless"):
        # Some models (e.g. WRF's CLDFRA, RH02) leave `units` blank or write
        # "dimensionless" for a plain 0-1 fraction rather than using the CF "1"
        # udunits actually recognizes.
        src_unit_str = "1"

    src_unit = Unit(src_unit_str)
    target_unit = Unit(target_unit_str)

    if src_unit == target_unit:
        return values
    return src_unit.convert(values, target_unit)


def read_values(
    name: str, ds: xr.Dataset, cfg: config.SimpleParameterConfiguration
) -> Data:
    da = ds[cfg.netcdf_name]

    src_units = cfg.source_units or da.attrs["units"]
    values = _convert_units(da.values, src_units, cfg.units)

    return Data(
        name=name,
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
    step = _steps_per(ds.time.values, 1)
    acc = read_values("", ds, cfg.variables["total_precipitation"])

    values = acc.values[step:] - acc.values[:-step]

    assert values.shape[0] + step == len(ds.time.values)

    return Data(name=name, values=values, times=ds.time.values[step:], units="mm")


def _steps_per(times: np.ndarray, hours: int) -> int:
    """Number of output steps spanning `hours`, from the file's actual cadence."""
    deltas = np.unique(np.diff(times))
    if len(deltas) != 1:
        raise ValueError("time axis is not evenly spaced")
    step_hours = deltas[0] / np.timedelta64(1, "h")
    if hours % step_hours != 0:
        raise ValueError(f"{hours}h is not a multiple of the {step_hours}h output interval")
    return int(hours // step_hours)


def precipitation_amount_6h(
    name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration
) -> Data:

    timestep_count = _steps_per(ds.time.values, 6)

    acc = read_values("", ds, cfg.variables["total_precipitation"])
    # A true "value `timestep_count` steps later minus now" difference.
    # np.diff(..., n=timestep_count) is NOT this: it's the n-th order finite
    # difference, which produces near-meaningless (and sometimes negative)
    # results for accumulator data.
    values = acc.values[timestep_count:] - acc.values[:-timestep_count]

    assert values.shape[0] + timestep_count == len(ds.time.values)

    return Data(
        name=name, values=values, times=ds.time.values[timestep_count:], units="mm"
    )


def _symbol_tz(cfg: config.ParameterConfiguration) -> dt.tzinfo:
    if cfg.timezone is None:
        return weather_symbol.DEFAULT_TZ
    return zoneinfo.ZoneInfo(cfg.timezone)


def symbol_1h(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    _steps_per(ds.time.values, 1)
    precip = read_values("", ds, cfg.variables["total_precipitation"])
    cloud = read_values("", ds, cfg.variables["total_cloud_cover"])
    symbol = weather_symbol.get_weather_symbol_1h(precip, cloud, tz=_symbol_tz(cfg))
    symbol.name = name
    return symbol


def symbol_6h(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    steps = _steps_per(ds.time.values, 6)
    precip = read_values("", ds, cfg.variables["total_precipitation"])
    cloud = read_values("", ds, cfg.variables["total_cloud_cover"])
    symbol = weather_symbol.get_weather_symbol_6h(
        precip, cloud, steps=steps, tz=_symbol_tz(cfg)
    )
    symbol.name = name
    return symbol


def symbol_12h(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    steps = _steps_per(ds.time.values, 12)
    precip = read_values("", ds, cfg.variables["total_precipitation"])
    cloud = read_values("", ds, cfg.variables["total_cloud_cover"])
    symbol = weather_symbol.get_weather_symbol_12h(
        precip, cloud, steps=steps, tz=_symbol_tz(cfg)
    )
    symbol.name = name
    return symbol


def _window_reduce(
    name: str,
    ds: xr.Dataset,
    cfg: config.ParameterConfiguration,
    reduce: typing.Callable[..., np.ndarray],
) -> Data:
    """Reduce each sample with the ones before it, over a `hours`-long window.

    Meant for period extremes (e.g. WRF's T02_MAX/T02_MIN, each covering the
    preceding output interval): each result at time T covers (T - hours, T], so
    the first `hours` worth of steps have no full window and are dropped. That
    also drops the unusable first step, which WRF fills with 0 K.
    """
    if cfg.hours is None:
        raise ValueError("this conversion requires an 'hours' config value")

    var_cfg = next(iter(cfg.variables.values()))
    data = read_values(name, ds, var_cfg)
    k = _steps_per(ds.time.values, cfg.hours)
    n = len(data.values)

    windows = np.stack([data.values[k - j : n - j] for j in range(k)])
    return Data(
        name=name,
        values=reduce(windows, axis=0),
        times=ds.time.values[k:],
        units=data.units,
    )


def max_over_window(
    name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration
) -> Data:
    return _window_reduce(name, ds, cfg, np.max)


def min_over_window(
    name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration
) -> Data:
    return _window_reduce(name, ds, cfg, np.min)


def sum_variables(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    """Add together several variables holding the same physical unit.

    Useful for models that split a quantity across components, e.g. WRF
    reporting accumulated precipitation as separate convective (RAINC) and
    grid-scale (RAINNC) fields that must be summed into one total.
    """
    parts = [read_values("", ds, var_cfg) for var_cfg in cfg.variables.values()]

    units = {p.units for p in parts}
    if len(units) != 1:
        raise ValueError(f"sum requires all variables to share one unit, got {units}")

    total = np.sum(np.stack([p.values for p in parts]), axis=0)
    return Data(name=name, values=total, times=parts[0].times, units=parts[0].units)


def max_over_dim(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    """Reduce a variable to 2D by taking its maximum over a named dimension.

    Useful for models that report a per-level field (e.g. WRF's 3D CLDFRA
    cloud fraction per model level) where only the column maximum is needed.
    """
    if cfg.dim is None:
        raise ValueError("max_over_dim requires a 'dim' config value")

    var_cfg = next(iter(cfg.variables.values()))
    da = ds[var_cfg.netcdf_name]

    src_units = var_cfg.source_units or da.attrs["units"]
    values = _convert_units(da.max(dim=cfg.dim).values, src_units, var_cfg.units)
    return Data(name=name, values=values, times=ds.time.values, units=var_cfg.units)


def select_level(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    """Reduce a variable to 2D by selecting a single index along a named dimension.

    Useful for models that report a field per vertical level (e.g. WRF's
    soil-layer SMOIS/TSLB) where only one specific layer is wanted, selected
    via `dim` and `index` (default 0, e.g. WRF's topmost/near-surface layer).
    """
    if cfg.dim is None:
        raise ValueError("select_level requires a 'dim' config value")

    var_cfg = next(iter(cfg.variables.values()))
    da = ds[var_cfg.netcdf_name].isel({cfg.dim: cfg.index})

    src_units = var_cfg.source_units or da.attrs["units"]
    values = _convert_units(da.values, src_units, var_cfg.units)
    return Data(name=name, values=values, times=ds.time.values, units=var_cfg.units)


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
    "sum": sum_variables,
    "max_over_dim": max_over_dim,
    "select_level": select_level,
    "max_over_window": max_over_window,
    "min_over_window": min_over_window,
}


def transform(name: str, ds: xr.Dataset, cfg: config.ParameterConfiguration) -> Data:
    if cfg.conversion_method is None:
        return read_values(name, ds, cfg.variables[name])
    func = _conversions[cfg.conversion_method]
    return func(name, ds, cfg)
