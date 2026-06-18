import typing

import numpy as np
import xarray as xr
from cf_units import Unit

from . import config, weather_symbol
from .data import Data


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
    # TODO: Take projection into account

    x = read_values("", ds, cfg.variables["x"])
    y = read_values("", ds, cfg.variables["y"])

    # make this into direction _from_, which is convention for wind and waves.
    values = np.degrees(np.arctan2(y.values, x.values)) + 180
    return Data(
        name=name,
        # timesteps=x.timesteps,
        values=values,
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
