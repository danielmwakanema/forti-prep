import datetime as dt
import typing
from typing import Optional

import bottleneck as bn
import numpy as np

from .data import Data, zeros


def get_weather_symbol_1h(
    accumulated_precipitation: Data,
    cloud_cover_in_percent: Data,
    fog_in_percent: Optional[Data] = None,
    # thunder: Optional[np.ndarray] = None,
) -> Data:

    # TODO: Support thunder
    droplets = _get_droplets_1h(accumulated_precipitation)

    clouds = _get_cloud_cover_code(cloud_cover_in_percent)
    if fog_in_percent is not None:
        fog = _get_fog_code(fog_in_percent)
    else:
        fog = zeros(clouds)
    thunder = zeros(clouds)

    s = _calculate_symbol("weather_symbol_1h", droplets, clouds, fog, thunder)

    _update_sun_state_in(s, dt.timedelta(hours=1))

    return s


def get_weather_symbol_6h(
    accumulated_precipitation: Data,
    cloud_cover_in_percent: Data,
    fog_in_percent: Optional[Data] = None,
    # thunder: Optional[Data] = None,
) -> Data:
    # TODO: Support thunder

    droplets = _get_droplets_6h(accumulated_precipitation)
    clouds = _get_cloud_cover_code(cloud_cover_in_percent, time_resolution=6)
    if fog_in_percent is not None:
        fog = _get_fog_code(fog_in_percent, hours=6)  # type: ignore
    else:
        fog = zeros(droplets)
    thunder = zeros(droplets)

    s = _calculate_symbol("weather_symbol_6h", droplets, clouds, fog, thunder)
    _update_sun_state_in(s, dt.timedelta(hours=6))
    return s


def get_weather_symbol_12h(
    accumulated_precipitation: Data,
    cloud_cover_in_percent: Data,
    fog_in_percent: Optional[Data] = None,
    # thunder: Optional[Data] = None,
) -> Data:
    # TODO: Support thunder

    droplets = _get_droplets_12h(accumulated_precipitation)
    clouds = _get_cloud_cover_code(cloud_cover_in_percent, time_resolution=12)
    if fog_in_percent is not None:
        fog = _get_fog_code(fog_in_percent, hours=12)  # type: ignore
    else:
        fog = zeros(droplets)
    thunder = zeros(droplets)

    s = _calculate_symbol("weather_symbol_12h", droplets, clouds, fog, thunder)
    _update_sun_state_in(s, dt.timedelta(hours=12))
    return s


def _update_sun_state_in(symbols: Data, period: dt.timedelta):
    period /= 2
    for i, t in enumerate(symbols.times.astype("datetime64[s]").tolist()):
        t = (
            t.replace(tzinfo=dt.UTC).astimezone(dt.timezone(dt.timedelta(hours=2)))
            - period
        )
        if t.hour < 6 or t.hour >= 18:
            symbols.values[i] |= night


def _get_droplets_1h(accumulated: Data) -> Data:
    precipitation = np.diff(accumulated.values, axis=0)

    droplets = np.full(precipitation.shape, np.nan, np.float32)
    droplets[precipitation <= 0.1] = 0
    droplets[(precipitation > 0.1) * (precipitation <= 0.25)] = 1
    droplets[(precipitation > 0.25) * (precipitation <= 0.95)] = 2
    droplets[precipitation > 0.95] = 3

    return Data("droplets", droplets, accumulated.times[1:], "")


def _get_droplets_6h(accumulated: Data) -> Data:
    hours = 6

    accum_start = accumulated.values[:-hours]
    accum_end = accumulated.values[hours:]
    precipitation = accum_end - accum_start

    droplets = np.full(precipitation.shape, np.nan, np.float32)
    droplets[precipitation <= 0.5] = 0
    droplets[precipitation > 0.5] = 1
    droplets[precipitation > 0.95] = 2
    droplets[precipitation > 4.95] = 3

    return Data("droplets_6h", droplets, accumulated.times[hours:], "")


def _get_droplets_12h(accumulated: Data) -> Data:
    hours = 12

    accum_start = accumulated.values[:-hours]
    accum_end = accumulated.values[hours:]
    precipitation = accum_end - accum_start

    droplets = np.full(precipitation.shape, np.nan, np.float32)
    droplets[precipitation <= 1] = 0
    droplets[precipitation > 1] = 1
    droplets[precipitation > 1.9] = 2
    droplets[precipitation > 9.9] = 3

    return Data("droplets_6h", droplets, accumulated.times[hours:], "")


def _get_cloud_cover_code(cloud_cover_in_percent: Data, time_resolution=1) -> Data:
    cloud_cover_values = cloud_cover_in_percent.values
    times = cloud_cover_in_percent.times
    if time_resolution > 1:
        cloud_cover_values = bn.move_mean(
            cloud_cover_values, window=time_resolution, min_count=1, axis=0
        )
        assert len(cloud_cover_values) == len(times)

    c = np.zeros(cloud_cover_values.shape, np.float32)
    c[cloud_cover_values <= 13] = 0
    c[(cloud_cover_values > 13)] = 1
    c[(cloud_cover_values > 38)] = 2
    c[cloud_cover_values > 86] = 3

    return Data(
        "cloud_cover_code",
        values=c,
        times=times,
        units="",
    )


def _get_fog_code(fog: Data, hours=1) -> Data:
    fog_values = fog.values
    if hours > 1:
        fog_values = bn.move_mean(fog_values, window=hours, min_count=1, axis=0)

    f = np.zeros(fog_values.shape, np.float32)
    f[fog.values > 0.25] = 1
    return Data("fog_code", f, fog.times, "")


def _calculate_symbol(
    name: str,
    droplets: Data,
    clouds: Data,
    fog: Data,
    thunder: Data,
) -> Data:

    times = set(droplets.times)

    _select_times(times, clouds)
    _select_times(times, fog)
    _select_times(times, thunder)

    assert len(droplets.times) == len(clouds.times)
    assert len(droplets.times) == len(fog.times)
    assert len(droplets.times) == len(thunder.times)

    assert droplets.values.shape == clouds.values.shape
    assert clouds.values.shape == fog.values.shape
    assert fog.values.shape == thunder.values.shape

    # cannot have light clouds with rain
    clouds.values[(clouds.values <= 1) * (droplets.values > 0)] = 2

    symbol_code = np.zeros(droplets.values.shape, np.int16)
    for key, value in symbols:
        symbol_code[
            (key[0] == clouds.values)
            & (key[1] == droplets.values)
            & (key[2] == thunder.values)
        ] = value
    symbol_code[fog.values == 1] = 15

    return Data(name=name, values=symbol_code, times=droplets.times, units="1")


def _select_times(times: typing.Set[np.datetime64], values: Data):
    for idx, t in enumerate(values.times):
        if t in times:
            break
    values.values = values.values[idx:]
    values.times = values.times[idx:]


unknown = -1
clearsky = 1
fair = 2
partlycloudy = 3
cloudy = 4
rainshowers = 5
rainshowersandthunder = 6
rain = 9
heavyrain = 10
heavyrainandthunder = 11
fog = 15
rainandthunder = 22
lightrainshowersandthunder = 24
heavyrainshowersandthunder = 25
lightrainandthunder = 30
lightrainshowers = 40
heavyrainshowers = 41
lightrain = 46

night = 1 << 7


# clouds, precipitation, thunder -> symbol
symbols = (
    ((0, 0, 0), clearsky),
    ((1, 0, 0), fair),
    ((2, 0, 0), partlycloudy),
    ((2, 1, 0), lightrainshowers),
    ((2, 1, 1), lightrainshowersandthunder),
    ((2, 2, 0), rainshowers),
    ((2, 2, 1), rainshowersandthunder),
    ((2, 3, 0), heavyrainshowers),
    ((2, 3, 1), heavyrainshowersandthunder),
    ((3, 0, 0), cloudy),
    ((3, 1, 0), lightrain),
    ((3, 1, 1), lightrainandthunder),
    ((3, 2, 0), rain),
    ((3, 2, 1), rainandthunder),
    ((3, 3, 0), heavyrain),
    ((3, 3, 1), heavyrainandthunder),
)
