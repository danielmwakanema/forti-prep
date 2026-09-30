# forti-prep

Converts NetCDF forecast files into the binary format expected by the [forti](https://github.com/metno/forti/) system, and writes the result to a local directory.

## Usage

```
forti-prep --config <config.json> --output-dir <output> --version <n> <nc-file>
```

| Option | Default | Description |
|---|---|---|
| `--config` | `config.json` | JSON config file describing parameters to extract |
| `--output-dir` | `data` | Directory to write output files into |
| `--version` | `0` | Version number to attach to the output |

To use, copy `sample_config.json` to `config.json` in the project root and populate it for your model/area — `--config` defaults to `config.json`. See `sample_config.json` for a config example, and the `dimensions.time`, `coordinates` and `source_units` notes below for what a non-CF model like WRF (`wrfout_*`) needs.

## Config format

```json
{
  "area": "arctic",
  "dimensions": {
    "x": "x",
    "y": "y",
    "time": { "dimension": "time" }
  },
  "coordinates": { "latitude": "latitude", "longitude": "longitude" },
  "parameters": {
    "wind_speed_10m": {
      "conversion_method": "vector_to_speed",
      "variables": {
        "x": { "netcdf_name": "x_wind_10m", "units": "m/s" },
        "y": { "netcdf_name": "y_wind_10m", "units": "m/s" }
      }
    }
  }
}
```

The top-level fields are:

| Field | Description |
|---|---|
| `area` | Identifier for the geographic area (e.g. `"arctic"`, `"meps"`) |
| `dimensions` | Names of the x, y and time dimensions in the NetCDF file |
| `coordinates` | Optional. Explicit lat/lon variable names, for models without CF `standard_name`/`grid_mapping` metadata |
| `parameters` | Map of output parameter names to their extraction configuration |

`dimensions.x` and `dimensions.y` are the spatial dimension names. `dimensions.time` is optional and defaults to `{"dimension": "time"}` (as shown above) — CF-compliant models like MEPS can omit it entirely. Set it when a model doesn't use the CF-standard `time` dimension/coordinate — e.g. WRF, whose time dimension is `Time` and whose decoded time values live in a separate `XTIME` variable:

```json
"dimensions": {
  "x": "west_east",
  "y": "south_north",
  "time": { "dimension": "Time", "variable": "XTIME" }
}
```

`coordinates` is optional too — the sample above sets it to `latitude`/`longitude`, which is also what CF auto-detection would find on its own for a compliant model, so it's a no-op there and can just as well be removed. Set it (or change the names) when a model has no `latitude`/`longitude` coordinates and no CF `grid_mapping` variable to auto-detect a projection from — e.g. WRF, which exposes the grid as 2D `XLAT`/`XLONG` fields instead:

```json
"coordinates": { "latitude": "XLAT", "longitude": "XLONG" }
```

Each parameter entry has a `conversion_method` and a `variables` map. Each variable entry requires `netcdf_name` (the variable name in the NetCDF file) and `units` (the desired output unit). If the unit in the NetCDF file differs, it is automatically converted using [cf-units](https://github.com/SciTools/cf-units). `max_over_dim` and `select_level` also require a `dim` field naming the dimension to reduce over; `select_level` additionally takes an optional `index` (default `0`) picking which position along that dimension to keep. `max_over_window` and `min_over_window` require `hours`, the length of the window. The `weather_symbol_*` methods take an optional `timezone` (an IANA name such as `"Africa/Dar_es_Salaam"`), used to decide whether a symbol is a day or night variant; without it they assume UTC+2.

A variable entry may also set `source_units`, overriding the NetCDF file's own `units` attribute for that variable. Use this when a model's metadata is wrong or missing — e.g. WRF's `CLDFRAC2D` claims `units = "%"` but actually stores a 0-1 fraction, so its config sets `"source_units": "1"` to get a correct conversion to `%`.

The keys of `parameters` are the names forti sees, and forti's `jsonfrontend` only serves parameters listed in its own mapping (`jsonformat.json`). By default that includes e.g. `air_temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_from_direction_10m`, `cloud_area_fraction`, `precipitation_amount_acc1h` and `precipitation_amount_acc6h`; anything else (e.g. soil moisture) is stored but not served until you add it to a custom `jsonformat.json`.

Supported `conversion_method` values and their required variable keys:

| Value | Required variable keys | Description |
|---|---|---|
| `null` | *(parameter name)* | No conversion; reads the variable whose key matches the parameter name directly |
| `vector_to_speed` | `x`, `y` | Wind speed from x/y vector components |
| `vector_to_direction_from` | `x`, `y` | Direction the wind is coming from (meteorological convention) |
| `vector_to_direction_to` | `x`, `y` | Direction the wind is blowing towards |
| `precipitation_amount_1h` | `total_precipitation` | 1-hour precipitation from accumulated values |
| `precipitation_amount_6h` | `total_precipitation` | 6-hour precipitation from accumulated values |
| `weather_symbol_1h` | `total_precipitation`, `total_cloud_cover` | 1-hour weather symbol from accumulated precipitation and cloud cover |
| `weather_symbol_6h` | `total_precipitation`, `total_cloud_cover` | 6-hour weather symbol from accumulated precipitation and cloud cover |
| `weather_symbol_12h` | `total_precipitation`, `total_cloud_cover` | 12-hour weather symbol from accumulated precipitation and cloud cover |
| `sum` | *(any number of keys)* | Adds together several variables sharing one unit, e.g. WRF's `RAINC` + `RAINNC` |
| `max_over_dim` | *(single key)* | Reduces one variable to 2D by taking its maximum over the dimension named in `dim`, e.g. WRF's per-level `CLDFRA` |
| `select_level` | *(single key)* | Reduces one variable to 2D by picking `index` along the dimension named in `dim`, e.g. WRF's topmost soil layer in `SMOIS`/`TSLB` |
| `max_over_window` / `min_over_window` | *(single key)* | Largest / smallest value over the last `hours` hours, e.g. a 6h max of WRF's per-interval `T02_MAX`. The first `hours` worth of steps are dropped since they have no full window |

The 1h/6h/12h precipitation and weather symbol methods measure their windows in hours using the file's actual output interval (a 6h window is 2 steps in a 3-hourly file), and fail if the interval doesn't divide the window evenly.

## Setup

Requires Python 3.13+. Dependencies are managed with [uv](https://github.com/astral-sh/uv):

```
uv sync
```
