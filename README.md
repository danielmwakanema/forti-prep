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

See `sample_config.json` for a config example.

## Config format

```json
{
  "area": "arctic",
  "dimensions": {
    "x": "x",
    "y": "y"
  },
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
| `dimensions` | Names of the x and y spatial dimensions in the NetCDF file |
| `parameters` | Map of output parameter names to their extraction configuration |

Each parameter entry has a `conversion_method` and a `variables` map. Each variable entry requires `netcdf_name` (the variable name in the NetCDF file) and `units` (the desired output unit). If the unit in the NetCDF file differs, it is automatically converted using [cf-units](https://github.com/SciTools/cf-units).

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

## Setup

Requires Python 3.13+. Dependencies are managed with [uv](https://github.com/astral-sh/uv):

```
uv sync
```
