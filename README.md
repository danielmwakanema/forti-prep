# nc2forti

Converts NetCDF forecast files into the binary format expected by the [forti](https://github.com/metno/forti/) system, and writes the result to a local directory.

## Usage

```
python main.py --config <config.json> --output-dir <output> --version <n> <nc-file>
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

Each variable entry has a `units` field specifying the desired output unit. If the unit in the NetCDF file differs, it is automatically converted using [cf-units](https://github.com/SciTools/cf-units).

Supported `conversion_method` values:

| Value | Description |
|---|---|
| `null` | No conversion; reads the variable directly |
| `vector_to_speed` | Wind speed from x/y vector components |
| `vector_to_direction_from` | Direction the wind is coming from (meteorological convention) |
| `vector_to_direction_to` | Direction the wind is blowing towards |
| `precipitation_amount_1h` | 1-hour precipitation from accumulated values |
| `precipitation_amount_6h` | 6-hour precipitation from accumulated values |
| `weather_symbol_1h` | Weather symbol from 1-hour precipitation and cloud cover |
| `weather_symbol_6h` | Weather symbol from 6-hour precipitation and cloud cover |
| `weather_symbol_12h` | Weather symbol from 12-hour precipitation and cloud cover |

## Setup

Requires Python 3.13+. Dependencies are managed with [uv](https://github.com/astral-sh/uv):

```
uv sync
```
