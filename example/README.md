# Example loading

This is an example of how to load data into the forti system. Invoke like this:

```shell
mkdir -p example/data
uv run forti-prep --config example/load.json --output-dir example/data/ https://thredds.met.no/thredds/dodsC/mepslatest/meps_lagged_6_h_latest_2_5km_latest.nc
```

Once this has run to completion, you can point forti's rawdataforecaster configuration to the example/data/ folder, and it should serve the latest meps data.
