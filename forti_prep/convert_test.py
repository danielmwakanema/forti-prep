import unittest

import numpy as np
import xarray as xr

from forti_prep import config, convert


def _dataset(**data_vars) -> xr.Dataset:
    times = np.array(["2024-02-20T00:00:00", "2024-02-20T01:00:00"], dtype="datetime64[s]")
    return xr.Dataset(
        data_vars=data_vars,
        coords={"time": times},
    )


class TestSumVariables(unittest.TestCase):
    def test_sums_matching_units(self):
        ds = _dataset(
            RAINC=(
                ["time", "y", "x"],
                np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
                {"units": "mm"},
            ),
            RAINNC=(
                ["time", "y", "x"],
                np.array([[[10.0, 20.0]], [[30.0, 40.0]]]),
                {"units": "mm"},
            ),
        )
        cfg = config.ParameterConfiguration(
            conversion_method="sum",
            variables={
                "a": config.SimpleParameterConfiguration(netcdf_name="RAINC", units="mm"),
                "b": config.SimpleParameterConfiguration(netcdf_name="RAINNC", units="mm"),
            },
        )

        result = convert.sum_variables("total_precipitation", ds, cfg)

        np.testing.assert_array_equal(
            result.values, np.array([[[11.0, 22.0]], [[33.0, 44.0]]])
        )
        self.assertEqual("mm", result.units)

    def test_raises_on_mismatched_units(self):
        ds = _dataset(
            A=(["time", "y", "x"], np.zeros((2, 1, 2)), {"units": "mm"}),
            B=(["time", "y", "x"], np.zeros((2, 1, 2)), {"units": "cm"}),
        )
        cfg = config.ParameterConfiguration(
            conversion_method="sum",
            variables={
                "a": config.SimpleParameterConfiguration(netcdf_name="A", units="mm"),
                "b": config.SimpleParameterConfiguration(netcdf_name="B", units="cm"),
            },
        )

        with self.assertRaises(ValueError):
            convert.sum_variables("total", ds, cfg)


class TestMaxOverDim(unittest.TestCase):
    def test_reduces_dimensionless_fraction_to_percent(self):
        ds = _dataset(
            CLDFRA=(
                ["time", "level", "y", "x"],
                np.array(
                    [
                        [[[0.1, 0.2]], [[0.5, 0.9]]],
                        [[[0.0, 0.0]], [[0.3, 0.1]]],
                    ]
                ),
                {"units": ""},
            ),
        )
        cfg = config.ParameterConfiguration(
            conversion_method="max_over_dim",
            dim="level",
            variables={
                "total_cloud_cover": config.SimpleParameterConfiguration(
                    netcdf_name="CLDFRA", units="%"
                )
            },
        )

        result = convert.max_over_dim("total_cloud_cover", ds, cfg)

        np.testing.assert_allclose(result.values, np.array([[[50.0, 90.0]], [[30.0, 10.0]]]))
        self.assertEqual("%", result.units)

    def test_raises_without_dim(self):
        ds = _dataset(
            CLDFRA=(["time", "level", "y", "x"], np.zeros((2, 1, 1, 2)), {"units": ""})
        )
        cfg = config.ParameterConfiguration(
            conversion_method="max_over_dim",
            variables={
                "total_cloud_cover": config.SimpleParameterConfiguration(
                    netcdf_name="CLDFRA", units="%"
                )
            },
        )

        with self.assertRaises(ValueError):
            convert.max_over_dim("total_cloud_cover", ds, cfg)


class TestPrecipitationAmount6h(unittest.TestCase):
    def test_true_six_step_difference_not_negative(self):
        # A monotonically increasing accumulator, as WRF's TACC_PRECIP is.
        acc = np.array([0, 1, 2, 4, 7, 11, 16, 22, 29, 37], dtype=np.float64)
        times = np.arange("2024-02-20", "2024-02-21", dtype="datetime64[h]")[: len(acc)]
        ds = xr.Dataset(
            data_vars={"TACC_PRECIP": (["time"], acc, {"units": "mm"})},
            coords={"time": times},
        )
        cfg = config.ParameterConfiguration(
            conversion_method="precipitation_amount_6h",
            variables={
                "total_precipitation": config.SimpleParameterConfiguration(
                    netcdf_name="TACC_PRECIP", units="mm"
                )
            },
        )

        result = convert.precipitation_amount_6h("precipitation_amount_6h", ds, cfg)

        expected = acc[6:] - acc[:-6]
        np.testing.assert_array_equal(result.values, expected)
        self.assertTrue((result.values >= 0).all())


class TestPrecipitationAmount6hCadence(unittest.TestCase):
    def _run(self, step_hours: int, n: int):
        acc = np.arange(n, dtype=np.float64) ** 2
        times = np.datetime64("2024-02-20T00") + np.arange(n) * np.timedelta64(step_hours, "h")
        ds = xr.Dataset(
            data_vars={"TACC_PRECIP": (["time"], acc, {"units": "mm"})},
            coords={"time": times},
        )
        cfg = config.ParameterConfiguration(
            conversion_method="precipitation_amount_6h",
            variables={
                "total_precipitation": config.SimpleParameterConfiguration(
                    netcdf_name="TACC_PRECIP", units="mm"
                )
            },
        )
        return acc, times, convert.precipitation_amount_6h("p", ds, cfg)

    def test_three_hourly_output_spans_two_steps(self):
        acc, times, result = self._run(step_hours=3, n=8)

        np.testing.assert_array_equal(result.values, acc[2:] - acc[:-2])
        np.testing.assert_array_equal(result.times, times[2:])

    def test_interval_not_dividing_six_hours_raises(self):
        with self.assertRaises(ValueError):
            self._run(step_hours=4, n=8)


def _three_hourly(n: int, **data_vars) -> xr.Dataset:
    times = np.datetime64("2024-02-20T06") + np.arange(n) * np.timedelta64(3, "h")
    return xr.Dataset(data_vars=data_vars, coords={"time": times})


class TestWindowReduce(unittest.TestCase):
    def _cfg(self, method: str, netcdf_name: str):
        return config.ParameterConfiguration(
            conversion_method=method,
            hours=6,
            variables={
                "v": config.SimpleParameterConfiguration(netcdf_name=netcdf_name, units="celsius")
            },
        )

    def test_max_and_min_over_six_hours_of_three_hourly_data(self):
        # First sample is WRF's 0 K fill value and must never be used.
        kelvin = np.array([0, 300, 301, 299, 305, 298], dtype=np.float64)
        ds = _three_hourly(6, T02_MAX=(["time"], kelvin, {"units": "K"}), T02_MIN=(["time"], kelvin, {"units": "K"}))

        high = convert.max_over_window("hi", ds, self._cfg("max_over_window", "T02_MAX"))
        low = convert.min_over_window("lo", ds, self._cfg("min_over_window", "T02_MIN"))

        np.testing.assert_allclose(high.values, np.array([301, 301, 305, 305]) - 273.15)
        np.testing.assert_allclose(low.values, np.array([300, 299, 299, 298]) - 273.15)
        np.testing.assert_array_equal(high.times, ds.time.values[2:])
        self.assertEqual("celsius", high.units)

    def test_raises_without_hours(self):
        ds = _three_hourly(4, T=(["time"], np.zeros(4), {"units": "K"}))
        cfg = config.ParameterConfiguration(
            conversion_method="max_over_window",
            variables={"v": config.SimpleParameterConfiguration(netcdf_name="T", units="K")},
        )

        with self.assertRaises(ValueError):
            convert.max_over_window("x", ds, cfg)


class TestSymbolsOnThreeHourlyData(unittest.TestCase):
    def test_symbol_6h_uses_config_name_and_two_step_window(self):
        ds = _three_hourly(
            6,
            TACC=(["time"], np.array([0, 0, 6, 6, 6, 6], dtype=np.float64), {"units": "mm"}),
            CLD=(["time"], np.full(6, 1.0), {"units": "%"}),
        )
        cfg = config.ParameterConfiguration(
            conversion_method="weather_symbol_6h",
            timezone="Africa/Dar_es_Salaam",
            variables={
                "total_precipitation": config.SimpleParameterConfiguration(netcdf_name="TACC", units="mm"),
                "total_cloud_cover": config.SimpleParameterConfiguration(
                    netcdf_name="CLD", units="%", source_units="1"
                ),
            },
        )

        result = convert.symbol_6h("weather_symbol_6h", ds, cfg)

        self.assertEqual("weather_symbol_6h", result.name)
        self.assertEqual((4,), result.values.shape)
        np.testing.assert_array_equal(result.times, ds.time.values[2:])

    def test_symbol_1h_rejects_three_hourly_data(self):
        ds = _three_hourly(
            6,
            TACC=(["time"], np.zeros(6), {"units": "mm"}),
            CLD=(["time"], np.zeros(6), {"units": "%"}),
        )
        cfg = config.ParameterConfiguration(
            conversion_method="weather_symbol_1h",
            variables={
                "total_precipitation": config.SimpleParameterConfiguration(netcdf_name="TACC", units="mm"),
                "total_cloud_cover": config.SimpleParameterConfiguration(netcdf_name="CLD", units="%"),
            },
        )

        with self.assertRaises(ValueError):
            convert.symbol_1h("weather_symbol_1h", ds, cfg)


class TestSourceUnitsOverride(unittest.TestCase):
    def test_overrides_incorrect_file_units(self):
        # Mirrors WRF's CLDFRAC2D: attrs claim "%" but values are a 0-1 fraction.
        ds = _dataset(
            CLDFRAC2D=(["time", "y", "x"], np.array([[[0.5, 1.0]], [[0.0, 0.25]]]), {"units": "%"}),
        )
        cfg = config.SimpleParameterConfiguration(
            netcdf_name="CLDFRAC2D", units="%", source_units="1"
        )

        result = convert.read_values("cloud_area_fraction", ds, cfg)

        np.testing.assert_allclose(result.values, np.array([[[50.0, 100.0]], [[0.0, 25.0]]]))


class TestSelectLevel(unittest.TestCase):
    def test_selects_index_and_converts_units(self):
        ds = _dataset(
            SMOIS=(
                ["time", "level", "y", "x"],
                np.array(
                    [
                        [[[0.1, 0.2]], [[0.3, 0.4]]],
                        [[[0.5, 0.6]], [[0.7, 0.8]]],
                    ]
                ),
                {"units": "m3 m-3"},
            ),
        )
        cfg = config.ParameterConfiguration(
            conversion_method="select_level",
            dim="level",
            index=0,
            variables={
                "soil_moisture": config.SimpleParameterConfiguration(
                    netcdf_name="SMOIS", units="%"
                )
            },
        )

        result = convert.select_level("soil_moisture", ds, cfg)

        np.testing.assert_allclose(result.values, np.array([[[10.0, 20.0]], [[50.0, 60.0]]]))
        self.assertEqual("%", result.units)

    def test_raises_without_dim(self):
        ds = _dataset(
            SMOIS=(["time", "level", "y", "x"], np.zeros((2, 2, 1, 2)), {"units": "m3 m-3"})
        )
        cfg = config.ParameterConfiguration(
            conversion_method="select_level",
            variables={
                "soil_moisture": config.SimpleParameterConfiguration(
                    netcdf_name="SMOIS", units="%"
                )
            },
        )

        with self.assertRaises(ValueError):
            convert.select_level("soil_moisture", ds, cfg)


if __name__ == "__main__":
    unittest.main()
