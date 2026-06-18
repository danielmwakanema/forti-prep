import datetime as dt
import unittest

import numpy as np

import weather_symbol
from convert import Data


class TestWeatherSymbols(unittest.TestCase):
    def test_create_symbols_1h(self):
        times = np.arange(
            "2024-02-20 03:00:00", "2024-02-20 08:00:00", dtype="datetime64[h]"
        )

        symbolData = weather_symbol.get_weather_symbol_1h(
            accumulated_precipitation=Data("", np.array([0, 0.5, 1, 8, 8]), times, ""),
            cloud_cover_in_percent=Data("", np.array([0, 25, 90, 0, 69.8]), times, ""),
            fog_in_percent=Data("", np.array([0, 90, 0, 0, 0]), times, ""),
        )

        self.assertEqual(np.datetime64("2024-02-20 04"), symbolData.times[0])

        symbols = symbolData.values

        self.assertEqual(4, len(symbols))
        self.assertEqual(weather_symbol.fog | weather_symbol.night, symbols[0])
        self.assertEqual(weather_symbol.rain, symbols[1])
        self.assertEqual(weather_symbol.heavyrainshowers, symbols[2])
        self.assertEqual(weather_symbol.partlycloudy, symbols[3])

    def test_create_symbols_6h(self):
        shape = (7,)
        times = np.arange("2024-02-20", "2024-02-20 07:00:00", dtype="datetime64[h]")
        symbols = weather_symbol.get_weather_symbol_6h(
            accumulated_precipitation=Data(
                "rain",
                values=np.zeros(shape, np.float32),
                times=times,
                units="",
            ),
            cloud_cover_in_percent=Data(
                "clouds",
                values=np.zeros(shape, np.float32),
                times=times,
                units="",
            ),
        )

        self.assertEqual((1,), symbols.values.shape)

        self.assertEqual(
            weather_symbol.clearsky | weather_symbol.night, symbols.values[0]
        )
        self.assertEqual(np.datetime64("2024-02-20 06:00:00", "h"), symbols.times[0])

    def test_getDroplets6h(self):
        values = Data(
            "rain",
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10]),
            times=np.arange("2024-02-20", "2024-02-20 13:00:00", dtype="datetime64[h]"),
            units="mm",
        )

        droplets = weather_symbol._get_droplets_6h(values)

        self.assertEqual((7,), droplets.values.shape)
        self.assertEqual(3, droplets.values[0])
        self.assertEqual(2, droplets.values[6])

        self.assertEqual((7,), droplets.times.shape)
        self.assertEqual(
            np.datetime64("2024-02-20 06:00:00", "h"),
            droplets.times[0],
        )
        self.assertEqual(
            np.datetime64("2024-02-20 12:00:00", "h"),
            droplets.times[6],
        )

    def test_multiDimensional_1h(self):
        shape = (12, 4, 3)
        times = np.arange("2024-02-20", "2024-02-20 12:00:00", dtype="datetime64[h]")
        self.assertEqual(12, len(times))

        symbols = weather_symbol.get_weather_symbol_1h(
            accumulated_precipitation=Data(
                "rain",
                values=np.zeros(shape, np.float32),
                times=times,
                units="",
            ),
            cloud_cover_in_percent=Data(
                "clouds",
                values=np.zeros(shape, np.float32),
                times=times,
                units="",
            ),
        )

        self.assertEqual((11, 4, 3), symbols.values.shape)
        self.assertEqual(
            weather_symbol.clearsky | weather_symbol.night, symbols.values[0, 0, 0]
        )
        self.assertEqual(np.datetime64("2024-02-20 01:00:00", "h"), symbols.times[0])

    def test_multiDimensional_6h(self):
        shape = (12, 4, 3)
        times = np.arange("2024-02-20", "2024-02-20 12:00:00", dtype="datetime64[h]")
        self.assertEqual(12, len(times))

        symbols = weather_symbol.get_weather_symbol_6h(
            accumulated_precipitation=Data(
                "rain",
                values=np.zeros(shape, np.float32),
                times=times,
                units="",
            ),
            cloud_cover_in_percent=Data(
                "clouds",
                values=np.zeros(shape, np.float32),
                times=times,
                units="",
            ),
        )

        self.assertEqual((6, 4, 3), symbols.values.shape)
        self.assertEqual(
            weather_symbol.clearsky | weather_symbol.night, symbols.values[0, 0, 0]
        )
        self.assertEqual(np.datetime64("2024-02-20 06:00:00", "h"), symbols.times[0])

    def test_multiDimensional_12h(self):
        shape = (13, 4, 3)
        times = np.arange("2024-02-20", "2024-02-20 13:00:00", dtype="datetime64[h]")
        self.assertEqual(13, len(times))

        symbols = weather_symbol.get_weather_symbol_12h(
            accumulated_precipitation=Data(
                "rain",
                values=np.zeros(shape, np.float32),
                times=times,
                units="",
            ),
            cloud_cover_in_percent=Data(
                "clouds",
                values=np.zeros(shape, np.float32),
                times=times,
                units="",
            ),
        )

        self.assertEqual((1, 4, 3), symbols.values.shape)
        self.assertEqual(weather_symbol.clearsky, symbols.values[0, 0, 0])
        self.assertEqual(np.datetime64("2024-02-20 12:00:00", "h"), symbols.times[0])

    def test_sun_state_1h(self):
        shape = (24, 1, 1)
        symbols = Data(
            "weather_symbol_1h",
            np.full(shape, 1, np.int16),
            np.arange("2024-02-20", "2024-02-21", dtype="datetime64[h]"),
            "1",
        )
        weather_symbol._update_sun_state_in(symbols, dt.timedelta(hours=1))

        values = symbols.values.flatten()
        self.assertEqual(1 | weather_symbol.night, values[0])
        self.assertEqual(1 | weather_symbol.night, values[1])
        self.assertEqual(1 | weather_symbol.night, values[2])
        self.assertEqual(1 | weather_symbol.night, values[3])
        self.assertEqual(1 | weather_symbol.night, values[4])
        self.assertEqual(1, values[5])
        self.assertEqual(1, values[6])
        self.assertEqual(1, values[7])
        self.assertEqual(1, values[8])
        self.assertEqual(1, values[9])
        self.assertEqual(1, values[10])
        self.assertEqual(1, values[11])
        self.assertEqual(1, values[12])
        self.assertEqual(1, values[13])
        self.assertEqual(1, values[14])
        self.assertEqual(1, values[15])
        self.assertEqual(1, values[16])
        self.assertEqual(1 | weather_symbol.night, values[17])
        self.assertEqual(1 | weather_symbol.night, values[18])
        self.assertEqual(1 | weather_symbol.night, values[19])
        self.assertEqual(1 | weather_symbol.night, values[20])
        self.assertEqual(1 | weather_symbol.night, values[21])
        self.assertEqual(1 | weather_symbol.night, values[22])
        self.assertEqual(1 | weather_symbol.night, values[23])

    def test_sun_state_6h(self):
        shape = (24, 1, 1)
        symbols = Data(
            "weather_symbol_6h",
            np.full(shape, 1, np.int16),
            np.arange("2024-02-20", "2024-02-21", dtype="datetime64[h]"),
            "1",
        )
        weather_symbol._update_sun_state_in(symbols, dt.timedelta(hours=6))

        values = symbols.values.flatten()
        self.assertEqual(1 | weather_symbol.night, values[0])
        self.assertEqual(1 | weather_symbol.night, values[1])
        self.assertEqual(1 | weather_symbol.night, values[2])
        self.assertEqual(1 | weather_symbol.night, values[3])
        self.assertEqual(1 | weather_symbol.night, values[4])
        self.assertEqual(1 | weather_symbol.night, values[5])
        self.assertEqual(1 | weather_symbol.night, values[6])
        self.assertEqual(1, values[7])
        self.assertEqual(1, values[8])
        self.assertEqual(1, values[9])
        self.assertEqual(1, values[10])
        self.assertEqual(1, values[11])
        self.assertEqual(1, values[12])
        self.assertEqual(1, values[13])
        self.assertEqual(1, values[14])
        self.assertEqual(1, values[15])
        self.assertEqual(1, values[16])
        self.assertEqual(1, values[17])
        self.assertEqual(1, values[18])
        self.assertEqual(1 | weather_symbol.night, values[19])
        self.assertEqual(1 | weather_symbol.night, values[20])
        self.assertEqual(1 | weather_symbol.night, values[21])
        self.assertEqual(1 | weather_symbol.night, values[22])
        self.assertEqual(1 | weather_symbol.night, values[23])

    def test_sun_state_12h(self):
        shape = (24, 1, 1)
        symbols = Data(
            "weather_symbol_12h",
            np.full(shape, 1, np.int16),
            np.arange("2024-02-20", "2024-02-21", dtype="datetime64[h]"),
            "1",
        )
        weather_symbol._update_sun_state_in(symbols, dt.timedelta(hours=12))

        values = symbols.values.flatten()
        self.assertEqual(1 | weather_symbol.night, values[0])
        self.assertEqual(1 | weather_symbol.night, values[1])
        self.assertEqual(1 | weather_symbol.night, values[2])
        self.assertEqual(1 | weather_symbol.night, values[3])
        self.assertEqual(1 | weather_symbol.night, values[4])
        self.assertEqual(1 | weather_symbol.night, values[5])
        self.assertEqual(1 | weather_symbol.night, values[6])
        self.assertEqual(1 | weather_symbol.night, values[7])
        self.assertEqual(1 | weather_symbol.night, values[8])
        self.assertEqual(1 | weather_symbol.night, values[9])
        self.assertEqual(1, values[10])
        self.assertEqual(1, values[11])
        self.assertEqual(1, values[12])
        self.assertEqual(1, values[13])
        self.assertEqual(1, values[14])
        self.assertEqual(1, values[15])
        self.assertEqual(1, values[16])
        self.assertEqual(1, values[17])
        self.assertEqual(1, values[18])
        self.assertEqual(1, values[19])
        self.assertEqual(1, values[20])
        self.assertEqual(1, values[21])
        self.assertEqual(1 | weather_symbol.night, values[22])
        self.assertEqual(1 | weather_symbol.night, values[23])


if __name__ == "__main__":
    unittest.main()
