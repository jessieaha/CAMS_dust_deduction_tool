"""Unit checks for local-day CAMS alignment and chunked station extraction.

These tests use synthetic grids only. They do not call the EEA download API
or the Copernicus ADS (that download needs an API key).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cams_data_to_eea_daily as daily


class TimezoneOffsetTests(unittest.TestCase):
    def test_labels_from_eea_metadata(self):
        self.assertEqual(daily.parse_tz_offset("UTC"), 0)
        self.assertEqual(daily.parse_tz_offset("UTC+01"), 1)
        self.assertEqual(daily.parse_tz_offset("UTC+02"), 2)
        self.assertEqual(daily.parse_tz_offset("UTC+01:00"), 1)
        self.assertEqual(daily.parse_tz_offset("UTC-04"), -4)
        self.assertEqual(daily.parse_tz_offset("UTC-03"), -3)
        self.assertEqual(daily.parse_tz_offset(None), 0)
        self.assertEqual(daily.parse_tz_offset(np.nan), 0)

    def test_daily_stamp_is_not_shifted_like_hourly_utc_plus_1(self):
        # Hourly path: naive stamp is fixed UTC+1, so local midnight becomes 23:00 the day before.
        shifted = daily.eea_to_utc(pd.Series(["2024-06-15 00:00:00"]))
        self.assertEqual(shifted.iloc[0], pd.Timestamp("2024-06-14 23:00:00", tz="UTC"))
        # Daily path: the same stamp is already local midnight and keeps its calendar date.
        kept = daily.naive_calendar_day(pd.Series(["2024-06-15 00:00:00"]))
        self.assertEqual(kept.iloc[0], pd.Timestamp("2024-06-15"))


class LocalDayWindowTests(unittest.TestCase):
    def _hourly(self, start, periods, values):
        times = pd.date_range(start, periods=periods, freq="h")
        return pd.DataFrame({"time": times, "STA": np.asarray(values, dtype="float32")})

    def test_utc_plus_2_uses_previous_evening(self):
        # 48 hours starting 2024-01-14 00:00 UTC. Value is 10 only inside the
        # UTC+2 local day of 15 Jan, which is [14 Jan 22:00, 15 Jan 22:00) UTC.
        start = pd.Timestamp("2024-01-14 00:00:00")
        times = pd.date_range(start, periods=72, freq="h")
        window = (times >= "2024-01-14 22:00") & (times < "2024-01-15 22:00")
        values = np.where(window, 10.0, 0.0).astype("float32")
        hourly = pd.DataFrame({"time": times, "UTC2": values, "UTC0": values})
        offsets = pd.Series({"UTC2": 2, "UTC0": 0})
        out = daily.aggregate_local_daily(hourly, offsets, min_hours=18)
        day = pd.Timestamp("2024-01-15")
        utc2 = out[(out["Samplingpoint"] == "UTC2") & (out["day"] == day)]
        utc0 = out[(out["Samplingpoint"] == "UTC0") & (out["day"] == day)]
        self.assertEqual(len(utc2), 1)
        self.assertAlmostEqual(float(utc2["cams_dust"].iloc[0]), 10.0, places=5)
        self.assertEqual(int(utc2["n_cams_hours"].iloc[0]), 24)
        # UTC day 15 Jan includes 22 hours of 10 and the 22:00/23:00 hours of 0.
        self.assertAlmostEqual(float(utc0["cams_dust"].iloc[0]), 220.0 / 24.0, places=4)

    def test_negative_offset_shifts_the_window_forward(self):
        times = pd.date_range("2024-01-15 00:00", periods=48, freq="h")
        # UTC-4 local day 15 Jan is [15 Jan 04:00, 16 Jan 04:00) UTC.
        window = (times >= "2024-01-15 04:00") & (times < "2024-01-16 04:00")
        values = np.where(window, 4.0, 1.0).astype("float32")
        hourly = pd.DataFrame({"time": times, "WEST": values})
        out = daily.aggregate_local_daily(hourly, pd.Series({"WEST": -4}), min_hours=18)
        row = out[(out["Samplingpoint"] == "WEST") & (out["day"] == pd.Timestamp("2024-01-15"))]
        self.assertAlmostEqual(float(row["cams_dust"].iloc[0]), 4.0, places=5)

    def test_short_local_day_is_omitted(self):
        times = pd.date_range("2024-01-15 00:00", periods=10, freq="h")
        hourly = pd.DataFrame({"time": times, "STA": np.ones(10, dtype="float32")})
        out = daily.aggregate_local_daily(hourly, pd.Series({"STA": 0}), min_hours=18)
        self.assertTrue(out.empty)

    def test_three_hourly_uses_the_same_fraction(self):
        self.assertEqual(daily.minimum_samples_for_day(3.0, 18), 6)
        times = pd.date_range("2024-01-15 00:00", periods=8, freq="3h")
        hourly = pd.DataFrame({"time": times, "STA": np.full(8, 2.0, dtype="float32")})
        out = daily.aggregate_local_daily(hourly, pd.Series({"STA": 0}), min_hours=18)
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out["n_cams_hours"].iloc[0]), 8)


class GriddedAlignmentTests(unittest.TestCase):
    def _dataset(self):
        times = pd.date_range("2024-01-14", periods=72, freq="h")
        lat = np.array([40.0, 42.0])
        lon = np.array([0.0, 2.0])
        window = (times >= "2024-01-14 22:00") & (times < "2024-01-15 22:00")
        series = np.where(window, 10.0, 0.0).astype("float32")
        dust = np.broadcast_to(series[:, None, None], (len(times), len(lat), len(lon))).copy()
        return xr.Dataset(
            {"dust": (("time", "lat", "lon"), dust)},
            coords={"time": times, "lat": lat, "lon": lon},
        )

    def test_two_stations_match_their_own_local_day(self):
        obs = pd.DataFrame(
            {
                "Samplingpoint": ["ES/A", "GR/B", "FAR"],
                "day": pd.to_datetime(["2024-01-15", "2024-01-15", "2024-01-15"]),
                "daily_mean": np.array([30, 40, 50], dtype="float32"),
                "Latitude": [40.4, 41.2, 70.0],
                "Longitude": [0.4, 1.2, 0.5],
                "Timezone": ["UTC", "UTC+02", "UTC"],
                "Pollutant": ["PM10", "PM10", "PM10"],
            }
        )
        out = daily.add_cams_local_daily_dust(self._dataset(), obs, min_hours=18, time_col="day")
        self.assertNotIn("FAR", set(out["Samplingpoint"]))
        utc = out.loc[out["Samplingpoint"] == "ES/A", "cams_dust"].iloc[0]
        plus2 = out.loc[out["Samplingpoint"] == "GR/B", "cams_dust"].iloc[0]
        self.assertAlmostEqual(float(plus2), 10.0, places=4)
        self.assertAlmostEqual(float(utc), 220.0 / 24.0, places=4)
        # The EEA local date itself is unchanged by the join.
        self.assertEqual(pd.Timestamp(out.loc[out["Samplingpoint"] == "GR/B", "day"].iloc[0]), pd.Timestamp("2024-01-15"))

    def test_missing_local_day_is_nan_not_nearest(self):
        times = pd.date_range("2024-01-16", periods=24, freq="h")
        dust = np.ones((24, 2, 2), dtype="float32")
        ds = xr.Dataset(
            {"dust": (("time", "lat", "lon"), dust)},
            coords={"time": times, "lat": [40.0, 42.0], "lon": [0.0, 2.0]},
        )
        obs = pd.DataFrame(
            {
                "Samplingpoint": ["ES/A"],
                "day": pd.to_datetime(["2024-01-15"]),
                "Latitude": [41.0],
                "Longitude": [1.0],
                "tz_offset": [0],
            }
        )
        out = daily.add_cams_local_daily_dust(ds, obs, min_hours=18)
        self.assertTrue(np.isnan(out["cams_dust"].iloc[0]))


class ChunkedFileTests(unittest.TestCase):
    def test_open_file_loads_one_utc_day_at_a_time(self):
        times = pd.date_range("2024-03-01", periods=72, freq="h")
        lat = np.linspace(40.0, 42.0, 8)
        lon = np.linspace(0.0, 2.0, 8)
        dust = np.zeros((len(times), len(lat), len(lon)), dtype="float32")
        dust[:] = np.arange(len(times), dtype="float32")[:, None, None]
        ds = xr.Dataset(
            {"dust": (("time", "lat", "lon"), dust)},
            coords={"time": times, "lat": lat, "lon": lon},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cams.eaq.ira.ENSa.dust.l0.2024-03.nc")
            ds.to_netcdf(path)
            opened = xr.open_dataset(path, chunks={"time": 24})
            try:
                self.assertTrue(hasattr(opened["dust"].data, "chunks"))
            finally:
                opened.close()

            stations = pd.DataFrame(
                {
                    "Samplingpoint": ["ES/A"],
                    "Latitude": [41.0],
                    "Longitude": [1.0],
                    "tz_offset": [1],
                }
            )
            original = xr.DataArray.load

            def wrapped(array, **kwargs):
                if "time" in array.dims and array.sizes["time"] > 30:
                    raise AssertionError(
                        f"loaded {array.sizes['time']} time steps at once; expected one UTC day"
                    )
                return original(array, **kwargs)

            with mock.patch.object(xr.DataArray, "load", wrapped):
                out = daily.add_cams_local_daily_dust(
                    [path],
                    pd.DataFrame(
                        {
                            "Samplingpoint": ["ES/A", "OUT"],
                            "day": pd.to_datetime(["2024-03-02", "2024-03-02"]),
                            "Latitude": [41.0, 10.0],
                            "Longitude": [1.0, 1.0],
                            "tz_offset": [1, 0],
                        }
                    ),
                    cache_dir=tmp,
                    min_hours=18,
                    year=2024,
                )
            self.assertEqual(list(out["Samplingpoint"]), ["ES/A"])
            self.assertFalse(np.isnan(out["cams_dust"].iloc[0]))
            cache = [
                name
                for name in os.listdir(os.path.join(tmp, "IRA_dust"))
                if name.startswith("cams_dust_local_daily_stations_")
            ]
            self.assertEqual(len(cache), 1)

            def fail_if_called(*_args, **_kwargs):
                raise AssertionError("NetCDF was read again even though the daily cache exists")

            with mock.patch.object(daily, "extract_hourly_at_stations", fail_if_called):
                again = daily.add_cams_local_daily_dust(
                    [path],
                    pd.DataFrame(
                        {
                            "Samplingpoint": ["ES/A"],
                            "day": pd.to_datetime(["2024-03-02"]),
                            "Latitude": [41.0],
                            "Longitude": [1.0],
                            "tz_offset": [1],
                        }
                    ),
                    cache_dir=tmp,
                    reuse_cache=True,
                    recompute_daily=False,
                    min_hours=18,
                    year=2024,
                )
            self.assertAlmostEqual(float(again["cams_dust"].iloc[0]), float(out["cams_dust"].iloc[0]), places=5)
            del stations


class EeaTableTests(unittest.TestCase):
    def test_request_matches_the_notebook(self):
        body = daily.eea_request_body(["ES", "PT"], "PM2.5", "E2a", 2024, "day")
        self.assertEqual(body["dataset"], 1)
        self.assertEqual(body["pollutants"], ["PM2.5"])
        self.assertEqual(body["aggregationType"], "day")
        self.assertEqual(body["dateTimeStart"], "2023-12-14T00:00:00Z")
        self.assertEqual(body["dateTimeEnd"], "2024-12-31T23:59:59Z")
        self.assertEqual(body["countries"], ["ES", "PT"])
        verified = daily.eea_request_body(["ES"], "PM10", "E1a", 2025, "hour")
        self.assertEqual(verified["dataset"], 2)

    def test_daily_parquet_keeps_local_date(self):
        frame = pd.DataFrame(
            {
                "Value": [12.5, -1.0, 8.0],
                "Validity": [1, 1, 1],
                "Verification": [1, 1, 3],
                "AggType": ["day", "day", "day"],
                "Start": ["2024-06-15 00:00:00", "2024-06-15 00:00:00", "2024-06-16 00:00:00"],
                "Samplingpoint": ["ES/A", "ES/A", "ES/A"],
                "Unit": ["µg/m3", "µg/m3", "µg/m3"],
                "Pollutant": ["PM10", "PM10", "PM10"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            frame.to_parquet(os.path.join(tmp, "part.parquet"), index=False)
            loaded = daily.load_and_filter_eea_daily(tmp, "day")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(pd.Timestamp(loaded["day"].iloc[0]), pd.Timestamp("2024-06-15"))
        self.assertAlmostEqual(float(loaded["daily_mean"].iloc[0]), 12.5, places=5)

    def test_deduction_columns_match_the_notebook(self):
        days = pd.date_range("2024-01-01", periods=8, freq="D")
        dust = [0, 0, 0, 9, 0, 0, 0, 0]
        values = [10, 10, 10, 40, 10, 10, 12, 10]
        frame = pd.DataFrame(
            {
                "Samplingpoint": ["ES/A"] * 8,
                "day": days,
                "daily_mean": np.array(values, dtype="float32"),
                "cams_dust": np.array(dust, dtype="float32"),
                "Pollutant": ["PM10"] * 8,
                "Latitude": [40.0] * 8,
                "Longitude": [0.0] * 8,
                "Altitude": [10.0] * 8,
            }
        )
        result = daily.run_deduction_for_pollutant(
            frame, year=2024, pollutant="PM10", dust_threshold=5, neighbor_n=2
        )
        self.assertIn("corrected_pollutant", result.columns)
        self.assertIn("pollutant_median", result.columns)
        self.assertIn("Dust_contribution", result.columns)
        dust_row = result.loc[result["dust_flag"]].iloc[0]
        # Neighbours are the non-dust values 10, 10 before and 10, 10 after. Median 10.
        # 40 is below the 50 µg/m3 exceedance limit, so the notebook baseline
        # (dust AND exceedance) leaves this day without a median.
        self.assertFalse(bool(dust_row["Exceedance"]))
        self.assertTrue(np.isnan(dust_row["pollutant_median"]))

        frame.loc[frame["cams_dust"] > 5, "daily_mean"] = np.float32(80)
        result = daily.run_deduction_for_pollutant(
            frame, year=2024, pollutant="PM10", dust_threshold=5, neighbor_n=2
        )
        dust_row = result.loc[result["dust_flag"]].iloc[0]
        self.assertAlmostEqual(float(dust_row["pollutant_median"]), 10.0, places=4)
        self.assertAlmostEqual(float(dust_row["corrected_pollutant"]), 10.0, places=4)
        self.assertAlmostEqual(float(dust_row["Dust_contribution"]), 70.0, places=4)


class FileSelectionTests(unittest.TestCase):
    def test_december_of_previous_year_is_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            ira = os.path.join(tmp, "IRA_dust")
            os.makedirs(ira)
            kept = [
                "cams.eaq.ira.ENSa.dust.l0.2023-12.nc",
                "cams.eaq.ira.ENSa.dust.l0.2024-01.nc",
            ]
            skipped = "cams.eaq.ira.ENSa.dust.l0.2023-06.nc"
            for name in kept + [skipped]:
                open(os.path.join(ira, name), "wb").close()
            found = daily.list_cams_netcdf(tmp, tmp, "dust", 2024)
        names = {os.path.basename(path) for path in found}
        self.assertEqual(names, set(kept))

    def test_cams_download_plan_matches_the_notebook(self):
        plan = daily.cams_download_plan(2024)
        self.assertEqual(plan[0]["month"], ["12"])
        self.assertEqual(plan[0]["year"], ["2023"])
        self.assertEqual([item["filename"] for item in plan], [
            "CAMS_IRA_2023_12.zip",
            "CAMS_IRA_2024_q1.zip",
            "CAMS_IRA_2024_q2.zip",
            "CAMS_IRA_2024_q3.zip",
            "CAMS_IRA_2024_q4.zip",
        ])


if __name__ == "__main__":
    unittest.main()
