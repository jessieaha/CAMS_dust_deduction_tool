"""Combined hourly/daily tables and the hourly dust-threshold sweep."""

import os
import tempfile
import unittest

import pandas as pd

import cams_data_to_eea_daily as cams
import dust_threshold_sensitivity as sens


def _product(resolution_days, cams_values):
    rows = []
    for day, dust in zip(resolution_days, cams_values):
        rows.append(
            {
                "Samplingpoint": "ES/STA1",
                "day": pd.Timestamp(day),
                "daily_mean": 40.0,
                "cams_dust": dust,
                "Latitude": 40.0,
                "Longitude": -3.0,
                "Altitude": 600.0,
                "Exceedance": False,
                "dust_flag": dust > 5,
                "pollutant_median": 10.0,
                "Dust_contribution": 0.0,
                "corrected_pollutant": 40.0,
            }
        )
    return pd.DataFrame(rows)


class CombineProductsTest(unittest.TestCase):
    def test_stack_keeps_both_clocks_for_the_same_label(self):
        hourly = _product(["2024-06-15", "2024-06-16"], [2.0, 8.0])
        daily = _product(["2024-06-15", "2024-06-16"], [3.0, 9.0])
        combined = cams.combine_hourly_and_daily(hourly, daily)
        self.assertEqual(len(combined), 4)
        self.assertEqual(set(combined["eea_resolution"]), {"hour", "day"})
        self.assertEqual(
            set(combined["day_definition"]),
            {"UTC day", "station local day"},
        )
        june15 = combined[combined["day"] == pd.Timestamp("2024-06-15")]
        self.assertEqual(len(june15), 2)
        self.assertEqual(
            set(june15["Samplingpoint"]),
            {"ES/STA1 (UTC day)", "ES/STA1 (station local day)"},
        )
        self.assertTrue((combined["Samplingpoint_id"] == "ES/STA1").all())

    def test_summary_counts_each_clock_on_its_own_row(self):
        hourly = _product(["2024-06-15"], [8.0])
        hourly["daily_mean"] = 80.0
        hourly["Exceedance"] = True
        hourly["corrected_pollutant"] = 20.0
        daily = _product(["2024-06-15"], [1.0])
        daily["daily_mean"] = 80.0
        daily["Exceedance"] = True
        daily["corrected_pollutant"] = 80.0
        combined = cams.combine_hourly_and_daily(hourly, daily)
        table = cams.station_summary_table(combined, 50.0)
        self.assertEqual(len(table), 2)
        by_clock = table.set_index("day_definition")
        self.assertEqual(by_clock.loc["UTC day", "Dust_exceedance_days"], 1)
        self.assertEqual(by_clock.loc["station local day", "Dust_exceedance_days"], 0)

    def test_loader_reads_both_parquet_files(self):
        hourly = _product(["2024-06-15"], [2.0])
        daily = _product(["2024-06-15"], [4.0])
        with tempfile.TemporaryDirectory() as folder:
            hour_path = os.path.join(folder, "hour.parquet")
            day_path = os.path.join(folder, "day.parquet")
            hourly.to_parquet(hour_path)
            daily.to_parquet(day_path)
            combined = cams.load_and_combine_deduction_parquets(hour_path, day_path)
        self.assertEqual(len(combined), 2)


def _hourly_case():
    """One station. Clean days sit beside three exceedances with rising dust."""
    days = pd.date_range("2024-03-01", periods=8, freq="D")
    means = [20, 20, 20, 20, 80, 90, 70, 15]
    dust = [1, 1, 1, 1, 4, 8, 12, 2]
    return pd.DataFrame(
        {
            "Samplingpoint": ["ES/STA1"] * 8,
            "day": days,
            "daily_mean": means,
            "cams_dust": dust,
            "Latitude": 40.0,
            "Longitude": -3.0,
            "Altitude": 600.0,
        }
    )


class ThresholdSensitivityTest(unittest.TestCase):
    def test_higher_threshold_flags_fewer_dust_exceedances(self):
        frames = sens.threshold_sweep(
            _hourly_case(),
            [3, 5, 10],
            pollutant="PM10",
            neighbor_n=2,
        )
        summary = sens.exceedance_summary(frames).set_index("dust_threshold_ug_m3")
        self.assertEqual(int(summary.loc[3, "exceedance_days"]), 3)
        self.assertGreater(
            int(summary.loc[3, "dust_exceedance_days"]),
            int(summary.loc[10, "dust_exceedance_days"]),
        )
        self.assertGreaterEqual(
            int(summary.loc[5, "dust_exceedance_days"]),
            int(summary.loc[10, "dust_exceedance_days"]),
        )
        # cams 4, 8 and 12 on the three exceedance days: only 12 is above 10.
        self.assertEqual(int(summary.loc[10, "dust_days"]), 1)

    def test_plots_write_pngs_for_exceedance_days(self):
        frames = sens.threshold_sweep(
            _hourly_case(),
            [3, 5, 10],
            pollutant="PM10",
            neighbor_n=2,
        )
        with tempfile.TemporaryDirectory() as folder:
            paths = sens.write_sensitivity_plots(frames, folder, reference=5)
            summary = sens.exceedance_summary(frames)
            summary_path = os.path.join(folder, "threshold_summary.csv")
            summary.to_csv(summary_path, index=False)
            for path in paths.values():
                self.assertTrue(os.path.isfile(path), path)
                self.assertGreater(os.path.getsize(path), 1000, path)
            self.assertTrue(os.path.isfile(summary_path))
            self.assertGreater(os.path.getsize(summary_path), 0)


if __name__ == "__main__":
    unittest.main()
