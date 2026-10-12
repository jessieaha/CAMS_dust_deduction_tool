"""Flag rules and CAMS zip storage. No network and no Copernicus key."""

import os
import tempfile
import unittest
import zipfile

import cams_data_to_eea_daily as cams


def _tiny_zip(path, member="hello.txt", text="ok"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, text)


class ResolveRunFlagsTest(unittest.TestCase):
    def test_download_cams_forces_compute_and_skips_saved_parquet(self):
        flags = cams.resolve_run_flags(
            download_eea=False,
            download_cams=True,
            compute_cams_daily=False,
            load_parquet=True,
            eea_temporal_flag="day",
        )
        self.assertTrue(flags["compute_cams_daily"])
        self.assertFalse(flags["load_parquet"])
        self.assertTrue(flags["local_day"])
        self.assertEqual(len(flags["notes"]), 2)

    def test_download_eea_does_not_rebuild_cams(self):
        flags = cams.resolve_run_flags(
            download_eea=True,
            download_cams=False,
            compute_cams_daily=False,
            load_parquet=True,
            eea_temporal_flag="hour",
        )
        self.assertFalse(flags["compute_cams_daily"])
        self.assertFalse(flags["load_parquet"])
        self.assertFalse(flags["local_day"])

    def test_quiet_reload_keeps_the_saved_parquet(self):
        flags = cams.resolve_run_flags(
            download_eea=False,
            download_cams=False,
            compute_cams_daily=False,
            load_parquet=True,
            eea_temporal_flag="hour",
        )
        self.assertTrue(flags["load_parquet"])
        self.assertEqual(flags["notes"], [])

    def test_unknown_resolution_is_rejected(self):
        with self.assertRaises(ValueError):
            cams.resolve_run_flags(
                download_eea=False,
                download_cams=False,
                compute_cams_daily=False,
                load_parquet=False,
                eea_temporal_flag="month",
            )


class StoredCamsZipTest(unittest.TestCase):
    def test_local_zip_is_copied_into_the_output_folder(self):
        with tempfile.TemporaryDirectory() as persistent, tempfile.TemporaryDirectory() as project:
            filename = "CAMS_IRA_2023_12.zip"
            local_dir = os.path.join(project, "IRA_dust")
            os.makedirs(local_dir)
            _tiny_zip(os.path.join(local_dir, filename))
            open(os.path.join(local_dir, "cams.nc"), "w").close()

            stored = cams.stored_cams_zip(persistent, project, "dust", filename)

            self.assertEqual(
                os.path.abspath(stored),
                os.path.abspath(os.path.join(persistent, "IRA_dust", filename)),
            )
            self.assertTrue(os.path.isfile(stored))
            self.assertFalse(os.path.exists(os.path.join(persistent, "IRA_dust", "cams.nc")))

    def test_existing_output_zip_is_left_in_place(self):
        with tempfile.TemporaryDirectory() as persistent, tempfile.TemporaryDirectory() as project:
            filename = "CAMS_IRA_2024_q1.zip"
            stored_path = os.path.join(persistent, "IRA_dust", filename)
            os.makedirs(os.path.dirname(stored_path))
            _tiny_zip(stored_path, text="drive")
            local_dir = os.path.join(project, "IRA_dust")
            os.makedirs(local_dir)
            _tiny_zip(os.path.join(local_dir, filename), text="local")

            cams.stored_cams_zip(persistent, project, "dust", filename)

            with zipfile.ZipFile(stored_path) as archive:
                self.assertEqual(archive.read("hello.txt"), b"drive")


class DownloadAndExtractTest(unittest.TestCase):
    def test_extract_writes_netcdf_locally_and_keeps_the_zip(self):
        with tempfile.TemporaryDirectory() as persistent, tempfile.TemporaryDirectory() as project:
            filename = "CAMS_IRA_2023_12.zip"
            member = "cams.eaq.ira.ENSa.dust.l0.2023-12.nc"
            zip_dir = os.path.join(persistent, "IRA_dust")
            os.makedirs(zip_dir)
            _tiny_zip(os.path.join(zip_dir, filename), member=member, text="grid")

            cams.download_and_extract_cams(
                project, persistent, 2024, "dust", download=False, extract=True
            )

            extracted = os.path.join(project, "IRA_dust", member)
            self.assertTrue(os.path.isfile(extracted))
            self.assertTrue(os.path.isfile(os.path.join(zip_dir, filename)))
            self.assertFalse(os.path.exists(os.path.join(persistent, "IRA_dust", member)))

    def test_without_extract_the_zip_stays_packed(self):
        with tempfile.TemporaryDirectory() as persistent, tempfile.TemporaryDirectory() as project:
            filename = "CAMS_IRA_2023_12.zip"
            local_dir = os.path.join(project, "IRA_dust")
            os.makedirs(local_dir)
            _tiny_zip(os.path.join(local_dir, filename), member="only-in-zip.nc")

            cams.download_and_extract_cams(
                project, persistent, 2024, "dust", download=False, extract=False
            )

            self.assertTrue(os.path.isfile(os.path.join(persistent, "IRA_dust", filename)))
            self.assertFalse(os.path.exists(os.path.join(project, "IRA_dust", "only-in-zip.nc")))


if __name__ == "__main__":
    unittest.main()
