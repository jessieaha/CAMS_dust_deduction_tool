# Plan: align `Dust_discount_hourly.py` with the notebook

Goal: `Dust_discount_hourly.py` reads user settings and writes one daily parquet of natural-dust deduction results. It stays a batch script. It does not draw maps or timeseries, and it does not write the annual station CSV (that table is built in the notebook from the parquet).

Source of truth: `Dust_deduction_tool_full.ipynb`, hourly path only (`EEA_temporal_flag = 'hour'`). Daily EEA preprocessing stays in `cams_data_to_eea_daily.py`.

## What the script must produce

One Snappy parquet, same columns and rules as the notebook save cell:

`{project_dir}/CAMS_dust_{POLLUTANT}_deduction_{dataset}_hour_{YEAR}_MA{Basline_MA_days}.parquet`

Rows are station-days for the target year (plus the previous December is used only to build the baseline, then dropped before save, matching `df_target_year`).

Columns written by the notebook before save:

- identifiers and metadata already on the frame: `Pollutant`, `day`, `Samplingpoint`, `daily_mean`, `n_hours`, `unit`, `Latitude`, `Longitude`, `Altitude`
- `cams_dust`, `dust_flag`, `Exceedance`
- `pollutant_median`, `Dust_contribution`, `corrected_pollutant`

`LOAD_PARQUET_DATA` is a notebook reload switch. The script always computes and overwrites this parquet. It does not grow a reload branch.

## User inputs

Replace the hardcoded 2024 / PM10 header with the notebook settings that affect the hourly parquet. Paths stay on the local filesystem (`project_dir`). Do not add Google Drive or Colab mounts.

| Input | Notebook default | Script today | Change |
| --- | --- | --- | --- |
| `project_dir` | `'.'` | TNO path | Keep as the only root; default `'.'` |
| `YEAR` | `2025` | `2024` | Use the variable everywhere `2024` is hardcoded |
| `dataset` | `'E1a'` | `'E1a'`, later overwritten by the CAMS dataset name | Keep the EEA code in its own variable so the output name cannot become `cams-europe-air-quality-reanalyses` |
| `POLLUTANT` | `'PM10'` | fixed `PM10` | `'PM10'` or `'PM2.5'` |
| `EEA_temporal_flag` | `'hour'` | `'hour'` | Keep `'hour'` only; exit with a clear message if it is not `'hour'` |
| `Countries` | full EEA code list | same list | Unchanged |
| `DOWNLOAD_EEA` | `False` | `False` | Unchanged meaning |
| `DOWNLOAD_CAMS` | `False` | `False` | Unchanged meaning |
| `COMPUTE_CAMS_DAILY` | `False` | `False` | Unchanged meaning |
| `CAMS_dust_threshold` | `5` | `5`, compared with `>=` | Compare with `>` |
| `POLLUTANT_daily_threshold` | `50` | `PM10_daily_threshold = 50` | Rename; `25` when the pollutant is PM2.5 |
| `Station_Temporal_coverage` | `65` | coverage call uses `75` and `reference_year=2024` | Pass both from the inputs |
| `Basline_MA_days` | `6` | `neighbor_n=15` inside the loop | Pass this value into the baseline |

Folder layout from the notebook:

- EEA parquet: `EEA_{POLLUTANT}/{YEAR}/{dataset}/hour/`
- CAMS zips and NetCDF: `{project_dir}/IRA_dust/`
- Metadata: `{project_dir}/Data/DataExtract.csv` (the file already in this repo). The script currently looks for `EEA_PM10/DataExtract.csv`.

## Processing steps to match

Work top to bottom. Drop the local copies of helpers that already live in `util.py` (`eea_hourly_to_utc`, `filter_daily_by_coverage`, `add_cams_daily_dust_by_station`, `compute_station_baseline`) and call those functions. That is what the notebook does, and it keeps the median rule in one place.

1. **EEA download** (only if `DOWNLOAD_EEA`). Copy the notebook request, not the script’s older one:
   - API `https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile/async`
   - dataset id: `E2a`/`UTD` → 1, `E1a` → 2, `Historical` → 3
   - pollutant from `POLLUTANT`, countries from `Countries`
   - `dateTimeStart` `{YEAR-1}-12-14T00:00:00Z` through `{YEAR}-12-31T23:59:59Z`
   - zip name `{dataset}_{POLLUTANT}_hour_{YEAR}.zip` inside the EEA folder
   - If `DOWNLOAD_EEA` is false, extract that zip when it is already on disk; otherwise expect parquet files in the EEA folder.

2. **CAMS download and daily mean** (download only if `DOWNLOAD_CAMS`). Same quarterly request as the notebook: December of `YEAR-1` plus four quarters, dataset `cams-europe-air-quality-reanalyses`, variable `dust`, ensemble, level `0`, type `interim_reanalysis`. Skip a zip that already exists. Extract only when the expected NetCDF for that quarter is missing.
   - Fix the path split: the script downloads to `wp-dust/IRA_dust/` but opens `{project_dir}/IRA_dust/`. Use `IRA_dust` for both.
   - If `COMPUTE_CAMS_DAILY`, resample to UTC daily means and write `IRA_dust/cams_dust_daily_mean.nc`. Otherwise open that file. Fail with the path if it is missing.

3. **Hourly observations to UTC daily means.** Read only the columns the notebook uses (`Value`, `Validity`, `Verification`, `AggType`, `Start`, `Samplingpoint`, `Unit`, `Pollutant`). Then:
   - `Validity == 1`, `Verification < 3`, `AggType == 'hour'`, `Value >= 0` (the script does not drop negatives today)
   - naive timestamps as CET / UTC+1, then UTC (`util.eea_hourly_to_utc`)
   - the script’s duplicate-hour drop on `(Pollutant, Start, Samplingpoint)` can stay; the notebook does not do it, and it only removes double-counted hours
   - daily mean, keep days with `n_hours == 24`
   - `util.filter_daily_by_coverage(..., reference_year=YEAR, min_pct=Station_Temporal_coverage)`
   - set `Exceedance` from `POLLUTANT_daily_threshold` on this daily frame, before the baseline

4. **Metadata.** Build `Samplingpoint` as the EoI country prefix plus sampling point id. Left-join `Longitude`, `Latitude`, `Altitude`. Stations with no coordinates cannot be interpolated; drop them before the CAMS step and print how many were dropped.

5. **CAMS at stations.** Drop stations outside the CAMS lat/lon domain, then `util.add_cams_daily_dust_by_station` with linear interpolation. Do not plot the preview; a short print of station counts and `head` is enough.

6. **Dust flag and baseline.** `dust_flag = cams_dust > CAMS_dust_threshold`. Restrict the saved frame to `YEAR`, with the notebook fallback that uses the only available year if that slice is empty. For each station call `util.compute_station_baseline` with `neighbor_n=Basline_MA_days` and store `pollutant_median`.
   - The util function fills the median only on days that are both dust-flagged and exceedances. The script currently comments out the exceedance half and uses every dust day. Follow util / the notebook.

7. **Contribution and corrected concentration.** Match the notebook order:
   - on dust days, `Dust_contribution = daily_mean - pollutant_median`, else `0.0`
   - on dust days, `corrected_pollutant = pollutant_median`, else `daily_mean`
   - then clip `Dust_contribution` at 0
   - The script clips first and then sets `corrected_PM10 = daily_mean - Dust_contribution`, and it stores `NaN` instead of `0` on non-dust days. That disagrees with the notebook after the clip, and the column names are PM10-specific.

8. **Write the parquet** with the filename above. Print the path and row count. Remove unused plotting imports (`cartopy`, `LogNorm`).

## Out of scope

- Plotly, cartopy, or matplotlib figures
- Annual station CSV and exceedance-day counts (notebook sections 6 and 7)
- Google Drive
- The daily-EEA preprocessor
- The commented CAMS-only dust scaling formula in the notebook

## Check after the edit

Run the script with `DOWNLOAD_EEA` and `DOWNLOAD_CAMS` left `False` against a small fixture (a few stations, a few days, a tiny daily CAMS file) and confirm:

- output path includes pollutant, dataset, `hour`, year, and `MA{Basline_MA_days}`
- `dust_flag` is true only when `cams_dust` is strictly above the threshold
- median, contribution, and corrected values follow one hand-computed dust-and-exceedance day and one non-dust day (`Dust_contribution == 0`, corrected equals the observation)
- a negative `daily_mean - median` is stored as `Dust_contribution == 0` while `corrected_pollutant` stays the median
- no figure files are created
