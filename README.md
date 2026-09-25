# CAMS Natural Dust Service Tool

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jessieaha/CAMS_dust_deduction_tool/blob/main/Dust_deduction_tool_full.ipynb)

[`Dust_deduction_tool_full.ipynb`](Dust_deduction_tool_full.ipynb) is the **CAMS Natural Dust service Tool**. It supports EU Member States in identifying and assessing natural contributions from dust to PM<sub>10</sub> and PM<sub>2.5</sub> concentrations, as requested by Article 16 of the Air Quality Directive (AAQD 2024/2881). The tool calculates and subtracts these natural dust contributions from measured concentrations to provide corrected surface concentrations.

> AAQD (2024/2881) Article 16: Member states are requested to identify a) zones where exceedances of limit values for a given pollutant are attributable to natural sources and b) average exposure territorial units, where exceedances of the level determined by the average exposure reduction obligations are attributable to natural sources.

The method follows the European Commission DG-ENV guidelines ([sec 2011-0208](https://data.consilium.europa.eu/doc/document/ST%206771%202011%20INIT/EN/pdf)) for assessing natural dust contributions before they can be subtracted from reported PM concentrations. Days affected by natural dust are flagged with the CAMS interim reanalysis dust product.

Created by Jessie Zhang, TNO, 22/05/2026.

## Outline

The notebook follows these steps:

1. [Import packages](#1-import-packages)
2. [Country selection and threshold settings](#2-country-selection-and-threshold-settings)
3. [Data preparation](#3-data-preparation)
4. [Flag dust days with CAMS data](#4-flag-dust-days-with-cams-data)
5. [Calculate dust for days with dust and exceedance](#5-calculate-dust-for-days-with-dust-and-exceedance)
6. [Table with results](#6-table-with-results)
7. [Visualization](#7-visualization)

## Repository files

| File | Role |
| --- | --- |
| `Dust_deduction_tool_full.ipynb` | Full interactive workflow: download, processing, deduction, tables, and plots |
| `util.py` | Shared functions: coverage filters, CAMS interpolation, baseline median, and plots |
| `cams_data_download.py` | Standalone download of CAMS interim reanalysis dust from the Atmosphere Data Store |
| `cams_data_to_eea_daily.py` | Preprocessor for **daily** EEA data: interpolates CAMS dust onto stations in local time |
| `Dust_discount_hourly.py` | Batch script for the **hourly** aggregation path (parquet of dust contribution and corrected PM) |
| `Data/DataExtract.csv` | Station metadata (coordinates, altitude, timezone) used to join observations |
| `storage_flowchart.py` | Diagram of the download, storage, and processing flow |

## How to run

Open `Dust_deduction_tool_full.ipynb` in Google Colab (badge above) or locally, and run the cells from top to bottom.

After the import cell, the last line should print that `util.py` has been imported. In Colab the notebook downloads `util.py` from this repository. Locally, keep `util.py` in the working directory. If a function in `util.py` was edited, reload it:

```python
import sys, importlib
import util
importlib.invalidate_caches()
if 'util' in sys.modules:
    del sys.modules['util']
import util
importlib.reload(util)
```

Interactive Plotly and ipywidgets maps need a notebook frontend. In VS Code or on HPC those widgets may not render; use the static timeseries and static map cells at the end of the notebook instead.

Set `LOAD_PARQUET_DATA = True` only after a full run has already written the deduction parquet. That flag reloads the saved daily result and skips recomputation.

## 1. Import packages

Python packages used by the notebook include `numpy`, `pandas`, `matplotlib`, `xarray`, `netCDF4`, `cartopy`, `plotly`, `requests`, `pyarrow`, and `cdsapi` (when downloading CAMS). Install a missing package with `pip install <package>`. For the Copernicus API client use `cdsapi>=0.7.7`.

## 2. Country selection and threshold settings

Edit the configuration cell before downloading or processing.

| Setting | Default in the notebook | Meaning |
| --- | --- | --- |
| `YEAR` | `2025` | Target year |
| `EEA_temporal_flag` | `'hour'` | `'hour'` or `'day'` |
| `dataset` | `'E1a'` | EEA dataset: `E2a` / `UTD`, `E1a`, or `Historical` |
| `POLLUTANT` | `'PM10'` | `'PM10'` or `'PM2.5'` |
| `Countries` | EEA country codes listed below | Countries requested from the EEA download API |
| `DOWNLOAD_EEA` | `False` | Download a new EEA parquet zip when `True` |
| `DOWNLOAD_CAMS` | `False` | Download CAMS interim reanalysis when `True` |
| `COMPUTE_CAMS_DAILY` | `False` | Build daily-mean CAMS files from the NetCDF archives |
| `USE_GOOGLE_DRIVE` | `True` | In Colab, read and write under `MyDrive/CAMS_Tool_output` |
| `LOAD_PARQUET_DATA` | `False` | Reload a previously saved deduction parquet |

EEA dataset codes used by the download API:

1. Unverified data transmitted continuously (UTD / E2a), from the beginning of 2023.
2. Verified data (E1a) from 2013, reported by countries by 30 September each year for the previous year.
3. Historical Airbase data delivered between 2002 and 2012, before Air Quality Directive 2008/50/EC entered into force.

The prototype ships a demonstration path based on a pre-downloaded daily EEA dataset for Spain (`ES`) and a smaller CAMS extract at Spanish station locations. With `DOWNLOAD_EEA` and `DOWNLOAD_CAMS` enabled, the same notebook can request other countries and years. You can also point the workflow at your own observation files.

Country codes:

| Code | Country | Code | Country |
| --- | --- | --- | --- |
| AD | Andorra | IT | Italy |
| AL | Albania | LT | Lithuania |
| AT | Austria | LU | Luxembourg |
| BA | Bosnia and Herzegovina | LV | Latvia |
| BE | Belgium | ME | Montenegro |
| BG | Bulgaria | MK | North Macedonia |
| CH | Switzerland | MT | Malta |
| CY | Cyprus | NL | Netherlands |
| CZ | Czechia | NO | Norway |
| DE | Germany | PL | Poland |
| DK | Denmark | PT | Portugal |
| EE | Estonia | RO | Romania |
| ES | Spain | RS | Serbia |
| FI | Finland | SE | Sweden |
| FR | France | SI | Slovenia |
| GB | United Kingdom | SK | Slovakia |
| GR | Greece | XK | Kosovo |
| HR | Croatia | IE | Ireland |
| HU | Hungary | IS | Iceland |

Thresholds are editable so later guideline or limit-value changes can be tested. Keep the dust-flag threshold unchanged when producing final results.

| Setting | Default | Meaning |
| --- | --- | --- |
| `CAMS_dust_threshold` | `5` µg/m³ | Total CAMS dust above this value flags a dust day |
| `POLLUTANT_daily_threshold` | `50` µg/m³ | Daily limit value (`50` for PM10, `25` for PM2.5) |
| `Station_Temporal_coverage` | `65` % | Minimum annual data coverage for a station to be kept |
| `Basline_MA_days` | `6` | Non-dust neighbour days before and after a dust day used for the background median |

## 3. Data preparation

### EEA observations

The notebook can download observations from the EEA Air Quality Download Service:

`https://eeadmz1-downloads-api-appservice.azurewebsites.net/` (`ParquetFile/async`)

The request covers `14 December (YEAR-1)` through `31 December YEAR` so the moving-window baseline has days before the target year. Files land in `EEA_{POLLUTANT}/{YEAR}/{dataset}/{hour|day}/`. With Google Drive enabled, the zip is stored under `CAMS_Tool_output` and extracted locally.

You can also download the same products from [https://eeadmz1-downloads-webapp.azurewebsites.net/](https://eeadmz1-downloads-webapp.azurewebsites.net/).

**Hourly path** (`EEA_temporal_flag = 'hour'`), applied when `LOAD_PARQUET_DATA` is `False`:

- Read parquet columns and keep `Validity == 1` (valid), `Verification < 3` (verified or preliminary verified), hourly aggregation, and non-negative values.
- Treat naive timestamps as CET (UTC+1) and convert them to UTC.
- Average each station-day and keep days that have all 24 hours.
- Keep stations whose coverage in `YEAR` is at least the coverage threshold. Rows outside that year are kept for those stations so the baseline window can extend into the previous December.
- Join `Data/DataExtract.csv` (or the copy on Google Drive) on sampling point to add longitude, latitude, altitude, and timezone.

Download a current metadata extract from [https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements](https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements) if `DataExtract.csv` is missing. The sampling-point key is the station EoI code country prefix plus the sampling point id.

Validity and verification labels used by the filter:

| Validity ID | Label | Kept |
| --- | --- | --- |
| -99 | Not valid due to station maintenance or calibration | No |
| -1 | Not valid | No |
| 1 | Valid | Yes |
| 2 | Valid, but below detection limit (measured value given) | No |
| 3 | Valid, but below detection limit (replaced by 0.5 × detection limit) | No |
| 4 | Valid (ozone only), CCQM.O3.2019 | No |

| Verification ID | Label | Kept |
| --- | --- | --- |
| 1 | Verified | Yes |
| 2 | Preliminary verified | Yes |
| 3 | Not verified | No |

**Daily path** (`EEA_temporal_flag = 'day'`): use preprocessed daily files. `cams_data_to_eea_daily.py` aggregates CAMS to the station-local day and writes a parquet that the notebook can read. Processing every EEA station in that preprocessor is memory- and CPU-heavy (on the order of several hours for the full network).

### CAMS interim reanalysis

Dust is taken from the ADS dataset `cams-europe-air-quality-reanalyses` (variable `dust`, ensemble model, level `0`, type `interim_reanalysis`). The notebook requests December of the previous year plus four quarterly files for `YEAR`, stored under `IRA_dust/` (or `CAMS_Tool_output/IRA_dust/` on Drive).

API access: [https://cds.climate.copernicus.eu/how-to-api](https://cds.climate.copernicus.eu/how-to-api). The CAMS regional reanalysis is retrieved through the Atmosphere Data Store. Put the key in `~/.cdsapirc`:

```text
url: https://ads.atmosphere.copernicus.eu/api
key: <your-api-key>
```

`cams_data_download.py` performs the same quarterly download outside the notebook.

With `COMPUTE_CAMS_DAILY = True`, hourly CAMS fields are averaged to UTC daily means. For hourly EEA data, both CAMS and the station series are on UTC days, then dust is interpolated to each station (`add_cams_daily_dust_by_station`, linear in space, nearest in time). Stations outside the CAMS latitude/longitude domain are dropped before interpolation.

Downloading and interpolating the full regional reanalysis can exceed the free memory on Google Colab. Run that step on a local machine or HPC, or load a preprocessed station extract. For daily EEA data, run the preprocessor and load the combined CAMS and EEA table.

## 4. Flag dust days with CAMS data

A day is a dust day when `cams_dust` is above `CAMS_dust_threshold`. A day is an exceedance when `daily_mean` is above `POLLUTANT_daily_threshold`. The notebook also reports any negative daily means.

## 5. Calculate dust for days with dust and exceedance

For each station, the background concentration is the median (50th percentile) of non-dust days: the `Basline_MA_days` non-dust neighbours before the day and the same number after it. The extra days downloaded from the previous December are there so early-January dust days still have a window.

On a day that is both dust-flagged and above the limit value:

- `Dust_contribution` = observed daily mean − background median
- `corrected_pollutant` = background median

On other days the dust contribution is zero and the corrected concentration stays the observation.

Negative dust contributions are clipped to zero. A moving-median background can sit above the concentration on a flagged dust day when strong winds lower local PM, so the raw difference can be negative. The notebook counts those days before clipping.

An alternative scaling that uses only CAMS dust is noted in the notebook and left commented out:

Corrected dust = (1 + (observed PM<sub>10</sub> − reanalysis PM<sub>10</sub>) / reanalysis PM<sub>10</sub>) × reanalysis dust

## Save and reload

When `LOAD_PARQUET_DATA` is `False`, the daily result is written as Snappy parquet:

`CAMS_dust_{POLLUTANT}_deduction_{dataset}_{hour|day}_{YEAR}_MA{Basline_MA_days}.parquet`

With Google Drive mounted, that file is under `/content/drive/MyDrive/CAMS_Tool_output/`. Set `LOAD_PARQUET_DATA = True` to read it back and continue from the annual statistics and plots.

`Dust_discount_hourly.py` writes the same kind of hourly-path parquet (dust contribution and corrected PM10) for later use in the notebook.

## 6. Table with results

Per station and year the notebook aggregates:

| Column | Meaning |
| --- | --- |
| `Exceedance_days` | Days with observed concentration above the limit value |
| `Dust_exceedance_days` | Exceedances that disappear after the natural-dust subtraction |
| `NonDust_exceedance` | Exceedances that remain after subtraction |
| `Average_pollutant` | Annual mean of the observed daily concentration |
| `Average_pollutant_dust_removed` | Annual mean after natural dust is removed |

The annual CSV is:

`{key}_station_annual_{POLLUTANT}_{dataset}_{hour|day}_{YEAR}_MA{Basline_MA_days}.csv`

`key` is `muti_countries` when more than one country is selected, otherwise the country code. The daily parquet keeps, for each station-day, the measured concentration, exceedance flag, CAMS dust, dust flag, background, dust contribution, corrected concentration, and whether the exceedance was due to natural dust.

## 7. Visualization

- Interactive map of corrected annual average concentration and of exceedance days. Hover a station to read the results.
- Clickable map: marker colour and size show the average concentration before dust removal; selecting a station plots measured and corrected PM for that year.
- Static timeseries (`plot_station_timeseries`) when widgets are unavailable. Set `station_name` from `station_table['Samplingpoint']`.
- Static maps (`plot_exceedance_maps_discrete`) for any numeric column in the station table. Colour maps follow the [Matplotlib colormap reference](https://matplotlib.org/stable/gallery/color/colormap_reference.html). The notebook example compares `Average_pollutant` with `Average_pollutant_dust_removed`.
