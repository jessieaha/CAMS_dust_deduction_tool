import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import calendar
import requests
import os
import glob
import zipfile
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm
import cdsapi
###############USER INPUT####################
project_dir          = '/tsn.tno.nl/Data/SV/sv-059025_unix/ProjectData/EU/CAMS/C71/Werkdocumenten/wp-dust/'
dataset              = 'E1a'
EEA_folder_path      = f'EEA_PM10/{dataset}'
DOWNLOAD_EEA         = False
EEA_temporal_flag    = 'hour'    # hour or day 
DOWNLOAD_CAMS        = False
COMPUTE_CAMS_DAILY   = False
Countries            = []        # select countries
YEAR                 = 2024      # int: target year 
cams_dust_threshold  = 5         # unit: ug/m3
PM10_daily_threshold = 50  # 50 µg m-3
##############################################
#############Download EEA#####################
##############################################

if (DOWNLOAD_EEA):
    from datetime import datetime 
    # Specify download path
    downloadPath = f'./EEA_data'
    os.makedirs(downloadPath, exist_ok=True)

    apiUrl = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/"
    endpoint = "ParquetFile/async"

    fileName = os.path.join(downloadPath, f"EEA_2024_FR.zip") 
    request_body = {
        "countries": Countries,
        "cities": [],
        "pollutants": ["PM10"],
        "dataset": 2,
        "dateTimeStart": f"{YEAR-1}-12-30T00:00:00Z",
        "dateTimeEnd": "2024-12-31T23:59:59Z",
        "aggregationType": EEA_temporal_flag,
        "email": ""
    }

    response = requests.post(f"{apiUrl}{endpoint}", json=request_body)
    downloadFile = response.text
    print(downloadFile)

    t_start = datetime.now() 
    while True:
        if (datetime.now() - t_start).total_seconds() > 3600:  # stop after 1 hour if the file has not been created
            break
        parquetResponse = requests.get(downloadFile) 
        if parquetResponse.status_code == 404:
            time.sleep(20)  
        else:
            break

    with open(fileName, "wb") as fp:
        fp.write(parquetResponse.content)

    with zipfile.ZipFile(fileName, 'r') as zip_ref:
        zip_ref.extractall(downloadPath)
##############################################
##################cams data###################
##############################################
netcdf_files = glob.glob(os.path.join(project_dir, 'IRA_dust/cams.eaq.ira.ENSa.dust*.nc'))
print(netcdf_files)
# datasets = [xr.open_dataset(file) for file in netcdf_files]
# cams = xr.concat(datasets, dim='time')
# cams = cams.sortby('time')
# data has been calculated loaded instead 
if COMPUTE_CAMS_DAILY:
    cams =  xr.open_mfdataset(
        netcdf_files,
        combine='by_coords',
        parallel=True,
        chunks={'time': 96},               # light time chunking on open
        compat='no_conflicts',
        join='outer'
    )
    cams = cams.sortby('time')
    daily_cams = cams.resample(time='1D').mean(keep_attrs=True).compute()
    daily_cams.to_netcdf(f"{project_dir}IRA_dust/cams_dust_daily_mean.nc")
    cams.close()
else:
    daily_cams = xr.open_mfdataset(f"{project_dir}IRA_dust/cams_dust_daily_mean.nc")
print('cams dust source:')
print(daily_cams['dust'].dims)
print(f'cams dust unit:')
print(daily_cams['dust'].units)
print(f'cams dust dimension:')
print(daily_cams['dust'].units)
# daily data are local time zone but hourly data is in UTC+1 
##################################
parquet_files = glob.glob(os.path.join(project_dir,EEA_folder_path, "*.parquet"))
dataframes = [pd.read_parquet(file) for file in parquet_files]
obs = pd.concat(dataframes, ignore_index=True)
obs['Value'] = pd.to_numeric(obs['Value'], errors='coerce').astype('float32')

def eea_to_utc(series: pd.Series, source_tz: str = "CET") -> pd.Series:

    """
    Convert a pandas Series of datetimes to timezone-aware UTC, assuming the source is a fixed UTC+1.
    - Naive timestamps are localized to the fixed-offset zone 'Etc/GMT-1' (which represents UTC+1).
    - Already tz-aware timestamps are kept and then converted to UTC.
    - Non-parsable values become NaT and are preserved.
    """
    # Parse to datetime; invalids -> NaT
    s = pd.to_datetime(series, errors="coerce")

    def _localize_if_naive(x):
        if pd.isna(x):
            return x
        # naive -> localize to fixed UTC+1 (Etc/GMT-1), avoids DST issues
        return x.tz_localize("Etc/GMT-1") if x.tzinfo is None else x

    s = s.apply(_localize_if_naive)

    # Convert everything to UTC, resulting dtype: datetime64[ns, UTC]
    return s.dt.tz_convert("UTC")


# --- Configure your datetime columns here ---
datetime_cols = ["Start", "End"]

# Apply conversion to your DataFrame `obs`
for col in datetime_cols:
    obs[col] = eea_to_utc(obs[col], source_tz="CET")
# Filter observation data

# Coerce types once (memory-friendly in-place overwrite)
obs['Validity']     = pd.to_numeric(obs['Validity'], errors='coerce')
obs['Verification'] = pd.to_numeric(obs['Verification'], errors='coerce')
obs['AggType']      = obs['AggType'].astype(str).str.lower()

valid     = obs['Validity'].gt(0)               # > 0
verified  = obs['Verification'].eq(1)           # == 1
is_hourly = obs['AggType'].eq('hour')           # string match

mask = valid & verified & is_hourly
obs = obs.loc[mask].drop(columns=['Verification', 'FkObservationLog', 'ResultTime', 'DataCapture'], errors='ignore')


# 4)Remove duplicate hours per (site, pollutant, Start)
#    Prevents double-counting if the source has overlaps.
dedup_keys = ['Pollutant', 'Start','Samplingpoint','End']
obs = obs.drop_duplicates(subset=dedup_keys)

# 5) Define the day by the *Start* timestamp (UTC)
obs['day'] = obs['Start'].dt.floor('D')  # or use df['End'].dt.floor('D') if you want to key by End

# 6) Group by site, pollutant, and day. Compute mean and count.
group_keys = ['Pollutant', 'day','Samplingpoint']
hourly_to_daily_eea = (
    obs.groupby(group_keys, sort=True)
      .agg(
          daily_mean=('Value', 'mean'),
          n_hours=('Value', 'count'),
          unit=('Unit', 'first'),         # assumes unit is consistent within group
          validity_min=('Validity', 'min')  # optional quality info
      )
      .reset_index()
)

# 7) Keep only full days (exactly 24 hourly samples)
hourly_to_daily_eea = hourly_to_daily_eea[hourly_to_daily_eea['n_hours'] == 24]

##############################################
##########75% data coverage filter############
##############################################
def filter_daily_by_coverage(
    df_daily: pd.DataFrame,
    day_col: str = 'day',
    station_col: str = 'Samplingpoint',
    reference_year: int | None = None,
    min_pct: float = 75.0,
    keep_coverage_columns: bool = True
) -> pd.DataFrame:
    """
    Keep ALL rows for stations that have coverage >= min_pct IN the reference year.
    Example: if a station has >=75% coverage in 2024, keep its 2023 data as well since average window need longer rolling time.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily dataframe (one row per day per station).
    day_col : str
        Column with daily timestamp (date or datetime).
    station_col : str
        Station/sampling point identifier column.
    reference_year : int | None
        The year used to evaluate coverage. If None, coverage is evaluated
        across ALL years (station-level over the whole dataset).
    min_pct : float
        Coverage threshold (default 75.0).
    keep_coverage_columns : bool
        If True, merge per-(station, year) coverage metrics onto the output.

    Returns
    -------
    pd.DataFrame
        All rows for stations that meet the coverage criterion in the reference year.
        If keep_coverage_columns=True, includes coverage columns:
        ['year','unique_days','total_days','coverage_percentage','sufficient_coverage'].
    """
    df = df_daily.copy()

    # Normalize day and extract year
    df[day_col] = pd.to_datetime(df[day_col], utc=True, errors='coerce').dt.floor('D')
    df = df.dropna(subset=[day_col, station_col])
    df['year'] = df[day_col].dt.year

    # --- Compute coverage per (station, year) ---
    coverage = (
        df.groupby([station_col, 'year'])[day_col]
          .nunique()
          .rename('unique_days')
          .reset_index()
    )
    coverage['total_days'] = coverage['year'].apply(lambda y: 366 if calendar.isleap(y) else 365)
    coverage['coverage_percentage'] = (coverage['unique_days'] / coverage['total_days']) * 100.0
    coverage['sufficient_coverage'] = coverage['coverage_percentage'] >= min_pct

    # --- Decide which stations qualify ---
    if reference_year is not None:
        # Stations that meet threshold in the reference year
        qualifying_stations = set(
            coverage.loc[
                (coverage['year'] == reference_year) & (coverage['sufficient_coverage']),
                station_col
            ].unique()
        )
    else:
        # If no reference year is provided, qualify stations that meet the threshold in ANY year
        qualifying_stations = set(
            coverage.loc[coverage['sufficient_coverage'], station_col].unique()
        )

    # --- Keep ALL rows for qualifying stations (across all years) ---
    out = df[df[station_col].isin(qualifying_stations)].copy()

    # Optionally attach coverage metrics (for all station-years) for traceability
    if keep_coverage_columns:
        out = out.merge(
            coverage,
            on=[station_col, 'year'],
            how='left',
            validate='many_to_one'
        )

    return out

df_processed = filter_daily_by_coverage(hourly_to_daily_eea, reference_year=2024, min_pct=75, keep_coverage_columns= False) 

# Flag exceedance days with concentration above 50 µg m-3
df_processed['Exceedance'] = df_processed['daily_mean'] > PM10_daily_threshold

print(f'total stations valid after filtering:')
print(len(df_processed['Samplingpoint'].unique()))
del(hourly_to_daily_eea); del(obs)
#####################################
############merge metadata###########
#####################################
# Filter for stations with 2024 data
# stations_with_2024_data = obs[
#     (obs['Start'] >= '2024-01-01') & (obs['Start'] < '2025-01-01')
# ]['Samplingpoint'].unique()
# obs = obs[obs['Samplingpoint'].isin(stations_with_2024_data)]

#Load metadata observational data
metadata = pd.read_csv('DataExtract.csv', low_memory = False)
# Retrieve stations from observational data
all_stations = pd.DataFrame({'Samplingpoint': df_processed['Samplingpoint'].unique()})
# Create dataframe with all stations and their location
metadata['Samplingpoint'] = metadata['Air Quality Station EoI Code'].str[:2] + '/' + metadata['Sampling Point Id']
stations = all_stations.merge(
    metadata[['Samplingpoint', 'Longitude', 'Latitude', 'Altitude','Timezone']], 
    on='Samplingpoint', 
    how='left'
).drop_duplicates(subset=['Samplingpoint'], keep='first')
# Add station coordinates to obs not need timezone since all in UTC+1 
df_processed = df_processed.merge(stations[['Samplingpoint', 'Latitude', 'Longitude','Altitude']], 
                on='Samplingpoint', 
                how='left')

############################################
############interpolate cams data###########
############################################
print('interpolate cams data')
def add_cams_daily_dust_by_station(
    cams_ds: xr.Dataset,
    df: pd.DataFrame,
    var_name: str = 'dust',
    time_col: str = 'day',
    lat_col: str = 'Latitude',
    lon_col: str = 'Longitude',
    station_col: str = 'Samplingpoint',
    spatial_method: str = 'linear'  # 'linear' (bilinear) or 'nearest'
) -> pd.DataFrame:
    """
    Add CAMS daily mean dust to df efficiently by looping over unique stations.

    For each station:
      - Interpolate CAMS daily mean field once at (lat, lon) to get a time series
      - Select nearest in time for all station rows in a single vectorized call

    Parameters
    ----------
    cams_ds : xr.Dataset
        CAMS dataset with dims [time, lat, lon] and variable `var_name` (e.g., 'dust').
    df : pd.DataFrame
        DataFrame with columns [time_col, lat_col, lon_col, station_col].
    var_name : str
        CAMS variable to sample (default 'dust').
    time_col : str
        DataFrame column name with daily timestamps (date/datetime; timezone OK).
    lat_col, lon_col : str
        Latitude & longitude column names in decimal degrees.
    station_col : str
        Station identifier column name.
    spatial_method : str
        'linear' (bilinear interpolation) or 'nearest' for lat/lon.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new column 'cams_dust'.
    """
    if var_name not in cams_ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in CAMS dataset.")
    daily_cams = cams_ds
    # Robust coord names
    lat_name = next((d for d in daily_cams.dims if d.lower().startswith('lat')), 'lat')
    lon_name = next((d for d in daily_cams.dims if d.lower().startswith('lon')), 'lon')

    # CAMS domain bounds (for clipping)
    lat_min = float(daily_cams[lat_name].min())
    lat_max = float(daily_cams[lat_name].max())
    lon_min = float(daily_cams[lon_name].min())
    lon_max = float(daily_cams[lon_name].max())

    out = df.copy()

    # Normalize times: midnight UTC, then drop tz to match xarray's naive time
    out[time_col] = (
        pd.to_datetime(out[time_col], utc=True, errors='coerce')
          .dt.floor('D')
          .dt.tz_convert(None)
    )

    # Ensure numeric lat/lon
    out[lat_col] = pd.to_numeric(out[lat_col], errors='coerce')
    out[lon_col] = pd.to_numeric(out[lon_col], errors='coerce')

    # Prepare output column
    out['cams_dust'] = np.nan

    # 2) Unique stations (first lat/lon per station)
    stations = (
        out[[station_col, lat_col, lon_col]]
        .dropna(subset=[station_col, lat_col, lon_col])
        .drop_duplicates(subset=[station_col], keep='first')
    )

    da = daily_cams[var_name]  # (time, lat, lon)

    # 3) Loop over stations, do one spatial interpolation & vectorized time selection
    for station_id, lat0, lon0 in stations.itertuples(index=False, name=None):
        # Clip to CAMS domain to avoid NaNs at edges
        lat0 = float(np.clip(lat0, lat_min, lat_max))
        lon0 = float(np.clip(lon0, lon_min, lon_max))

        # Interpolate the full daily time series at the station location (fast)
        ts_station = da.interp({lat_name: lat0, lon_name: lon0}, method=spatial_method)  # dims: time

        # All rows for this station
        mask = (out[station_col] == station_id)
        t_values = out.loc[mask, time_col].values

        # Vectorized nearest-time selection for those rows
        t_da = xr.DataArray(t_values, dims='index')  # aligns to row order
        vals = ts_station.sel(time=t_da, method='nearest').values  # dims: index

        # Optional: fallback to nearest spatial if any NaNs (e.g., outside convex hull)
        if np.isnan(vals).any():
            ts_station_nearest = da.interp({lat_name: lat0, lon_name: lon0}, method='nearest')
            vals = ts_station_nearest.sel(time=t_da, method='nearest').values

        out.loc[mask, 'cams_dust'] = np.asarray(vals, dtype=np.float32)

    return out

# cams: your xarray.Dataset with hourly dust
# df_processed: your daily DataFrame with many rows per station

df = add_cams_daily_dust_by_station(
    cams_ds=daily_cams,
    df=df_processed,
    var_name='dust',
    time_col='day',
    lat_col='Latitude',
    lon_col='Longitude',
    station_col='Samplingpoint',
    spatial_method='linear'  # bilinear; use 'nearest' for exact grid cell
)

# Example preview
print(df.head())


############################################
#############average windows################
############################################
# -------------- Setup: dust flag ----------------
df['dust_flag'] = df['cams_dust'] >= cams_dust_threshold

# Auto-detect the PM10 value column (daily)
value_col = 'daily_mean' if 'daily_mean' in df.columns else None
if value_col is None:
    raise KeyError("'daily_mean' found in df. Please specify your PM10 concentration column.")

# Normalize time column and sort once (saves repeated sorting work)
time_col = 'day'
df.sort_values([ 'Samplingpoint', time_col ], inplace=True)

# for year (2024), but use all data for baseline context 
df_2024 = df[(df[time_col] >= '2024-01-01') & (df[time_col] < '2025-01-01')].copy()
df_2024['PM10_median'] = np.nan  # pre-allocate

# if data is large
df[value_col] = pd.to_numeric(df[value_col], errors='coerce').astype('float32')
df_2024[value_col] = pd.to_numeric(df_2024[value_col], errors='coerce').astype('float32')

# Optional: make stations categorical to reduce memory
df['Samplingpoint'] = df['Samplingpoint'].astype('category')
df_2024['Samplingpoint'] = df_2024['Samplingpoint'].astype('category')

# -------------- compute per-station median when dust and exeedance are met --------------
def compute_station_baseline(st_all: pd.DataFrame,
                             st_2024: pd.DataFrame,
                             time_col: str = 'day',
                             value_col: str = value_col,
                             neighbor_n: int = 15) -> pd.Series:
    """
    For one station:
      - Find nondust days across ALL years (st_all)
      - For each dust & exceedance day in st_2024, take the last `neighbor_n` nondust days before
        and the next `neighbor_n` nondust days after -> median.
    Returns a Series aligned to st_2024.index with medians where applicable (NaN otherwise).
    """
    # Nondust pool across all years
    mask_nondust = (~st_all['dust_flag'].astype(bool)) & st_all[time_col].notna() & st_all[value_col].notna()
    nd_times = st_all.loc[mask_nondust, time_col].to_numpy()
    nd_vals  = st_all.loc[mask_nondust, value_col].to_numpy()

    # Sort nondust by time
    order = np.argsort(nd_times)
    nd_times = nd_times[order]
    nd_vals  = nd_vals[order]

    # Dust & exceedance days in 2024
    mask_dust_exc = st_2024['dust_flag'].astype(bool) & st_2024['Exceedance'].astype(bool)
    dust_idx_2024 = st_2024.index[mask_dust_exc]
    if dust_idx_2024.empty or nd_times.size == 0:
        return pd.Series(index=st_2024.index, dtype='float32')  # all NaN

    # Vector of target times
    t_targets = st_2024.loc[dust_idx_2024, time_col].to_numpy()

    # For each target time, find insertion position in nondust times
    pos = np.searchsorted(nd_times, t_targets, side='left')

    # Compute medians (loop over dust days of this station only; NumPy, not pandas)
    medians = np.full(t_targets.shape[0], np.nan, dtype='float32')
    for i, p in enumerate(pos):
        # Previous `neighbor_n` nondust days
        start_b = max(0, p - neighbor_n)
        before_vals = nd_vals[start_b:p]

        # Next `neighbor_n` nondust days
        end_a = min(nd_vals.shape[0], p + neighbor_n)
        after_vals = nd_vals[p:end_a]

        window = np.concatenate([before_vals, after_vals])
        if window.size > 0:
            medians[i] = np.nanmedian(window)  # robust to NaNs if any slipped in

    # Build result Series aligned to st_2024
    result = pd.Series(index=st_2024.index, dtype='float32')
    result.loc[dust_idx_2024] = medians
    return result

# -------------- Apply per station --------------
for station in df_2024['Samplingpoint'].cat.categories:
    # all years for station
    st_all = df[df['Samplingpoint'] == station]
    if st_all.empty:
        continue
    # 2024 rows for station
    st_2024 = df_2024[df_2024['Samplingpoint'] == station]
    if st_2024.empty:
        continue

    med_series = compute_station_baseline(st_all, st_2024, time_col=time_col, value_col=value_col, neighbor_n=15)
    df_2024.loc[st_2024.index, 'PM10_median'] = med_series.values
############################################
########Compute dust & corrected PM10#######
############################################ 
dust_exceedance_mask = df_2024['dust_flag'].astype(bool) & df_2024['Exceedance'].astype(bool)

# Dust mass on dust-exceedance days = Value - baseline
df_2024['Dust_contribution'] = np.where(dust_exceedance_mask,
                           df_2024[value_col] - df_2024['PM10_median'],
                           np.nan).astype('float32')

# Corrected PM10: baseline on dust days, original on others
df_2024['corrected_PM10'] = np.where(dust_exceedance_mask,
                                     df_2024[value_col] - df_2024['Dust_contribution'],  # equals baseline
                                     df_2024[value_col]).astype('float32')
import pyarrow as pa
import pyarrow.parquet as pq
output_parquet = f'{project_dir}CAMS_dust_deduction_{dataset}_{EEA_temporal_flag}_{YEAR}.parquet'
table = pa.Table.from_pandas(df_2024)
pqwriter = pq.ParquetWriter(output_parquet, table.schema, use_dictionary=True, compression='snappy')
pqwriter.write_table(table)
pqwriter.close()
print(f'save to {output_parquet}')