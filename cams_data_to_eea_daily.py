"""
For using the daily EEA aggregated observation only! 

Preporcessed the data to daily based on EEA daily station at local time zone 
"""
import numpy as np
import pandas as pd
import os
import glob
import xarray as xr
import psutil
import time
############### INPUT ####################
project_dir         = '/tsn.tno.nl/Data/SV/sv-059025_unix/ProjectData/EU/CAMS/C71/Werkdocumenten/wp-dust/'
# project_dir          = '.'
dataset              = 'E1a'
EEA_temporal_flag    = 'day'    # hour or day 
EEA_folder_path      = f'EEA_PM10/{dataset}/{EEA_temporal_flag}' 
YEAR                 = 2024
METADATA_PATH        = os.path.join(project_dir, 'EEA_PM10/DataExtract.csv')
CAMS_FILES           = os.path.join(project_dir, 'IRA_dust/cams.eaq.ira.ENSa.dust*.nc')
###########################################
# Monitoring System Resources
n_cpus = os.cpu_count()
n_threads = psutil.cpu_count(logical=True)
print(f"{n_cpus} Physical CPUs | {n_threads} Logical Threads detected.")
start_time = time.time()
print("Reading CAMS NetCDF files...")
cams_ds = xr.open_mfdataset(
    CAMS_FILES,
    combine='by_coords', 
    parallel=True,
    chunks={'time': 100} # Chunking helps Dask manage memory
).sortby('time')

# ---  Process Observations & Timezones ---
print("Loading EEA Parquet files...")
eea_daily = glob.glob(os.path.join(project_dir, EEA_folder_path, "*.parquet"))
obs = pd.concat([pd.read_parquet(f) for f in eea_daily], ignore_index=True)

obs['Start'] = pd.to_datetime(obs['Start'])
obs = obs[obs['AggType'] == EEA_temporal_flag].drop(
    columns=['End', 'ResultTime', 'Pollutant', 'Unit', 'DataCapture', 'FkObservationLog']
)

# Filter for YEAR and valid stations
stations_YEAR = obs.loc[obs['Start'].dt.year == YEAR, 'Samplingpoint'].unique()
df_processed = obs[obs['Samplingpoint'].isin(stations_YEAR)].copy()

# ---  Merge Metadata ---
metadata = pd.read_csv(METADATA_PATH, low_memory=False)
metadata['Samplingpoint'] = metadata['Air Quality Station EoI Code'].str[:2] + '/' + metadata['Sampling Point Id']
metadata = metadata[metadata['Air Pollutant'] == 'PM10']

def get_tz_offset(tz_str):
    if pd.isna(tz_str) or 'UTC' not in str(tz_str): return 0
    tz_str = str(tz_str).replace('UTC', '').replace('+', '')
    return int(tz_str) if tz_str else 0

metadata['tz_offset'] = metadata['Timezone'].apply(get_tz_offset)
metadata_clean = metadata[
    ['Samplingpoint', 'Longitude', 'Latitude', 'Altitude', 'tz_offset']
].drop_duplicates(subset=['Samplingpoint'])

df_processed = df_processed.merge(metadata_clean, on='Samplingpoint', how='left')

# ---  Optimized Local-Time Interpolation ---
def add_cams_local_daily_dust(cams_ds, df, var_name='dust', buffer=0.2):
    out = df.copy()
    out['cams_dust'] = np.nan
    
    lat_dim = next(d for d in cams_ds.dims if d.lower().startswith('lat'))
    lon_dim = next(d for d in cams_ds.dims if d.lower().startswith('lon'))

    # Determine CAMS spatial bounds
    lat_min, lat_max = cams_ds[lat_dim].min().item(), cams_ds[lat_dim].max().item()
    lon_min, lon_max = cams_ds[lon_dim].min().item(), cams_ds[lon_dim].max().item()

    # Get unique stations
    all_stats = out[['Samplingpoint', 'Latitude', 'Longitude', 'tz_offset']].drop_duplicates('Samplingpoint')

    # Apply Spatial Filter: Station must be within bounds +/- buffer
    mask_in_bounds = (
        (all_stats['Latitude'] >= lat_min + buffer) & 
        (all_stats['Latitude'] <= lat_max - buffer) & 
        (all_stats['Longitude'] >= lon_min + buffer) & 
        (all_stats['Longitude'] <= lon_max - buffer)
    )
    
    unique_stats = all_stats[mask_in_bounds].copy()
    dropped_count = len(all_stats) - len(unique_stats)
    
    if dropped_count > 0:
        print(f"Dropped {dropped_count} stations located outside the CAMS domain (Buffer: {buffer}°)")

    print(f"Vectorized spatial interpolation for {len(unique_stats)} valid stations...")
    
    # 1. Vectorized spatial interpolation
    # This creates a 'station' dimension
    cams_at_stations = cams_ds[var_name].interp(
        {lat_dim: xr.DataArray(unique_stats['Latitude'], dims="station"),
         lon_dim: xr.DataArray(unique_stats['Longitude'], dims="station")},
        method='linear'
    )

    # 2. Assign the labels to the NEWLY created 'station' dimension 
    cams_at_stations = cams_at_stations.assign_coords(
        station=("station", unique_stats['Samplingpoint'].values)
    )
    print(f"Temporal local-time resampling...")
    for sid in unique_stats['Samplingpoint']:
        offset = unique_stats.loc[unique_stats['Samplingpoint'] == sid, 'tz_offset'].values[0]
        
        ts_utc = cams_at_stations.sel(station=sid)
        local_times = ts_utc.time.values + np.timedelta64(int(offset), 'h')
        
        ts_local = ts_utc.assign_coords(time=local_times)
        ts_daily_local = ts_local.resample(time='1D').mean().compute()
        
        mask = out['Samplingpoint'] == sid
        target_dates = out.loc[mask, 'Start'].dt.floor('D').values
        
        vals = ts_daily_local.sel(time=target_dates, method='nearest').values
        out.loc[mask, 'cams_dust'] = vals

    return out
# Execution
final_df = add_cams_local_daily_dust(cams_ds, df_processed)

# ---  Saving ---
output_filename = f"EEA_CAMS_merged_{dataset}_{EEA_temporal_flag}_{YEAR}.parquet"
output_path = os.path.join(project_dir, output_filename)

print(f"Saving to {output_path}...")
final_df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
 
end_time = time.time()
duration = end_time - start_time
print("-" * 30) 
print(f"Total Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
print(f"Rows processed: {len(final_df)}")
print("-" * 30)
# #Load metadata observational data

# # Retrieve stations from observational data
# all_stations = pd.DataFrame({'Samplingpoint': df_processed['Samplingpoint'].unique()})
# # Create dataframe with all stations and their location
# stations = all_stations.merge(
#     metadata[['Samplingpoint', 'Longitude', 'Latitude', 'Altitude','Timezone']], 
#     on='Samplingpoint', 
#     how='left'
# ).drop_duplicates(subset=['Samplingpoint'], keep='first')
# # Add station coordinates to obs not need timezone since all in UTC+1 
# df_processed = df_processed.merge(stations[['Samplingpoint', 'Latitude', 'Longitude','Altitude']], 
#                 on='Samplingpoint', 
#                 how='left')

# ###############process CAMS##################
# # Definition to extract timezone
# def extract_timezone_offset(tz_string):
#     if tz_string == 'UTC':
#         return 0
#     else:
#         return int(tz_string.replace('UTC+', ''))

# # Definition to interpolate from closest model gridpoints
# def interpolate_local_region(ds, lat, lon, buffer=0.25):
#     local_region = ds.sel(
#         lat=slice(lat-buffer, lat+buffer),
#         lon=slice(lon-buffer, lon+buffer)
#     )
#     return local_region.interp(lat=lat, lon=lon, method='linear')

# # Flag dust for each samplingpoint separately
# unique_stations = obs['Samplingpoint'].unique()

# datasets = []

# for i, station in enumerate(unique_stations):
#     print(f"\rProcessing station {i+1}/{len(unique_stations)}", end="")
    
#     # Get station coordinates and timezone
#     station_obs = obs[obs['Samplingpoint'] == station]
#     lat = station_obs['Latitude'].iloc[0]
#     lon = station_obs['Longitude'].iloc[0]
#     tz_offset = extract_timezone_offset(station_obs['Timezone'].iloc[0])
    
#     # Interpolate CAMS data to this station location
#     station_cams = interpolate_local_region(cams_dust, lat, lon)
    
#     # Convert UTC to local timezone
#     station_cams['time_local'] = station_cams['time'] + pd.to_timedelta(tz_offset, unit='h')
    
#     # Calculate daily average dust using local timezone
#     daily_cams = station_cams.groupby(station_cams['time_local'].dt.date).mean()
#     datasets.append(daily_cams)

# cams = xr.concat(datasets, dim='time')
# cams = cams.sortby('time')

# # Prepare arrays of unique station coordinates
# unique_stations_df = obs[['Samplingpoint', 'Longitude', 'Latitude', 'Timezone']].drop_duplicates().reset_index(drop=True)

# # Interpolate CAMS data to all station locations at once
# cams_interp = cams_dust.interp(
#     lat=xr.DataArray(unique_stations_df['Latitude'], dims='station'),
#     lon=xr.DataArray(unique_stations_df['Longitude'], dims='station')
# )

# # Add timezone offsets
# def extract_timezone_offset_vec(tz_series):
#     return tz_series.replace('UTC', 'UTC+0').str.replace('UTC\\+', '', regex=True).astype(int)

# tz_offsets = extract_timezone_offset_vec(unique_stations_df['Timezone'])
# cams_interp = cams_interp.assign_coords(station=('station', unique_stations_df['Samplingpoint']))

# # Convert UTC to local time for each station
# cams_interp['time_local'] = cams_interp['time'] + xr.DataArray(
#     tz_offsets.values, dims='station'
# ).astype('timedelta64[h]')

# # Create a date coordinate from time_local
# date = xr.DataArray(
#     cams_interp["time_local"].dt.floor("D").data,
#     dims=("time", "station"),
#     name="date"
# )
# cams_interp = cams_interp.assign_coords(date=date)

# # Group by date and station, then average over time
# cams_daily = cams_interp.groupby("date").mean(dim="time")

# # Re-chunk to smaller blocks
# cams_daily = cams_daily.chunk({'date': 50, 'station': 50})
