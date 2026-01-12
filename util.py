import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def station_plot(
    station_name: str,
    obs_df: pd.DataFrame,
    # stations_df: pd.DataFrame,
    # Column names in obs_df
    station_column: str = 'Samplingpoint',
    time_column: str = 'Start',
    observed_pm10_col: str = 'observed_PM10',
    corrected_pm10_col: str = 'corrected_PM10',
    cams_dust_col: str = 'cams_dust',
    dust_flag_col: str = 'dust_flag',
    exceedance_threshold: float = 50.0,
    cams_dust_threshold: float = 5.0,
    # Column names in stations_df
    stations_id_col: str = 'Samplingpoint',
    altitude_col: str = 'Altitude',
    # Plot controls
    year: int = 2024,
    figsize=(12, 5),
    title_fontsize: int = 12,   # smaller title font
    label_fontsize: int = 10,   # smaller axis label font
    legend_fontsize: int = 9,   # smaller legend font
):
    """
    Plot PM10 (observed vs. corrected) and CAMS dust for a station.

    Parameters
    ----------
    station_name : str
        Station identifier to select in `obs_df` 
    obs_df : pd.DataFrame
        Observations dataframe containing time series and flags.
    station_column, time_column : str
        Column names in `obs_df` for station ID and time.
    observed_pm10_col, corrected_pm10_col : str
        Column names for observed and corrected PM10 in `obs_df`.
    cams_dust_col : str
        Column name for CAMS dust values in `obs_df`.
    dust_flag_col : str
        Column name for dust flag (boolean) in `obs_df`.
    exceedance_threshold : float
        Horizontal line for exceedance threshold (µg/m³).
    cams_dust_threshold : float
        Horizontal line for CAMS dust threshold (µg/m³).
    stations_id_col, altitude_col : str
        Column names in `stations_df` for station ID and altitude.
    year : int
        Year window to show on the x-axis (Jan 1 to Dec 31).
    figsize : tuple
        Matplotlib figure size.
    title_fontsize, label_fontsize, legend_fontsize : int
        Font sizes for title, axis labels, and legend.
    """

    # --- Select data for this station ---
    station_mask = (obs_df[station_column] == station_name)
    station_data = obs_df.loc[station_mask].copy()
    if station_data.empty:
        raise ValueError(f"No data found for station '{station_name}' in obs_df (column '{station_column}').")

    # --- Normalize time column to datetime (UTC ok or naive) ---
    station_data[time_column] = pd.to_datetime(station_data[time_column], errors='coerce')
    # filter to requested year range
    t_min = pd.Timestamp(f'{year}-01-01')
    t_max = pd.Timestamp(f'{year}-12-31')
    station_data = station_data[(station_data[time_column] >= t_min) & (station_data[time_column] <= t_max)]

    if station_data.empty:
        raise ValueError(f"No rows for '{station_name}' within year {year} in '{time_column}'.")

    # --- Get altitude from stations_df (optional) ---
    altitude = None
    if not obs_df.empty and stations_id_col in obs_df.columns:
        alt_row = obs_df.loc[obs_df[stations_id_col] == station_name, altitude_col]
        if not alt_row.empty and pd.notna(alt_row.iloc[0]):
            altitude = float(alt_row.iloc[0])

    # --- Build figure ---
    fig, axs = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)

    # --- Upper panel: PM10 observed vs corrected ---
    axs[0].plot(
        station_data[time_column], station_data[observed_pm10_col],
        marker='o', markersize=4, label='Original PM10',
        linewidth=1.5, alpha=0.8
    )
    axs[0].plot(
        station_data[time_column], station_data[corrected_pm10_col],
        marker='x', markersize=4, linestyle='--', label='Corrected PM10',
        linewidth=1.5, alpha=0.8
    )

    # Highlight dust days (only where dust_flag is True)
    if dust_flag_col in station_data.columns:
        dust_days = station_data[station_data[dust_flag_col].astype(bool)]
        if not dust_days.empty:
            axs[0].scatter(
                dust_days[time_column], dust_days[observed_pm10_col],
                color='red', s=30, alpha=0.7, label='Dust days', zorder=5
            )

    axs[0].axhline(y=exceedance_threshold, color='blue', linestyle=':', alpha=0.6, label=f'Exceedance ({exceedance_threshold} µg/m³)')
    axs[0].set_ylabel('PM10 (µg/m³)', fontsize=label_fontsize)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='upper right', fontsize=legend_fontsize, framealpha=0.4)

    title_alt = f" | Altitude: {altitude:.0f} m" if altitude is not None else ""
    axs[0].set_title(f'{station_name} — PM10 Timeseries {year}{title_alt}', fontsize=title_fontsize, fontweight='bold')

    # --- Lower panel: CAMS dust ---
    axs[1].plot(
        station_data[time_column], station_data[cams_dust_col],
        color='green', linewidth=2, alpha=0.7, label='CAMS Dust'
    )
    axs[1].axhline(y=cams_dust_threshold, color='green', linestyle=':', alpha=0.6, label=f'Dust threshold ({cams_dust_threshold} µg/m³)')
    axs[1].set_xlabel('Date', fontsize=label_fontsize)
    axs[1].set_ylabel('CAMS Surface Dust (µg/m³)', fontsize=label_fontsize, color='green')
    axs[1].tick_params(axis='y', labelcolor='green')
    axs[1].legend(loc='upper right', fontsize=legend_fontsize, framealpha=0.4)

    # --- X-axis limits ---
    axs[1].set_xlim(t_min, t_max)

    plt.tight_layout()
    plt.show()

    return fig, axs

def until_check():
    return "util.py has been imported"

def calculate_data_coverage(df, start=None, end=None, min_pct=75, 
                            date_col : str = 'Start'
 ):
    """
    Coverage per station across the chosen period [start, end] (inclusive).
    Automatically handles leap years and partial ranges.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['tmpdate'] = df[date_col].dt.normalize()

    period_start = pd.to_datetime(start) if start is not None else df['tmpdate'].min()
    period_end   = pd.to_datetime(end)   if end   is not None else df['tmpdate'].max()
    period_end   = period_end.normalize()

    # Expected days: inclusive date range count
    expected_days_total = pd.date_range(period_start, period_end, freq='D').size

    # Observed unique days per station within window
    observed = (
        df[(df['tmpdate'] >= period_start) & (df['tmpdate'] <= period_end)]
          .groupby('Samplingpoint')['tmpdate']
          .nunique()
          .rename('unique_days')
          .to_frame()
    )

    observed['expected_days'] = expected_days_total
    observed['coverage_percentage'] = (observed['unique_days'] / observed['expected_days']) * 100.0
    observed['sufficient_coverage'] = observed['coverage_percentage'] >= float(min_pct)

    return observed

def compute_median_for_station(station_df: pd.DataFrame,
                               flag_col : str = 'dust_flag',
                               obs_col  : str = 'observed_PM10',
                               date_col : str = 'Start',
                               Exceedance_col : str='Exceedance'
                               ):
    # Sort by date
    station_df = station_df.sort_values('Start')

    # Separate nondust days
    nondust = station_df[station_df[flag_col] == False][[date_col,obs_col]].copy()
    nd_dates = nondust[date_col].to_numpy()
    nd_vals = nondust[obs_col].to_numpy(dtype=float)

    # Prepare result
    result = pd.Series(index=station_df.index, dtype=float)

    # Identify dust days with Exceedance
    dust_days = station_df[(station_df[flag_col]) & (station_df[Exceedance_col])]

    if nondust.empty or dust_days.empty:
        return result

    # Vectorized search for positions
    positions = np.searchsorted(nd_dates, dust_days[date_col].to_numpy())

    for i, idx in enumerate(dust_days.index):
        pos = positions[i]
        before_slice = nd_vals[max(0, pos-15):pos]
        after_slice = nd_vals[pos:pos+15]
        window = np.concatenate([before_slice, after_slice])
        if window.size > 0:
            result[idx] = np.median(window)

    return result
