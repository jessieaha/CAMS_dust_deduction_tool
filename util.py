import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from ipywidgets import Output, HTML, VBox
from ipyleaflet import Map, CircleMarker, LayersControl, basemaps
import matplotlib.cm as cm
import plotly.express as px

def plot_station_timeseries(
    station_name: str,
    obs_df: pd.DataFrame,
    station_column: str = 'Samplingpoint',
    time_column: str = 'Start',
    observed_pm10_col: str = 'observed_PM10',
    corrected_pm10_col: str = 'corrected_PM10',
    cams_dust_col: str = 'cams_dust',
    dust_flag_col: str = 'dust_flag',
    altitude_col : str='Altitude',
    exceedance_threshold: float = 50.0,
    cams_dust_threshold: float = 5.0,
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
    altitude = None
    if not obs_df.empty and station_column in obs_df.columns:
        alt_row = obs_df.loc[obs_df[station_column] == station_name, altitude_col]
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
    return fig, axs



def calculate_data_coverage(df, 
                            start : str,
                            end : str,
                            min_pct: int =75, 
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

def plot_exceedance_maps_discrete(df, columns_to_plot, titles, vmax,cbar_text, cmap_name='gist_heat_r', n_colors=10, extent=None):
   
    """
    Plots maps of exceedance data with discrete color levels.

    Args:
        df (pd.DataFrame): The DataFrame containing the data with Latitude, Longitude, and columns to plot.
        columns_to_plot (list): A list of column names from the DataFrame to plot.
        titles (list): A list of titles for each subplot, corresponding to columns_to_plot.
        vmax (float): The maximum value for the color scale.
        cmap_name (str, optional): The name of the colormap. Defaults to 'gist_heat_r'.
        n_colors (int, optional): The number of discrete color levels. Defaults to 10.
        extent (list, optional): [min_lon, max_lon, min_lat, max_lat] for map extent. If None, calculated dynamically.
    """
    assert len(columns_to_plot) == len(titles), "Number of columns to plot must match number of titles."

    # Create discrete colormap and normalizer
    cmap = plt.get_cmap(cmap_name, n_colors)
    bounds = np.linspace(0, vmax, n_colors + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1, len(columns_to_plot), figsize=(5 * len(columns_to_plot), 5),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    # Ensure axs is iterable even for a single subplot
    if len(columns_to_plot) == 1:
        axs = [axs]

    # Store the last scatter object to use for the single colorbar
    last_scatter = None

    for i, col in enumerate(columns_to_plot):
        ax = axs[i]

        # Scatter plot
        scatter = ax.scatter(df['Longitude'], df['Latitude'],
                             c=df[col],
                             s=15, cmap=cmap, norm=norm,
                             transform=ccrs.PlateCarree(), alpha=0.8,
                             edgecolors='black', linewidth=0.5)
        last_scatter = scatter # Keep track of the last scatter for the colorbar

        # Add map features
        ax.add_feature(cfeature.COASTLINE, alpha=0.6)
        ax.add_feature(cfeature.BORDERS, alpha=0.6)
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)
        ax.add_feature(cfeature.LAKES, alpha=0.3)

        # Set extent
        if extent is None:
            buffer_val = 1.5
            dynamic_extent = [
                df['Longitude'].min() - buffer_val, df['Longitude'].max() + buffer_val,
                df['Latitude'].min() - buffer_val, df['Latitude'].max() + buffer_val
            ]
            ax.set_extent(dynamic_extent, crs=ccrs.PlateCarree())
        else:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Add gridlines
        gl = ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
        gl.bottom_labels = True
        gl.left_labels = True

        # Set individual subplot titles
        ax.set_title(titles[i], fontsize=12)

    # Create a single colorbar for the entire figure if there's a scatter plot
    if last_scatter:
        cbar = fig.colorbar(last_scatter, ax=axs, orientation='vertical',
                            shrink=0.8, pad=0.05, extend='max', ticks=bounds) # Changed orientation and pad
        cbar.set_label(cbar_text, fontsize=11) # Combined label

    # plt.tight_layout() 
    plt.show()

def plot_interactive_station_map(
    df,
    color_column,
    size_column=None,
    color_type='auto',  # 'auto', 'continuous_scale', 'discrete_sequence'
    colorscale='Viridis',
    zoom=5,
    center_lat=None,
    center_lon=None,
    map_title=None,
    map_subtitle=None,
    colorbar_title='', # Default to empty string to remove title
    marker_size_range=[8, 20],
    n_discrete_colors=10 # Number of bins/colors if discrete and color_column is continuous
):
    """
    Plots an interactive scatter mapbox of stations using Plotly Express.

    Args:
        df (pd.DataFrame): The DataFrame containing the station data.
        color_column (str): The name of the column to use for coloring the markers.
        size_column (str, optional): The name of the column to use for sizing the markers.
        color_type (str): How to interpret the color. 
                          'auto' (default: px decides based on data type),
                          'continuous_scale', 'discrete_sequence'.
                          If 'discrete_sequence' and color_column is continuous, data will be binned.
        colorscale (str or list/dict): The Plotly color scale name (for continuous) 
                                       or a list/dict (for discrete sequence) to use.
        zoom (int): The initial zoom level for the map.
        center_lat (float, optional): The latitude for the map's center.
        center_lon (float, optional): The longitude for the map's center.
        map_title (str, optional): The main title for the map.
        map_subtitle (str, optional): A subtitle for the map.
        colorbar_title (str): The title for the color bar. Defaults to '' (empty string).
        marker_size_range (list): [min_size, max_size] for marker sizing.
        n_discrete_colors (int): Number of bins/colors to use if `color_type` is 'discrete_sequence' 
                                 and `color_column` is continuous.
    """

    center_dict = {'lat': center_lat, 'lon': center_lon} if center_lat is not None and center_lon is not None else None
    
    plotting_df = df.copy() # Work on a copy to avoid modifying original df
    color_col_to_plot = color_column

    plot_kwargs = {
        'lat': 'Latitude',
        'lon': 'Longitude',
        'size': size_column,
        'zoom': zoom,
        'center': center_dict,
        'hover_name': 'Samplingpoint',
        'mapbox_style': 'open-street-map',
        'size_max': marker_size_range[1]
    }

    if color_type == 'continuous_scale' or \
       (color_type == 'auto' and pd.api.types.is_numeric_dtype(plotting_df[color_column])):
        plot_kwargs['color_continuous_scale'] = colorscale
        plot_kwargs['color'] = color_col_to_plot
        # color_continuous_colorbar is not a direct argument for px.scatter_mapbox.
        # It's usually part of the color_continuous_scale dict or set via update_layout/update_coloraxes.
        # For simplicity, if colorbar_title is desired, we'll set it after fig creation.

    elif color_type == 'discrete_sequence' or \
         (color_type == 'auto' and not pd.api.types.is_numeric_dtype(plotting_df[color_column])):
        # If continuous column needs discrete colors, bin it
        if pd.api.types.is_numeric_dtype(plotting_df[color_column]):
            color_col_to_plot = color_column + '_binned'
            plotting_df[color_col_to_plot] = pd.cut(
                plotting_df[color_column],
                bins=n_discrete_colors,
                labels=[f'{i}-{i+1}' for i in range(n_discrete_colors)], # Generic labels
                include_lowest=True
            )
        plot_kwargs['color'] = color_col_to_plot
        plot_kwargs['color_discrete_sequence'] = colorscale if isinstance(colorscale, list) else px.colors.qualitative.Plotly # Fallback
        # For discrete colors, legend title is often derived from column name, no direct colorbar title option in px.

    fig = px.scatter_mapbox(plotting_df, **plot_kwargs)

    # Explicitly set colorbar title AFTER figure creation if color_continuous_scale was used
    if colorbar_title and (color_type == 'continuous_scale' or \
        (color_type == 'auto' and pd.api.types.is_numeric_dtype(plotting_df[color_column]))):
        fig.update_layout(coloraxis_colorbar_title_text=colorbar_title)


    # Update layout for title and subtitle
    fig.update_layout(
        title_text=map_title,
        title_x=0.5,
        title_y=0.95,
    )

    if map_subtitle:
        fig.add_annotation(
            text=map_subtitle,
            xref="paper", yref="paper",
            x=0.5, y=0.91,
            showarrow=False,
            font=dict(size=12, color="gray"),
            xanchor="center", yanchor="top"
        )
    
    # Explicitly configure size legend if size_column is present
    if size_column:
        # Ensure the size legend title is clear. px usually does this, but explicit can help.
        fig.update_layout(
            legend_title_text=f'Size: {size_column}'
        )

    fig.show()
    return fig  



def map_timeseries_clickable_plot(obs_df, year, exceedance_threshold, cams_dust_threshold):
    
    # -----------------------------
    # Config / Inputs
    # -----------------------------
    station_col = 'Samplingpoint'
    time_col    = 'Start'
    lat_col     = 'Latitude'
    lon_col     = 'Longitude'
    observed_pm10_col = 'observed_PM10'

    # Use obs_df (fixed year view)
    df = obs_df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # Single coordinate per station (first non-null)
    coords = (
        df[[station_col, lat_col, lon_col]]
        .dropna(subset=[lat_col, lon_col])
        .drop_duplicates(subset=[station_col])
        .set_index(station_col)
    )

    # Per-station average PM10 (for marker color/size)
    avg_pm10 = (
        df.groupby(station_col)[observed_pm10_col]
          .mean()
          .rename('avg_pm10')
          .to_frame()
    )

    summary = (
        avg_pm10.join(coords, how='inner')
                .dropna(subset=[lat_col, lon_col])
                .reset_index()
    )

    if summary.empty:
        # Instead of raising an error, return a VBox with an error message
        return VBox([HTML("<b>Error: No stations with coordinates and PM10 in obs_df. Check Latitude/Longitude availability.</b>")])

    # -----------------------------
    # Helpers for color/size scaling
    # -----------------------------
    def color_for_value(val, vmin, vmax, cmap_name='plasma'):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap_name)
        return mcolors.to_hex(cmap(norm(val)))

    def radius_for_value(val, vmin, vmax, rmin=6, rmax=16):
        return float(np.interp(val, [vmin, vmax], [rmin, rmax]))

    # Scale bounds
    vmin = float(summary['avg_pm10'].min())
    vmax = float(summary['avg_pm10'].max())

    # -----------------------------
    # Build the map (centered over median coords)
    # -----------------------------
    center_lat = float(summary[lat_col].median())
    center_lon = float(summary[lon_col].median())

    m = Map(center=(center_lat, center_lon), zoom=6, basemap=basemaps.OpenStreetMap.Mapnik)
    m.add_control(LayersControl())

    # Output area for the time series
    plot_out = Output()

    # Instruction
    desc = HTML("<b>Click a station marker</b> to load its PM10 time series below.")

    # -----------------------------
    # Add markers (click → plot_station_timeseries)
    # -----------------------------
    for _, row in summary.iterrows():
        sp  = row[station_col]
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        val = float(row['avg_pm10'])

        color  = color_for_value(val, vmin, vmax, cmap_name='plasma')
        radius = int(radius_for_value(val, vmin, vmax, rmin=6, rmax=16)) # Cast radius to int

        marker = CircleMarker(
            location=(lat, lon),
            radius=radius,
            color=color,
            fill_color=color,
            fill_opacity=0.75,
            stroke=False
        )

        # Use a factory function to capture the correct `sp` for each marker
        def create_on_click_callback(current_sp):
            def on_click_callback(**kwargs):
                station_name = current_sp
                with plot_out:
                    plot_out.clear_output(wait=True)
                    try:
                        fig_ts, axs = plot_station_timeseries(
                            station_name=station_name,
                            obs_df=df,                      # obs_df (fixed year view)
                            year=year,                      # fixed
                            exceedance_threshold=exceedance_threshold,
                            cams_dust_threshold=cams_dust_threshold,
                            figsize=(12, 5),
                        )
                        plt.show()
                    except Exception as e:
                        print(f"Cannot render station '{station_name}': {e}")
            return on_click_callback

        marker.on_click(create_on_click_callback(sp))
        m.add_layer(marker)

    # -----------------------------
    # Display
    # -----------------------------
    return VBox([desc, m, plot_out])

def until_check():
    return "util.py has been imported"
