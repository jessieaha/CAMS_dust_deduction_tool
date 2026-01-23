import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import ipywidgets  
from IPython.display import display
from ipywidgets import Output, HTML, VBox
from ipyleaflet import Map, CircleMarker, LayersControl, basemaps, LayersControl, LegendControl
from ipywidgets import VBox, HTML, Output
import matplotlib.pyplot as plt  # used by plot_station_timeseries if it shows figures
import matplotlib.cm as cm
import plotly.express as px 
import plotly.graph_objects as go

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

    fig, axs = plt.subplots(1, len(columns_to_plot), figsize=(5 * len(columns_to_plot), 6),
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

        # --- Gridlines & labels ---
        if i == 0:
            # First subplot: show left labels; hide grid "lines"
            gl = ax.gridlines(
                draw_labels=True, dms=False, x_inline=False, y_inline=False
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.bottom_labels = True
            gl.left_labels = True

            # Hide the actual grid lines (keep only labels)
            # These attributes are supported in newer Cartopy versions.
            # If not available in your env, you can also do: gl.alpha = 0
            gl.xlines = False
            gl.ylines = False

            # Optional: tweak label style
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 9}
        else:
            # Other subplots: no labels, no lines
            gl = ax.gridlines(
                draw_labels=False, dms=False, x_inline=False, y_inline=False
            )
            gl.bottom_labels = True
            gl.xlines = False
            gl.ylines = False
            gl.xlabel_style = {'size': 9}

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
    color_type='auto',             # 'auto', 'continuous_scale', 'discrete_sequence'
    colorscale='Viridis',
    zoom=5,
    center_lat=None,
    center_lon=None,
    map_title=None,
    map_subtitle=None,
    colorbar_title=' ',            # Default to single space to show no title if unwanted
    marker_size_range=[8, 20],     # [min_px, max_px] — note: px.scatter_mapbox only uses size_max directly
    n_discrete_colors=10,
    # NEW: user-controlled size legend values (your smin/smax)
    size_legend_min=None,
    size_legend_max=None, 
    size_legend_steps=4
):
    """
    Plots an interactive scatter mapbox of stations using Plotly Express.

    Args:
        df (pd.DataFrame): DataFrame containing station data.
        color_column (str): Column for coloring.
        size_column (str, optional): Column for sizing bubbles.
        color_type (str): 'auto' | 'continuous_scale' | 'discrete_sequence'.
        colorscale (str or list/dict): Colors for color dim.
        zoom (int): Map zoom.
        center_lat (float, optional), center_lon (float, optional): Map center.
        map_title (str, optional), map_subtitle (str, optional).
        colorbar_title (str): Title for colorbar (continuous color).
        marker_size_range (list[int,int]): [min_px, max_px] desired display sizes. Only max is used by px directly.
        n_discrete_colors (int): #bins if discrete sequence on numeric col.
        size_legend_min (float, optional): Minimum data value to display in size legend.
        size_legend_max (float, optional): Maximum data value to display in size legend.
        size_legend_steps (int): Number of size bubbles to show in legend.
    """

    # Center dict may be None; we'll also provide a safe fallback for legend-only traces later
    center_dict = {'lat': center_lat, 'lon': center_lon} if (center_lat is not None and center_lon is not None) else None

    plotting_df = df.copy()
    color_col_to_plot = color_column

    plot_kwargs = {
        'lat': 'Latitude',
        'lon': 'Longitude',
        'size': size_column,
        'zoom': zoom,
        'center': center_dict,
        'hover_name': 'Samplingpoint',
        'mapbox_style': 'open-street-map',
        'size_max': marker_size_range[1],  # px uses only size_max to set the top bubble size
    }

    # --- Color handling ---
    is_color_numeric = pd.api.types.is_numeric_dtype(plotting_df[color_column])

    if color_type == 'continuous_scale' or (color_type == 'auto' and is_color_numeric):
        plot_kwargs['color'] = color_col_to_plot
        plot_kwargs['color_continuous_scale'] = colorscale

    elif color_type == 'discrete_sequence' or (color_type == 'auto' and not is_color_numeric):
        # If user forced discrete sequence on a numeric column, bin it:
        if is_color_numeric:
            color_col_to_plot = color_column + '_binned'
            plotting_df[color_col_to_plot] = pd.cut(
                plotting_df[color_column],
                bins=n_discrete_colors,
                labels=[f'{i}-{i+1}' for i in range(n_discrete_colors)],
                include_lowest=True
            )
        plot_kwargs['color'] = color_col_to_plot
        plot_kwargs['color_discrete_sequence'] = colorscale if isinstance(colorscale, list) else px.colors.qualitative.Plotly

    # --- Create figure ---
    fig = px.scatter_mapbox(plotting_df, **plot_kwargs)

    # --- Continuous color: set colorbar title + tick format ---
    if colorbar_title and (color_type == 'continuous_scale' or (color_type == 'auto' and is_color_numeric)):
        fig.update_coloraxes(colorbar=dict(title=colorbar_title, tickformat=".2f"))

    # --- Hover formatting for size and numeric color ---
    custom_cols, custom_idx = [], {}
    pos = 0
    if size_column is not None:
        custom_cols.append(size_column); custom_idx['size'] = pos; pos += 1
    if color_column and is_color_numeric:
        custom_cols.append(color_column); custom_idx['color'] = pos; pos += 1

    if custom_cols:
        fig.update_traces(
            customdata=plotting_df[custom_cols].to_numpy(),
            selector=dict(type="scattermapbox")
        )
        hover_lines = ["<b>%{hovertext}</b>"]
        if 'size' in custom_idx:
            hover_lines.append(f"{size_column}: %{{customdata[{custom_idx['size']}]:.0f}}")
        if 'color' in custom_idx:
            hover_lines.append(f"{color_column}: %{{customdata[{custom_idx['color']}]:.2f}}")
        hover_lines.append("<extra></extra>")
        fig.update_traces(
            hovertemplate="<br>".join(hover_lines),
            selector=dict(type="scattermapbox")
        )

    # --- Title & subtitle ---
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

    # ===============================
    # Size legend (legend-only traces)
    # ===============================
    if size_column is not None:
        # 1) Determine representative data values to show in legend
        data_min = float(np.nanmin(plotting_df[size_column]))
        data_max = float(np.nanmax(plotting_df[size_column]))

        smin = data_min if size_legend_min is None else float(size_legend_min)
        smax = data_max if size_legend_max is None else float(size_legend_max)
        if smax < smin:
            smin, smax = smax, smin  # swap if user accidentally inverted

        # Choose values to display
        size_values = [smin] if np.isclose(smin, smax) else list(np.linspace(smin, smax, max(2, size_legend_steps)))

        # 2) Reuse PX-computed sizing parameters so legend matches map
        main_trace = next((t for t in fig.data if t.type == "scattermapbox"), None)
        if main_trace and hasattr(main_trace, "marker") and hasattr(main_trace.marker, "sizeref"):
            computed_sizeref = main_trace.marker.sizeref
            sizemode = getattr(main_trace.marker, "sizemode", "diameter")
            sizemin = getattr(main_trace.marker, "sizemin", None)
        else:
            # Fallback if not found (rare)
            computed_sizeref = None
            sizemode = "diameter"
            sizemin = None

        # 3) Safe coordinates for legend-only traces
        if center_dict is not None:
            lat0, lon0 = center_dict["lat"], center_dict["lon"]
        else:
            # Fallback to first row coords (required even if legendonly)
            lat0 = float(plotting_df['Latitude'].iloc[0])
            lon0 = float(plotting_df['Longitude'].iloc[0])

        # 4) Add one trace per size value (legend-only)
        for v in size_values:
            fig.add_trace(go.Scattermapbox(
                lat=[lat0], lon=[lon0],
                mode="markers",
                marker=dict(
                    size=v,                        # IMPORTANT: pass the DATA value, not pixels
                    sizemode=sizemode,
                    sizeref=computed_sizeref,     # reuse mapping from main trace
                    sizemin=sizemin,
                    color="blue"
                ),
                name=f"{v:.2f}",
                legendgroup="size-legend",
                showlegend=True,
                visible="legendonly",
                hoverinfo="skip"
            ))

        # 5) Make legend respect per-trace marker sizes
        fig.update_layout(
            legend=dict(
                title=f"{size_column}",
                itemsizing="trace",            # <-- KEY so the legend icon size matches the trace
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.6)"
            )
        )

        # Optional: ensure the size legend entries appear after color categories
        fig.update_traces(legendrank=100, selector=dict(legendgroup="size-legend"))
      

    return fig

def map_timeseries_clickable_plot(
    obs_df,
    year,
    exceedance_threshold,
    cams_dust_threshold,
    # --- NEW: column parameters (no more fixed names) ---
    station_col='Samplingpoint',
    time_col='Start',
    lat_col='Latitude',
    lon_col='Longitude',
    value_col='observed_PM10',
    # --- Color & size controls ---
    cmap_name='plasma',           # any Matplotlib colormap name
    radius_range=(6, 16),         # (min_px, max_px)
    # --- Legend controls ---
    legend_title=None,            # defaults to f"Average {value_col}"
    legend_bins=5,                # number of bins (discrete steps to represent the continuous colormap)
    legend_round=1,               # decimals in legend labels
    legend_position='bottomright' # any Leaflet corner: 'topleft', 'topright', 'bottomleft', 'bottomright'
):
    """
    Build an ipyleaflet map with station markers colored & sized by the per-station
    average of `value_col`. Clicking a marker renders that station's time series.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Observation dataframe containing at least the columns specified by
        station_col, time_col, lat_col, lon_col, value_col.
    year : int
        Year passed to plot_station_timeseries (no filtering here).
    exceedance_threshold : float
        Passed to plot_station_timeseries.
    cams_dust_threshold : float
        Passed to plot_station_timeseries.
    station_col, time_col, lat_col, lon_col, value_col : str
        Column names to use for station ID, timestamp, latitude, longitude, and measured value.
    cmap_name : str
        Matplotlib colormap name for coloring markers.
    radius_range : (int, int)
        Pixel radius range for marker sizes (min, max).
    legend_title : str or None
        Title for the legend. If None, uses f"Average {value_col}".
    legend_bins : int
        Number of discrete color bins used to represent the continuous colormap.
    legend_round : int
        Number of decimals to format legend bin labels.
    legend_position : str
        Position of the legend on the map.
    """

    df = obs_df.copy()
    # Parse time column if present
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # Coordinates per station (first non-null)
    coords = (
        df[[station_col, lat_col, lon_col]]
        .dropna(subset=[lat_col, lon_col])
        .drop_duplicates(subset=[station_col])
        .set_index(station_col)
    )

    # Per-station average of the value_col (for marker color & size)
    avg_col_name = f"avg_{value_col}"
    avg_vals = (
        df.groupby(station_col)[value_col]
          .mean()
          .rename(avg_col_name)
          .to_frame()
    )

    summary = (
        avg_vals.join(coords, how='inner')
                .dropna(subset=[lat_col, lon_col])
                .reset_index()
    )

    if summary.empty:
        return VBox([HTML("<b>Error: No stations with coordinates and values.</b>")])

    # -----------------------------
    # Helpers for color/size scaling
    # -----------------------------
    def color_for_value(val, vmin, vmax, cmap_name='plasma'):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap_name)
        return mcolors.to_hex(cmap(norm(val)))

    def radius_for_value(val, vmin, vmax, rmin=6, rmax=16):
        return float(np.interp(val, [vmin, vmax], [rmin, rmax]))

    # Scale bounds (across stations' averages)
    vmin = float(summary[avg_col_name].min())
    vmax = float(summary[avg_col_name].max())

    # -----------------------------
    # Build the map (centered over median coords)
    # -----------------------------
    center_lat = float(summary[lat_col].median())
    center_lon = float(summary[lon_col].median())

    m = Map(center=(center_lat, center_lon), zoom=6, basemap={'url': 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'})
    m.add_control(LayersControl())

    # Output area for the time series below the map
    plot_out = Output()
    desc = HTML("<b>Click a station marker</b> to load its time series below.")

    # -----------------------------
    # Color legend (discrete bins from the continuous colormap)
    # -----------------------------
    legend_title = legend_title or f"Average {value_col}"
    # Bin edges & labels
    edges = np.linspace(vmin, vmax, legend_bins + 1)
    # Use midpoints to sample the colormap
    mids = (edges[:-1] + edges[1:]) / 2.0

    legend_items = {}
    for i, mid in enumerate(mids):
        label = f"{edges[i]:.{legend_round}f}–{edges[i+1]:.{legend_round}f}"
        legend_items[label] = color_for_value(mid, vmin, vmax, cmap_name=cmap_name)

    legend = LegendControl(legend_items, title=legend_title, position=legend_position)
    m.add(legend)

    # -----------------------------
    # Add markers (click → plot_station_timeseries)
    # -----------------------------
    rmin, rmax = radius_range
    for _, row in summary.iterrows():
        sp  = row[station_col]
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        val = float(row[avg_col_name])

        color  = color_for_value(val, vmin, vmax, cmap_name=cmap_name)
        radius = int(radius_for_value(val, vmin, vmax, rmin=rmin, rmax=rmax))

        marker = CircleMarker(
            location=(lat, lon),
            radius=radius,
            color=color,
            fill_color=color,
            fill_opacity=0.75,
            stroke=False
        )

        # Use a factory to bind the current station to the click handler
        def create_on_click_callback(current_sp):
            def on_click_callback(**kwargs):
                with plot_out:
                    plot_out.clear_output(wait=True)
                    try:
                        # Try to pass value_col if the function supports it
                        try:
                            fig_ts, axs = plot_station_timeseries(
                                station_name=current_sp,
                                obs_df=df,
                                year=year,
                                exceedance_threshold=exceedance_threshold,
                                cams_dust_threshold=cams_dust_threshold,
                                value_col=value_col,  # NEW: hand over the chosen column
                                figsize=(12, 5),
                            )
                        except TypeError:
                            # Fallback if plot_station_timeseries doesn't accept value_col
                            fig_ts, axs = plot_station_timeseries(
                                station_name=current_sp,
                                obs_df=df,
                                year=year,
                                exceedance_threshold=exceedance_threshold,
                                cams_dust_threshold=cams_dust_threshold,
                                figsize=(12, 5),
                            )
                        plt.show()
                    except Exception as e:
                        print(f"Cannot render station '{current_sp}': {e}")
            return on_click_callback

        marker.on_click(create_on_click_callback(sp))
        m.add_layer(marker)

    # -----------------------------
    # Display
    # -----------------------------
    return VBox([desc, m, plot_out])

def until_check():
    return "util.py has been imported"
