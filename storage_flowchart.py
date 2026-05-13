import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 20))
ax.set_xlim(0, 10)
ax.set_ylim(0, 28)
ax.axis('off')

# Color scheme
color_input = '#FFE5B4'      # Peach - Input/Download
color_storage = '#B4D7FF'   # Light Blue - Storage
color_process = '#C8E6C9'   # Light Green - Processing
color_output = '#F8BBD0'    # Light Pink - Output
color_decision = '#FFF9C4'  # Light Yellow - Decision

def draw_box(ax, x, y, width, height, text, color, fontsize=9, fontweight='normal'):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            fontweight=fontweight, wrap=True)

def draw_diamond(ax, x, y, width, height, text, color, fontsize=8):
    """Draw a diamond shape for decisions"""
    points = np.array([[x, y+height/2], [x+width/2, y], 
                       [x, y-height/2], [x-width/2, y]])
    diamond = mpatches.Polygon(points, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, label='', curve=0):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black',
                           connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x+0.2, mid_y+0.2, label, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============= TITLE =============
ax.text(5, 27, 'CAMS Dust Deduction Tool: Storage & Processing Flow', 
        ha='center', fontsize=16, fontweight='bold')

# ============= SECTION 1: DATA DOWNLOAD =============
ax.text(0.5, 25.5, 'STAGE 1: DATA DOWNLOAD & CONFIGURATION', 
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray'))

draw_box(ax, 2.5, 24.5, 2, 0.8, 'User Settings:\n• YEAR, POLLUTANT\n• Countries, Dataset', color_input)
draw_box(ax, 2.5, 23.2, 2, 0.8, 'Configuration:\n• USE_GOOGLE_DRIVE\n• EEA_temporal_flag', color_input)
draw_arrow(ax, 2.5, 24.1, 2.5, 23.6)

draw_diamond(ax, 2.5, 22, 1.5, 1, 'USE_GOOGLE\nDRIVE?', color_decision, fontsize=7)
draw_arrow(ax, 3.25, 22, 5.5, 22, 'YES', curve=0)
draw_arrow(ax, 1.75, 22, 1.5, 21.2, 'NO', curve=0)

# Google Drive path
draw_box(ax, 6.5, 22, 2.5, 0.8, 'Google Drive Path:\n/MyDrive/CAMS_Tool_output', color_storage)
draw_arrow(ax, 5.5, 22, 7.75, 22)

# Local path
draw_box(ax, 1.5, 20.4, 2, 0.8, 'Local Path:\n./project_dir', color_storage)
draw_arrow(ax, 2.5, 21.5, 1.5, 20.8)

# ============= SECTION 2: EEA DATA =============
ax.text(0.5, 19.5, 'STAGE 2: EEA OBSERVATION DATA', 
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray'))

draw_diamond(ax, 2.5, 18.5, 1.5, 1, 'DOWNLOAD\nEEA?', color_decision, fontsize=7)
draw_arrow(ax, 3.25, 18.5, 5.5, 18.5, 'YES', curve=0)
draw_arrow(ax, 1.75, 18.5, 0.8, 18.5, 'NO', curve=0)

# Download EEA
draw_box(ax, 6.5, 18.5, 2.5, 0.8, 'EEA API Request:\nParquetFile/async', color_input)
draw_arrow(ax, 5.5, 18.5, 7.75, 18.5)

# Load existing
draw_box(ax, 0.8, 17.8, 1.5, 0.6, 'Load Existing\nZip File', color_input)
draw_arrow(ax, 0.8, 18.1, 0.8, 18)

# Both converge to extraction
draw_box(ax, 4.5, 16.8, 3, 0.8, 'Extract to:\nEEA_PM10/{YEAR}/{DATASET}/{TEMPORAL}/', color_storage)
draw_arrow(ax, 6.5, 18.1, 5.5, 17.2)
draw_arrow(ax, 0.8, 17.5, 3, 17.2)

# Store zip
draw_box(ax, 8, 16.8, 2, 0.8, 'Store ZIP:\n[gdrive_output_dir]/\nEEA_PM10/hour/\n[dataset]_[year].zip', color_storage)
draw_arrow(ax, 6, 18.5, 7.2, 17.2)

# ============= SECTION 3: CAMS DATA =============
ax.text(0.5, 15.8, 'STAGE 3: CAMS DUST DATA', 
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray'))

draw_diamond(ax, 2.5, 15, 1.5, 1, 'DOWNLOAD\nCAMS?', color_decision, fontsize=7)
draw_arrow(ax, 3.25, 15, 5.5, 15, 'YES', curve=0)
draw_arrow(ax, 1.75, 15, 0.8, 15, 'NO', curve=0)

# Download CAMS
draw_box(ax, 6.5, 15, 2.5, 0.8, 'CDSAPI Request:\n5 Quarterly ZIP files\n(Q1-Q4 + Dec prev yr)', color_input)
draw_arrow(ax, 5.5, 15, 7.75, 15)

# Load existing
draw_box(ax, 0.8, 14.3, 1.5, 0.6, 'Load Existing\nZip Files', color_input)
draw_arrow(ax, 0.8, 14.5, 0.8, 14.6)

# Extraction
draw_box(ax, 4.5, 13.2, 3, 0.8, 'Extract ALL to:\n./IRA_dust/', color_storage)
draw_arrow(ax, 6.5, 14.6, 5.5, 13.6)
draw_arrow(ax, 0.8, 14, 3, 13.6)

# Store files
draw_box(ax, 8, 13.2, 2, 0.8, 'Store ZIP Files:\n[gdrive_output_dir]/\nIRA_dust/\nCAMS_IRA_*.zip', color_storage)
draw_arrow(ax, 6, 15, 7.2, 13.6)

# ============= SECTION 4: DATA LOADING =============
ax.text(0.5, 12, 'STAGE 4: DATA LOADING & PREPARATION', 
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray'))

draw_box(ax, 2, 11.2, 2.5, 0.8, 'Load EEA Parquet\nFiles (All countries)', color_process)
draw_box(ax, 5.5, 11.2, 2.5, 0.8, 'Load CAMS NetCDF\nFiles (xarray)', color_process)
draw_box(ax, 9, 11.2, 2, 0.8, 'Normalize time\nto UTC', color_process)

draw_arrow(ax, 4.5, 13.2, 2, 11.6)
draw_arrow(ax, 4.5, 13.2, 5.5, 11.6)
draw_arrow(ax, 8, 13.2, 9, 11.6)

# ============= SECTION 5: FILTERING =============
ax.text(0.5, 10.2, 'STAGE 5: DATA FILTERING & COVERAGE CHECK', 
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray'))

draw_box(ax, 2.5, 9.4, 3, 0.8, 'calculate_data_coverage():\nMin coverage >= 75%', color_process)
draw_box(ax, 6.5, 9.4, 3, 0.8, 'filter_daily_by_coverage():\nKeep qualifying stations', color_process)

draw_arrow(ax, 2, 10.8, 2.5, 9.8)
draw_arrow(ax, 5.5, 10.8, 6.5, 9.8)

# ============= SECTION 6: DUST FLAGGING =============
ax.text(0.5, 8.4, 'STAGE 6: DUST FLAGGING & INTERPOLATION', 
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray'))

draw_box(ax, 2.5, 7.6, 3.5, 0.8, 'add_cams_daily_dust_by_station():\nInterpolate CAMS @ station coords', color_process)
draw_box(ax, 7, 7.6, 2.5, 0.8, 'Apply dust threshold:\nCAMS dust > 5 µg/m³', color_process)

draw_arrow(ax, 2.5, 9, 2.5, 8)
draw_arrow(ax, 6.5, 9, 7, 8)

# ============= SECTION 7: EXCEEDANCE DETECTION =============
ax.text(0.5, 6.6, 'STAGE 7: PM10 EXCEEDANCE DETECTION', 
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray'))

draw_diamond(ax, 2.5, 5.8, 1.5, 1, 'Dust Day\n& Exceedance\n>50µg/m³?', color_decision, fontsize=7)
draw_arrow(ax, 2.5, 7.2, 2.5, 6.3, curve=0)

draw_arrow(ax, 3.25, 5.8, 5, 5.8, 'YES', curve=0)
draw_arrow(ax, 1.75, 5.8, 1.2, 5.8, 'NO', curve=0)

draw_box(ax, 5.5, 5.8, 1.8, 0.6, 'Calculate dust\ncontribution', color_process)
draw_box(ax, 1.2, 5.1, 1.5, 0.6, 'Skip', color_process)

# ============= SECTION 8: BASELINE COMPUTATION =============
ax.text(0.5, 4.4, 'STAGE 8: BASELINE & DUST CONTRIBUTION', 
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray'))

draw_box(ax, 2.5, 3.6, 3.5, 0.8, 'compute_station_baseline():\nMedian of 30 non-dust days', color_process)
draw_box(ax, 6.5, 3.6, 3, 0.8, 'Dust contribution =\nObserved - Baseline', color_process)

draw_arrow(ax, 5.5, 5.5, 2.5, 4)
draw_arrow(ax, 5.5, 5.5, 6.5, 4)

# ============= SECTION 9: OUTPUT & VISUALIZATION =============
ax.text(0.5, 2.6, 'STAGE 9: RESULTS & VISUALIZATION', 
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray'))

draw_box(ax, 1.5, 1.8, 2.2, 0.8, 'Results Table:\nStation, Date, PM10,\nDust, Contribution', color_output)
draw_box(ax, 4.5, 1.8, 2.2, 0.8, 'Time Series Plot:\nObserved vs Corrected\nPM10 + CAMS dust', color_output)
draw_box(ax, 7.5, 1.8, 2.2, 0.8, 'Interactive Map:\nStation markers\nby dust contribution', color_output)

draw_arrow(ax, 2.5, 3.2, 1.5, 2.2)
draw_arrow(ax, 5, 3.2, 4.5, 2.2)
draw_arrow(ax, 6.5, 3.2, 7.5, 2.2)

# Export options
draw_box(ax, 2.5, 0.6, 3, 0.8, 'Export:\nCSV, PNG, Interactive HTML', color_output)
draw_arrow(ax, 1.5, 1.4, 2.5, 1)
draw_arrow(ax, 4.5, 1.4, 2.5, 1)
draw_arrow(ax, 7.5, 1.4, 2.5, 1)

# Storage location for outputs
draw_box(ax, 7.5, 0.6, 2.2, 0.8, 'Store outputs in:\n[gdrive_output_dir]\nif USE_GOOGLE_DRIVE', color_storage)

plt.tight_layout()
plt.savefig('storage_flowchart.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Flowchart saved as 'storage_flowchart.png'")
plt.close()

# ============= SECOND FIGURE: STORAGE HIERARCHY =============
fig, ax = plt.subplots(1, 1, figsize=(14, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

ax.text(5, 19.5, 'File Storage Hierarchy & Options', 
        ha='center', fontsize=16, fontweight='bold')

# Root level
draw_box(ax, 5, 18.5, 3, 0.7, 'Project Root / Google Drive', color_storage)

# Level 1
draw_box(ax, 2, 17.3, 2.5, 0.7, 'EEA_PM10/', color_storage)
draw_box(ax, 5, 17.3, 2.5, 0.7, 'IRA_dust/', color_storage)
draw_box(ax, 8, 17.3, 2.5, 0.7, 'Results/', color_output)

draw_arrow(ax, 4.2, 18.1, 2.8, 17.7, curve=0.3)
draw_arrow(ax, 5, 18.1, 5, 17.7, curve=0)
draw_arrow(ax, 5.8, 18.1, 7.2, 17.7, curve=-0.3)

# EEA branch
draw_box(ax, 2, 15.8, 2.5, 0.7, '{YEAR}/', color_storage)
draw_arrow(ax, 2, 16.95, 2, 16.2)

draw_box(ax, 2, 14.3, 2.5, 0.7, '{DATASET}/ (E1a/E2a)', color_storage)
draw_arrow(ax, 2, 15.45, 2, 14.95)

draw_box(ax, 2, 12.8, 2.5, 0.7, '{TEMPORAL}/ (hour/day)', color_storage)
draw_arrow(ax, 2, 13.95, 2, 13.45)

draw_box(ax, 2, 11.3, 2.5, 0.7, 'Parquet Files\n(*.parquet)', color_process)
draw_arrow(ax, 2, 12.45, 2, 11.95)

draw_box(ax, 0.2, 10.3, 1.5, 0.7, 'ZIP Archive:\n[dataset]_[year].zip', color_storage)
draw_arrow(ax, 1.2, 11.3, 0.8, 10.7)

# CAMS branch
draw_box(ax, 5, 15.8, 2.5, 0.7, 'NetCDF Files\n(*.nc)', color_process)
draw_arrow(ax, 5, 16.95, 5, 16.2)

draw_box(ax, 5, 14.3, 2.5, 0.7, 'Organized by\nmonth/quarter', color_storage)
draw_arrow(ax, 5, 15.45, 5, 14.95)

draw_box(ax, 3.2, 12.8, 2.5, 0.7, 'ZIP Archives:\nCAMS_IRA_*.zip\n(5 files/year)', color_storage)
draw_arrow(ax, 4.2, 14.3, 3.8, 13.45)

# Output branch
draw_box(ax, 8, 15.8, 2.5, 0.7, 'Results Files\n(.csv/.png/.html)', color_output)
draw_arrow(ax, 8, 16.95, 8, 16.2)

# ============= KEY FUNCTIONS SIDE =============
ax.text(0.3, 8.8, 'KEY STORAGE FUNCTIONS', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))

functions_text = """
1. calculate_data_coverage():
   • Filters by coverage % per station
   • Returns stations with ≥75% data

2. filter_daily_by_coverage():
   • Keeps all data for qualifying stations
   • Attaches coverage metrics

3. add_cams_daily_dust_by_station():
   • Interpolates CAMS @ station location
   • Vectorized time selection

4. compute_station_baseline():
   • Finds 15 non-dust days before & after
   • Calculates median baseline

5. plot_station_timeseries():
   • Visualizes PM10 vs Corrected
   • Shows dust days & thresholds

6. plot_exceedance_maps_discrete():
   • Maps station exceedances
   • Multiple colorbars options

7. map_timeseries_clickable_plot():
   • Interactive ipyleaflet map
   • Click markers for time series
"""

ax.text(0.3, 4.3, functions_text, fontsize=8, family='monospace',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.8))

# ============= STORAGE OPTIONS =============
ax.text(6, 8.8, 'STORAGE OPTIONS', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen'))

options_text = """
OPTION 1: LOCAL STORAGE
• USE_GOOGLE_DRIVE = False
• Path: ./project_dir/
• Suitable for: Small datasets, 
  personal workstations

OPTION 2: GOOGLE DRIVE
• USE_GOOGLE_DRIVE = True
• Path: /MyDrive/CAMS_Tool_output
• Suitable for: Collaborative work,
  large datasets, Colab notebooks

DATA FLOW:
1. Download → ZIP (temp)
2. Extract → Local folder
3. (Optional) Move ZIP to cloud
4. Process data in memory
5. Save results to cloud/local
6. Export visualizations
"""

ax.text(6, 4.3, options_text, fontsize=8, family='monospace',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#F0E6FF', alpha=0.8))

plt.tight_layout()
plt.savefig('storage_hierarchy.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Storage hierarchy diagram saved as 'storage_hierarchy.png'")
plt.close()

print("\n✓ Two flowchart PNG files created successfully!")
print("  1. storage_flowchart.png - Complete data processing pipeline")
print("  2. storage_hierarchy.png - File organization & storage options")
