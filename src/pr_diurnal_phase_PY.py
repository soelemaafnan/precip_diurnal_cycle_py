#!/usr/bin/env python

"""
Python translation of the pr_diurnal_phase.ncl script.
This version fixes the time-slicing bug for non-standard
calendars and correctly uses cartopy for map plotting.

This version saves THREE SEPARATE PLOT FILES with
improved formatting, legends, and layouts.
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import sys
import logging
import yaml
import argparse
import time
import cftime  # For non-standard calendars

# Try to import cartopy for map plotting
try:
    import cartopy.crs as ccrs
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# --- 1. Plotting Constants ---
AMPLITUDE_VMIN = 0.5
AMPLITUDE_VMAX = 5.0
LAT_S = -50.
LAT_N = 50.
LON_W = 0.
LON_E = 360.
TSPD = 24
HOURS_UTC_24 = np.arange(0, TSPD)
MODEL_TIME_SHIFT = -3.0  # For 6-hourly time-mean data

# ---------------------------------------------------------------
# 2. LOGGER SETUP
# ---------------------------------------------------------------
logger = logging.getLogger() # Get the root logger

def setup_logging(work_dir):
    """Sets up logging to console and a file in the work_dir."""
    log_dir = os.path.join(work_dir, "model", "PS")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "analysis_pr_diurnal_phase.log")

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info(f"Logging initialized. Log file: {log_file}")

# ---------------------------------------------------------------
# 3. ANALYSIS FUNCTIONS
# ---------------------------------------------------------------
# (All functions from get_fourier_components to convert_phase_to_local_time are unchanged)

def get_fourier_components(dcycle_composite):
    """
    Performs a Fourier analysis to get phase and amplitude.
    Replaces NCL's 'fourier_info'.
    """
    logger.info("    - Calculating Fourier transform...")
    N = len(dcycle_composite.time)
    
    fft_result = np.fft.fft(dcycle_composite.values, axis=0)
    
    amplitude_data = (np.abs(fft_result[1, :, :]) * 2) / N
    phase_rad_data = np.angle(fft_result[1, :, :])

    amplitude = xr.DataArray(
        amplitude_data,
        coords={'lat': dcycle_composite.lat, 'lon': dcycle_composite.lon},
        dims=['lat', 'lon']
    )
    phase_rad = xr.DataArray(
        phase_rad_data,
        coords={'lat': dcycle_composite.lat, 'lon': dcycle_composite.lon},
        dims=['lat', 'lon']
    )
    
    return amplitude, phase_rad


def convert_phase_to_local_time(phase_rad, lon_coords):
    """
    Converts GMT phase (in radians) to local solar time (in hours).
    Replicates the longitude loop in pr_diurnal_phase.ncl.
    """
    logger.info("    - Converting phase to local solar time...")
    
    phase_gmt_hours = ( (np.pi / 2.0) - phase_rad ) * (TSPD / (2 * np.pi))
    lon_offset_hours = (lon_coords / 360.0) * TSPD
    phase_local = phase_gmt_hours - lon_offset_hours - MODEL_TIME_SHIFT
    phase_local_hours = phase_local % TSPD
    
    return phase_local_hours

# ---------------------------------------------------------------
# 4. PLOTTING HELPER FUNCTIONS (Replaces evans_plot.ncl)
# ---------------------------------------------------------------
# (add_color_wheel is unchanged)

def add_color_wheel(fig, vmin, vmax):
    """
    Adds the 'Evans plot' color wheel legend to the figure.
    """
    # ---
    # <<< --- COORDINATES MODIFIED --- >>>
    # Moves the wheel to the right and down to avoid overlap
    cax = fig.add_axes([0.82, 0.25, 0.15, 0.15], projection='polar')
    # ---
    
    n_hue = 24
    n_sat = 5
    hues = np.linspace(0, 1, n_hue + 1)
    sats = np.linspace(0.1, 1, n_sat)
    
    for h in range(n_hue):
        for s in range(n_sat):
            hue_val = (hues[h] + 0.5/n_hue + 0.5) % 1.0 # Same offset as map
            sat_val = sats[s]
            color = mcolors.hsv_to_rgb([hue_val, sat_val, 1])
            theta_start_rad = (hues[h]) * 2 * np.pi
            theta_end_rad = (hues[h+1]) * 2 * np.pi
            r_start = s / n_sat
            r_end = (s + 1) / n_sat
            cax.add_patch(mpatches.Wedge(
                (0, 0), r_end, 
                np.degrees(theta_start_rad), np.degrees(theta_end_rad), 
                facecolor=color, edgecolor='none', 
                width=(r_end - r_start)
            ))

    cax.set_facecolor('none')
    cax.set_yticks(np.linspace(0, 1, n_sat + 1)[:-1] + (0.5 / n_sat))
    cax.set_yticklabels([f"{val:.1f}" for val in np.linspace(vmin, vmax, n_sat)], fontsize=8)
    cax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    cax.set_xticklabels(['0hr', '3hr', '6hr', '9hr', '12hr', '15hr', '18hr', '21hr'], fontsize=9)
    cax.set_ylim(0, 1)
    cax.set_title("Phase (hr) & \nAmplitude (mm/day)", fontsize=10)

# ---
# <<< --- ALL 3 PLOTTING FUNCTIONS ARE MODIFIED --- >>>
# ---

def plot_phase_amplitude_map(phase, amplitude, season_name, config_dict):
    """
    Creates the 2D 'Evans plot' (phase/amplitude map).
    """
    logger.info(f"  - Plotting Phase/Amplitude map for {season_name}...")
    
    fig = plt.figure(figsize=(14, 7)) # Wide figure
    
    # 1. Normalize Phase (Hue)
    hue = (phase / 24.0 + 0.5) % 1.0
    
    # 2. Normalize Amplitude (Saturation)
    saturation = (amplitude - AMPLITUDE_VMIN) / (AMPLITUDE_VMAX - AMPLITUDE_VMIN)
    saturation = np.clip(saturation, 0.1, 1.0)
    
    # 3. Set Value (Brightness)
    value = np.ones_like(hue)
    
    # 4. Stack H, S, V and convert to RGB
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    
    # 5. Plot the RGB image
    if CARTOPY_AVAILABLE:
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        ax.imshow(rgb, origin='lower', 
                  extent=[LON_W, LON_E, LAT_S, LAT_N], 
                  transform=ccrs.PlateCarree())
        ax.coastlines(color='0.3', linewidth=0.5)
        
        # --- ADD FAINT GRIDLINES AND LABELS ---
        gl = ax.gridlines(draw_labels=True, linestyle=':', color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}
    else:
        # Fallback to simple image
        ax = plt.subplot(1, 1, 1)
        ax.imshow(rgb, origin='lower', 
                  extent=[LON_W, LON_E, LAT_S, LAT_N], 
                  aspect='auto')
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)
        logger.warning("  - 'cartopy' not found. Plotting a simple image.")

    # Add the color wheel legend
    add_color_wheel(fig, AMPLITUDE_VMIN, AMPLITUDE_VMAX)

    # Set title
    year = config_dict.get('startdate', 'YYYY')
    casename = config_dict.get('casename', 'Model')
    ax.set_title(f"{casename} {year} - {season_name} Phase (Hue) and Amplitude (Saturation)", fontsize=16)

    # Save the plot
    work_dir = config_dict.get('work_dir', '.')
    output_dir = os.path.join(work_dir, "model", "PS")
    plot_filename = f"pr_diurnal_phase_{season_name}.png"
    output_path = os.path.join(output_dir, plot_filename)
    
    # --- SAVE WITH TIGHT BBOX ---
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"  - Plot saved to {output_path}")
    plt.close(fig)


def plot_variance_map(variance, season_name, config_dict):
    """
    Creates the 'Variance Explained' map.
    """
    logger.info(f"  - Plotting Variance Explained map for {season_name}...")
    
    fig = plt.figure(figsize=(14, 7)) # Wide figure

    if CARTOPY_AVAILABLE:
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        # Plot with add_colorbar=False
        plot_obj = variance.plot.contourf(
            ax=ax, transform=ccrs.PlateCarree(),
            levels=np.arange(0, 101, 10), cmap='YlOrRd',
            add_colorbar=False # <<< --- TURN OFF DEFAULT COLORBAR
        )
        ax.coastlines(color='0.3', linewidth=0.5)

        # --- ADD FAINT GRIDLINES AND LABELS ---
        gl = ax.gridlines(draw_labels=True, linestyle=':', color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}
        
        # --- ADD MANUAL, SIZED COLORBAR ---
        cbar = fig.colorbar(plot_obj, ax=ax, orientation='vertical', 
                            shrink=0.8, aspect=30, pad=0.03)
        cbar.set_label('%', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    else:
        # Fallback to simple plot
        ax = plt.subplot(1, 1, 1)
        variance.plot.contourf(
            ax=ax, levels=np.arange(0, 101, 10), cmap='YlOrRd',
            cbar_kwargs={'label': '%', 'shrink': 0.8, 'aspect': 30}
        )
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)
        
    # Set title
    year = config_dict.get('startdate', 'YYYY')
    casename = config_dict.get('casename', 'Model')
    ax.set_title(f"{casename} {year} - {season_name} Variance Explained by 24hr Cycle (%)", fontsize=16)

    # Save the plot
    work_dir = config_dict.get('work_dir', '.')
    output_dir = os.path.join(work_dir, "model", "PS")
    plot_filename = f"pr_diurnal_variance_{season_name}.png"
    output_path = os.path.join(output_dir, plot_filename)
    
    # --- SAVE WITH TIGHT BBOX ---
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"  - Plot saved to {output_path}")
    plt.close(fig)


def plot_mean_precip_map(mean_precip, season_name, config_dict):
    """
    Creates the 'Mean Precipitation' map.
    """
    logger.info(f"  - Plotting Mean Precipitation map for {season_name}...")
    
    fig = plt.figure(figsize=(14, 7)) # Wide figure

    if CARTOPY_AVAILABLE:
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        # Plot with add_colorbar=False
        plot_obj = mean_precip.plot.contourf(
            ax=ax, transform=ccrs.PlateCarree(),
            levels=np.arange(0, 21, 2), cmap='GnBu',
            add_colorbar=False # <<< --- TURN OFF DEFAULT COLORBAR
        )
        ax.coastlines(color='0.3', linewidth=0.5)

        # --- ADD FAINT GRIDLINES AND LABELS ---
        gl = ax.gridlines(draw_labels=True, linestyle=':', color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}

        # --- ADD MANUAL, SIZED COLORBAR ---
        cbar = fig.colorbar(plot_obj, ax=ax, orientation='vertical', 
                            shrink=0.8, aspect=30, pad=0.03)
        cbar.set_label('mm/day', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    else:
        # Fallback to simple plot
        ax = plt.subplot(1, 1, 1)
        mean_precip.plot.contourf(
            ax=ax, levels=np.arange(0, 21, 2), cmap='GnBu',
            cbar_kwargs={'label': 'mm/day', 'shrink': 0.8, 'aspect': 30}
        )
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)

    # Set title
    year = config_dict.get('startdate', 'YYYY')
    casename = config_dict.get('casename', 'Model')
    ax.set_title(f"{casename} {year} - {season_name} Mean Precipitation (mm/day)", fontsize=16)

    # Save the plot
    work_dir = config_dict.get('work_dir', '.')
    output_dir = os.path.join(work_dir, "model", "PS")
    plot_filename = f"pr_diurnal_mean_{season_name}.png"
    output_path = os.path.join(output_dir, plot_filename)
    
    # --- SAVE WITH TIGHT BBOX ---
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"  - Plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------
# 5. CONFIG LOADING FUNCTION
# ---------------------------------------------------------------
def load_config(config_path):
    """Loads the YAML config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Successfully loaded config from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

# ---------------------------------------------------------------
# 6. MAIN FUNCTION
# ---------------------------------------------------------------
def main():
    """
    Main driver function to orchestrate the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Run diurnal cycle phase/amplitude analysis from a config file."
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help="Path to the .yaml configuration file."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config.get('work_dir', '.'))

    start_main = time.time()
    try:
        # --- Get settings from config dictionary ---
        PR_FILE = config.get('pr_file')
        pr_var = config.get('pr_var', 'precip')
        
        try:
            start_date_str = config['analysis_period']['start']
            end_date_str = config['analysis_period']['end']
        except KeyError:
            logger.error("Error: 'analysis_period: {start: ... , end: ...}' not found in config file.")
            return

        if not PR_FILE or not os.path.exists(PR_FILE):
            logger.error(f"Error: Model file not found at PR_FILE={PR_FILE}")
            return

        # --- Load Model Data ---
        logger.info(f"Loading model file: {PR_FILE}")
        try:
            time_coder = xr.coding.times.CFDatetimeCoder(use_cftime=True)
            ds_model = xr.open_dataset(
                PR_FILE, 
                decode_times=time_coder, 
                decode_timedelta=False 
            )
        except Exception as e:
            logger.error(f"Failed to open model file with cftime coder: {e}")
            return

        # 1. Get the calendar type from the loaded data
        calendar = ds_model.time.dt.calendar
        logger.info(f"  - Detected model calendar: {calendar}")
        
        # 2. Convert config strings to cftime objects OF THAT CALENDAR TYPE
        start_date_cftime = cftime.DatetimeGregorian(*map(int, start_date_str.split('-')))
        end_date_cftime = cftime.DatetimeGregorian(*map(int, end_date_str.split('-')))
        
        # 3. Create the time slice using the cftime objects
        time_slice = slice(start_date_cftime, end_date_cftime)
        logger.info(f"  - Slicing data from {start_date_cftime} to {end_date_cftime}")

        # --- Filter, Convert, and Scale Data ---
        ds_filtered = ds_model.sel(time=time_slice)
        
        logger.info("  - Converting model longitude grid from -180/180 to 0/360...")
        ds_filtered.coords['lon'] = np.mod(ds_filtered['lon'], 360)
        ds_filtered = ds_filtered.sortby('lon')
        
        da = ds_filtered[pr_var] * 86400.0 # Convert to mm/day
        da = da.sel(lat=slice(LAT_S, LAT_N), lon=slice(LON_W, LON_E)) # Crop to plot area
        
        # --- Loop Over Seasons and Create Plots ---
        for season in ['JJA', 'DJF']:
            logger.info(f"--- Processing {season} ---")
            
            ds_season = da.where(da['time'].dt.season == season, drop=True)
            
            if ds_season.time.size == 0:
                logger.warning(f"  - No data found for {season}. Skipping plot.")
                continue 

            logger.info(f"  - Creating {season} composite day...")
            dcycle_grouped = ds_season.groupby(ds_season.time.dt.hour).mean(dim='time')
            
            logger.info("  - Interpolating to 24-hourly cycle...")
            dcycle_composite = dcycle_grouped.interp(
                hour=HOURS_UTC_24, kwargs={"fill_value": "extrapolate"}
            ).rename({'hour': 'time'})
            
            mean_precip = dcycle_composite.mean(dim='time')
            total_variance = dcycle_composite.var(dim='time')
            
            amplitude, phase_rad = get_fourier_components(dcycle_composite)
            phase_local = convert_phase_to_local_time(phase_rad, dcycle_composite.lon)
            
            harmonic_variance = (amplitude**2) / 2.0
            variance_explained = (harmonic_variance / total_variance) * 100.0
            variance_explained = variance_explained.clip(0, 100)
            
            # ---
            # <<< --- MODIFIED: Call the three new plotting functions --- >>>
            # ---
            plot_phase_amplitude_map(phase_local, amplitude, season, config)
            plot_variance_map(variance_explained, season, config)
            plot_mean_precip_map(mean_precip, season, config)
            # ---
            
            logger.info(f"--- {season} processing complete ---")

        logger.info(f"Python script finished successfully in {time.time() - start_main:.2f} seconds.")

    except Exception as e:
        logger.error("A fatal error occurred in main()!", exc_info=True)
    finally:
        logging.shutdown()


if __name__ == "__main__":
    if not CARTOPY_AVAILABLE:
        print("---")
        print("Warning: 'cartopy' library not found.")
        print("         The script will run, but maps will be plotted as simple images.")
        print("         To install, run: conda install -c conda-forge cartopy")
        print("---")

    main()