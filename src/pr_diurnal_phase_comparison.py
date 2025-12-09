#!/usr/bin/env python

"""
Python translation of the pr_diurnal_phase.ncl script.
This version fixes the time-slicing bug for non-standard
calendars and correctly uses cartopy for map plotting.

This version saves THREE SEPARATE PLOT FILES with
improved formatting, legends, and layouts.

Fixes:
- Corrected Fourier phase calculation to match NCL (cosine peak).
- Corrected color wheel hue offset to 240 degrees to match plot image.
- Corrected color wheel to map both Saturation and Value (Brightness)
  to fix "washed out" colors.
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

# --- 1. Plotting & Analysis Constants ---
AMPLITUDE_VMIN = 0.5
AMPLITUDE_VMAX = 5.0
LAT_S = -50.
LAT_N = 50.
LON_W = 0.
LON_E = 360.
TSPD = 24 # Time steps per diurnal cycle (24 hours)
HOURS_UTC_24 = np.arange(0, TSPD) # For 1-hourly interpolation
MODEL_TIME_SHIFT = -3.0  # For 6-hourly time-mean model data
IMERG_TIME_SHIFT = 0.0   # IMERG 3-hourly data is 0, 3, 6... UTC

# ---
# <<< --- NEW HUE OFFSET TO MATCH NCL PLOT --- >>>
# ---
# NCL plot shows 0hr=Blue(0.66), 12hr=Yellow(0.16).
# This implies a (0.66) or 240-degree offset, not 180.
HUE_OFFSET = 0.66  # (240 / 360)
# ---

# ---------------------------------------------------------------
# 2. LOGGER SETUP
# ---------------------------------------------------------------
logger = logging.getLogger() # Get the root logger

def setup_logging(work_dir):
    """Sets up logging to console and a file in the work_dir."""
    log_dir = os.path.join(work_dir, "model", "PS")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "analysis_pr_diurnal_phase_comparison.log")

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

# ---
# <<< --- FOURIER ANALYSIS IS CORRECTED --- >>>
# ---
def get_fourier_components(dcycle_composite):
    """
    Performs a Fourier analysis to get phase and amplitude.
    Returns amplitude and PHASE IN DEGREES (like NCL).
    """
    logger.info("    - Calculating Fourier transform...")
    N = len(dcycle_composite.time) # Number of time steps (24)
    
    # Compute the Fast Fourier Transform along the time axis
    # Input must be (time, lat, lon) for this to result in (lat, lon)
    fft_result = np.fft.fft(dcycle_composite.values, axis=0)
    
    # Amplitude is (2 * abs(fft[1])) / N
    amplitude_data = (np.abs(fft_result[1, :, :]) * 2) / N
    
    # Phase is the angle of the complex number
    # We get phase in RADIANS, then convert to DEGREES
    phase_rad_data = np.angle(fft_result[1, :, :])
    phase_deg_data = np.degrees(phase_rad_data) # Convert to degrees

    amplitude = xr.DataArray(
        amplitude_data,
        coords={'lat': dcycle_composite.lat, 'lon': dcycle_composite.lon},
        dims=['lat', 'lon']
    )
    # Return the phase in DEGREES
    phase_deg = xr.DataArray(
        phase_deg_data,
        coords={'lat': dcycle_composite.lat, 'lon': dcycle_composite.lon},
        dims=['lat', 'lon']
    )
    
    return amplitude, phase_deg

# ---
# <<< --- PHASE CONVERSION IS CORRECTED --- >>>
# ---
def convert_phase_to_local_time(phase_deg, lon_coords, time_shift):
    """
    Converts GMT phase (in DEGREES) to local solar time (in hours).
    """
    logger.info("    - Converting phase to local solar time...")
    
    # 1. Convert phase from cosine degrees to GMT hours (0-24)
    # NCL: peak time in hours = phase_degrees / (360 / N_timesteps)
    phase_gmt_hours = phase_deg / (360.0 / TSPD) # phase_deg / 15.0
    
    # 2. Calculate the longitude time offset in hours
    lon_offset_hours = (lon_coords / 360.0) * TSPD
    
    # 3. Subtract longitude offset and model/imerg specific time shift
    phase_local = phase_gmt_hours - lon_offset_hours - time_shift
    
    # 4. Use modulo (%) to wrap the time around the 24-hour clock
    phase_local_hours = phase_local % TSPD
    
    return phase_local_hours

# ---------------------------------------------------------------
# 4. DATA LOADING AND PREPROCESSING FUNCTIONS
# ---------------------------------------------------------------
# (load_and_process_model_data and load_and_process_imerg_data are unchanged)

def load_and_process_model_data(config_dict):
    """Loads and processes the model precipitation data."""
    logger.info("--- Processing Model Data ---")
    
    pr_file = config_dict.get('pr_file')
    pr_var = config_dict.get('pr_var', 'pr')
    
    start_date_str = config_dict['analysis_period']['start']
    end_date_str = config_dict['analysis_period']['end']

    if not pr_file or not os.path.exists(pr_file):
        logger.error(f"Error: Model file not found at PR_FILE={pr_file}")
        sys.exit(1)

    logger.info(f"  - Loading model file: {pr_file}")
    try:
        time_coder = xr.coding.times.CFDatetimeCoder(use_cftime=True)
        ds_model = xr.open_dataset(
            pr_file, 
            decode_times=time_coder, 
            decode_timedelta=False 
        )
    except Exception as e:
        logger.error(f"Failed to open model file with cftime coder: {e}")
        sys.exit(1)

    try:
        date_type = ds_model.indexes['time'].date_type
        logger.info(f"  - Detected model calendar type: {date_type}")
        start_parts = list(map(int, start_date_str.split('-')))
        end_parts = list(map(int, end_date_str.split('-')))
        start_date_cftime = date_type(*start_parts)
        end_date_cftime = date_type(*end_parts)
        time_slice = slice(start_date_cftime, end_date_cftime)
        logger.info(f"  - Slicing model data from {start_date_cftime} to {end_date_cftime}")
    except Exception as e:
        logger.warning(f"  - Could not auto-detect calendar type ({e}). Falling back to string slicing.")
        time_slice = slice(start_date_str, end_date_str)

    ds_filtered = ds_model.sel(time=time_slice)
    
    logger.info("  - Converting model longitude grid from -180/180 to 0/360...")
    ds_filtered.coords['lon'] = np.mod(ds_filtered['lon'], 360)
    ds_filtered = ds_filtered.sortby('lon')
    
    da = ds_filtered[pr_var] * 86400.0 # Convert to mm/day
    da = da.sel(lat=slice(LAT_S, LAT_N), lon=slice(LON_W, LON_E)) # Crop to plot area
    
    # Ensure standard dimension order (time, lat, lon)
    da = da.transpose('time', 'lat', 'lon')
    
    return da

def load_and_process_imerg_data(config_dict, season):
    """Loads and processes the IMERG precipitation data for a specific season."""
    logger.info(f"--- Processing IMERG Data for {season} ---")
    
    base_dir = config_dict.get('processed_satellite_dir')
    if not base_dir:
        logger.error("Error: 'processed_satellite_dir' not found in config.yaml")
        return None
        
    imerg_file = os.path.join(base_dir, f"TRMM_{season}.nc") 
    imerg_pr_var = "precip" 
    
    if not imerg_file or not os.path.exists(imerg_file):
        logger.warning(f"Warning: IMERG file not found for {season} at {imerg_file}")
        return None 

    logger.info(f"  - Loading IMERG file: {imerg_file}")
    try:
        ds_imerg = xr.open_dataset(imerg_file)
        ds_imerg[imerg_pr_var] = ds_imerg[imerg_pr_var] * 24.0
    except Exception as e:
        logger.error(f"Failed to open IMERG file for {season}: {e}")
        return None

    da_imerg = ds_imerg[imerg_pr_var]
    da_imerg = da_imerg.sel(lat=slice(LAT_S, LAT_N), lon=slice(LON_W, LON_E)) # Crop
    
    logger.info("  - Transposing IMERG dimensions to (time, lat, lon)...")
    da_imerg = da_imerg.transpose('time', 'lat', 'lon')
    
    return da_imerg


# ---------------------------------------------------------------
# 5. PLOTTING HELPER FUNCTIONS
# ---------------------------------------------------------------

# ---
# <<< --- COLOR WHEEL IS CORRECTED (Saturation & Value) --- >>>
# ---
def add_color_wheel(fig, vmin, vmax, subplot_position=[0.85, 0.35, 0.13, 0.3]):
    """
    Adds the 'Evans plot' color wheel legend to the figure at a specified position.
    This now replicates the NCL script's piecewise ramp for Saturation and Value.
    """
    cax = fig.add_axes(subplot_position, projection='polar')
    
    n_hue = 24
    n_sat = 8 # Use 8 levels to match the NCL "gt 5" logic
    # The 'minval' from NCL. Fades to a slightly dim white.
    MIN_VALUE = 0.8 
    
    hues = np.linspace(0, 1, n_hue + 1)
    
    # --- Replicate NCL's weird piecewise saturation/value ramps ---
    # This is from the if(nSats.gt.5) block in evans_plot.ncl
    # NCL: sat(i*nSats:(i+1)*nSats-4) = fspan(0.,1.,nSats-3)
    # NCL: sat((i+1)*nSats-4:(i+1)*nSats-1) = fspan(1.,1.,4)
    sats_ramp = np.linspace(0.1, 1, n_sat - 3) # Use 0.1 so center isn't 0
    sats_full = np.ones(3)
    sats = np.concatenate([sats_ramp, sats_full])

    # NCL: val((i*nSats):(i+1)*nSats-4) = fspan(minval,1.,nSats-3)
    # NCL: val((i+1)*nSats-4:(i+1)*nSats-1) = fspan(1.,0.6,4)
    vals_ramp = np.linspace(MIN_VALUE, 1, n_sat - 3)
    vals_fade = np.linspace(1.0, 0.6, 4) # This is 4 levels, so we take 3
    vals = np.concatenate([vals_ramp, vals_fade[:3]])
    # --- End of NCL ramp logic ---
    
    for h in range(n_hue):
        for s in range(n_sat):
            # Use the correct HUE_OFFSET
            hue_val = (hues[h] + 0.5/n_hue + HUE_OFFSET) % 1.0
            sat_val = sats[s]
            val_val = vals[s] # Use the new brightness value
            
            color = mcolors.hsv_to_rgb([hue_val, sat_val, val_val])
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
    # Create 6 labels for the radius
    n_labels = 6
    cax.set_yticks(np.linspace(0, 1, n_labels + 1)[:-1] + (0.5 / n_labels))
    cax.set_yticklabels([f"{val:.1f}" for val in np.linspace(vmin, vmax, n_labels)], fontsize=8)
    cax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    cax.set_xticklabels(['0hr', '3hr', '6hr', '9hr', '12hr', '15hr', '18hr', '21hr'], fontsize=9)
    cax.set_ylim(0, 1)
    cax.set_title("Phase (hr) & \nAmplitude (mm/day)", fontsize=10)


# ---
# <<< --- PLOT IS CORRECTED (Hue, Saturation & Value) --- >>>
# ---
def plot_phase_amplitude_map(phase, amplitude, season_name, data_source, config_dict):
    """
    Creates the 2D 'Evans plot' (phase/amplitude map).
    """
    logger.info(f"  - Plotting {data_source} Phase/Amplitude map for {season_name}...")
    
    fig = plt.figure(figsize=(14, 7)) # Wide figure
    
    # 1. Normalize Phase (Hue)
    # Use the correct HUE_OFFSET
    hue = (phase / 24.0 + HUE_OFFSET) % 1.0
    
    # --- Replicate NCL's weird piecewise saturation/value ramps ---
    MIN_VALUE = 0.8
    n_sat_levels = 8 # Must match the legend
    
    # 2. Normalize Amplitude (Saturation)
    saturation = (amplitude - AMPLITUDE_VMIN) / (AMPLITUDE_VMAX - AMPLITUDE_VMIN)
    saturation = np.clip(saturation, 0.0, 1.0) # Clip 0-1
    # Apply ramp: np.linspace(0.1, 1, 5) then np.ones(3)
    sat_ramp = np.linspace(0.1, 1, n_sat_levels - 3)
    saturation = np.interp(saturation, np.linspace(0, 1, n_sat_levels-3), sat_ramp)
    saturation[saturation > 1.0] = 1.0 # Force full saturation for last 3 steps
    
    # 3. Set Value (Brightness)
    value = (amplitude - AMPLITUDE_VMIN) / (AMPLITUDE_VMAX - AMPLITUDE_VMIN)
    value = np.clip(value, 0.0, 1.0) # Clip 0-1
    # Apply ramp: np.linspace(0.8, 1, 5) then np.linspace(1, 0.6, 4)
    val_ramp1 = np.linspace(MIN_VALUE, 1.0, n_sat_levels - 3)
    val_ramp2 = np.linspace(1.0, 0.6, 4) # NCL's fade
    full_val_ramp = np.concatenate([val_ramp1, val_ramp2[1:]]) # Combine
    value = np.interp(value, np.linspace(0, 1, n_sat_levels), full_val_ramp)
    # --- End of NCL ramp logic ---

    # 4. Stack H, S, V and convert to RGB
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    
    # 5. Plot the RGB image
    if CARTOPY_AVAILABLE:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        ax.imshow(rgb, origin='lower', 
                  extent=[LON_W, LON_E, LAT_S, LAT_N], 
                  transform=ccrs.PlateCarree())
        ax.coastlines(color='0.3', linewidth=0.5)
        
        gl = ax.gridlines(draw_labels=True, linestyle=':', color='gray', alpha=0.5,
                          ylocs=np.arange(-40, 51, 20)) # Correct keyword
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12, 'rotation': 90} 
    else:
        ax = fig.add_subplot(1, 1, 1)
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
    
    plot_source_name = casename if data_source == "MODEL" else "IMERG"
    plot_year = year if data_source == "MODEL" else "" 
    
    ax.set_title(f"{plot_source_name} {plot_year} - {season_name} Phase (Hue) and Amplitude (Saturation)", fontsize=16)

    # Manually adjust subplot to make 15% room on the right for the legend
    fig.subplots_adjust(left=0.05, right=0.80, bottom=0.1, top=0.9)
    
    # Save the plot
    work_dir = config_dict.get('work_dir', '.')
    output_dir = os.path.join(work_dir, "model", "PS")
    plot_filename = f"pr_diurnal_phase_{data_source.lower()}_{season_name}.png"
    output_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(output_path, dpi=150) # Removed bbox_inches='tight'
    logger.info(f"  - Plot saved to {output_path}")
    plt.close(fig)


def plot_variance_map(variance, season_name, data_source, config_dict):
    """
    Creates the 'Variance Explained' map.
    """
    logger.info(f"  - Plotting {data_source} Variance Explained map for {season_name}...")
    
    fig = plt.figure(figsize=(14, 7)) # Wide figure

    if CARTOPY_AVAILABLE:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        plot_obj = variance.plot.contourf(
            ax=ax, transform=ccrs.PlateCarree(),
            levels=np.arange(0, 101, 10), cmap='YlOrRd',
            add_colorbar=False
        )
        ax.coastlines(color='0.3', linewidth=0.5)

        gl = ax.gridlines(draw_labels=True, linestyle=':', color='gray', alpha=0.5,
                          ylocs=np.arange(-40, 51, 20)) # Correct keyword
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12, 'rotation': 90} 
        
        cbar = fig.colorbar(plot_obj, ax=ax, orientation='vertical', 
                            shrink=0.8, aspect=30, pad=0.03)
        cbar.set_label('%', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    else:
        # Fallback for no cartopy
        ax = fig.add_subplot(1, 1, 1)
        variance.plot.contourf(
            ax=ax, levels=np.arange(0, 101, 10), cmap='YlOrRd',
            cbar_kwargs={'label': '%', 'shrink': 0.8, 'aspect': 30}
        )
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)
        
    year = config_dict.get('startdate', 'YYYY')
    casename = config_dict.get('casename', 'Model')
    plot_source_name = casename if data_source == "MODEL" else "IMERG"
    plot_year = year if data_source == "MODEL" else ""
    
    ax.set_title(f"{plot_source_name} {plot_year} - {season_name} Variance Explained by 24hr Cycle (%)", fontsize=16)

    work_dir = config_dict.get('work_dir', '.')
    output_dir = os.path.join(work_dir, "model", "PS")
    plot_filename = f"pr_diurnal_variance_{data_source.lower()}_{season_name}.png"
    output_path = os.path.join(output_dir, plot_filename)
    
    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9)
    plt.savefig(output_path, dpi=150) # Removed bbox_inches='tight'
    logger.info(f"  - Plot saved to {output_path}")
    plt.close(fig)


def plot_mean_precip_map(mean_precip, season_name, data_source, config_dict):
    """
    Creates the 'Mean Precipitation' map.
    """
    logger.info(f"  - Plotting {data_source} Mean Precipitation map for {season_name}...")
    
    fig = plt.figure(figsize=(14, 7)) # Wide figure

    if CARTOPY_AVAILABLE:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        plot_obj = mean_precip.plot.contourf(
            ax=ax, transform=ccrs.PlateCarree(),
            levels=np.arange(0, 21, 2), cmap='GnBu',
            add_colorbar=False
        )
        ax.coastlines(color='0.3', linewidth=0.5)

        gl = ax.gridlines(draw_labels=True, linestyle=':', color='gray', alpha=0.5,
                          ylocs=np.arange(-40, 51, 20))
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12, 'rotation': 90} 

        cbar = fig.colorbar(plot_obj, ax=ax, orientation='vertical', 
                            shrink=0.8, aspect=30, pad=0.03)
        cbar.set_label('mm/day', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    else:
        # Fallback for no cartopy
        ax = fig.add_subplot(1, 1, 1)
        mean_precip.plot.contourf(
            ax=ax, levels=np.arange(0, 21, 2), cmap='GnBu',
            cbar_kwargs={'label': 'mm/day', 'shrink': 0.8, 'aspect': 30}
        )
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)

    year = config_dict.get('startdate', 'YYYY')
    casename = config_dict.get('casename', 'Model')
    plot_source_name = casename if data_source == "MODEL" else "IMERG"
    plot_year = year if data_source == "MODEL" else ""
    
    ax.set_title(f"{plot_source_name} {plot_year} - {season_name} Mean Precipitation (mm/day)", fontsize=16)

    work_dir = config_dict.get('work_dir', '.')
    output_dir = os.path.join(work_dir, "model", "PS")
    plot_filename = f"pr_diurnal_mean_{data_source.lower()}_{season_name}.png"
    output_path = os.path.join(output_dir, plot_filename)
    
    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9)
    plt.savefig(output_path, dpi=150) # Removed bbox_inches='tight'
    logger.info(f"  - Plot saved to {output_path}")
    plt.close(fig)

# ---------------------------------------------------------------
# 6. CONFIG LOADING FUNCTION
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
# 7. MAIN FUNCTION
# ---------------------------------------------------------------
def main():
    """
    Main driver function to orchestrate the analysis for both model and IMERG.
    """
    parser = argparse.ArgumentParser(
        description="Run diurnal cycle phase/amplitude analysis for model and IMERG from a config file."
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
        # --- Process Model Data (once) ---
        model_da_full = load_and_process_model_data(config)
        
        # --- Loop Over Seasons for BOTH Model and IMERG ---
        for season in ['JJA', 'DJF']:
            logger.info(f"\n===== Starting {season} Diurnal Cycle Analysis =====")

            # --- Process Model for this Season ---
            if model_da_full is not None:
                logger.info(f"--- Calculating Model {season} Diurnal Cycle ---")
                model_ds_season = model_da_full.where(model_da_full['time'].dt.season == season, drop=True)
                
                if model_ds_season.time.size == 0:
                    logger.warning(f"  - No model data found for {season}. Skipping model plots.")
                else:
                    logger.info("  - Grouping model data by hour...")
                    dcycle_grouped_model = model_ds_season.groupby(model_ds_season.time.dt.hour).mean(dim='time')
                    
                    logger.info("  - Interpolating model data to 24-hourly cycle...")
                    dcycle_composite_model = dcycle_grouped_model.interp(
                        hour=HOURS_UTC_24, kwargs={"fill_value": "extrapolate"}
                    ).rename({'hour': 'time'})
                    
                    mean_precip_model = dcycle_composite_model.mean(dim='time')
                    total_variance_model = dcycle_composite_model.var(dim='time')
                    amplitude_model, phase_deg_model = get_fourier_components(dcycle_composite_model)
                    phase_local_model = convert_phase_to_local_time(phase_deg_model, dcycle_composite_model.lon, MODEL_TIME_SHIFT)
                    harmonic_variance_model = (amplitude_model**2) / 2.0
                    variance_explained_model = (harmonic_variance_model / total_variance_model) * 100.0
                    variance_explained_model = variance_explained_model.clip(0, 100)
                    
                    # Plot model results
                    plot_phase_amplitude_map(phase_local_model, amplitude_model, season, "MODEL", config)
                    plot_variance_map(variance_explained_model, season, "MODEL", config)
                    plot_mean_precip_map(mean_precip_model, season, "MODEL", config)
                    logger.info(f"--- Model {season} plots complete ---")
            else:
                logger.warning(f"  - Model data is None. Skipping model plots.")


            # --- Process IMERG for this Season ---
            imerg_da_season = load_and_process_imerg_data(config, season)

            if imerg_da_season is not None:
                logger.info(f"--- Calculating IMERG {season} Diurnal Cycle ---")
                
                # IMERG is already 3-hourly, just interpolate to 24-hourly
                logger.info("  - Interpolating IMERG data to 24-hourly cycle...")
                dcycle_composite_imerg = imerg_da_season.interp(
                    time=HOURS_UTC_24, kwargs={"fill_value": "extrapolate"}
                )
                
                mean_precip_imerg = dcycle_composite_imerg.mean(dim='time')
                total_variance_imerg = dcycle_composite_imerg.var(dim='time')
                amplitude_imerg, phase_deg_imerg = get_fourier_components(dcycle_composite_imerg)
                phase_local_imerg = convert_phase_to_local_time(phase_deg_imerg, dcycle_composite_imerg.lon, IMERG_TIME_SHIFT)
                harmonic_variance_imerg = (amplitude_imerg**2) / 2.0
                variance_explained_imerg = (harmonic_variance_imerg / total_variance_imerg) * 100.0
                variance_explained_imerg = variance_explained_imerg.clip(0, 100)
                
                # Plot IMERG results
                plot_phase_amplitude_map(phase_local_imerg, amplitude_imerg, season, "IMERG", config)
                plot_variance_map(variance_explained_imerg, season, "IMERG", config)
                plot_mean_precip_map(mean_precip_imerg, season, "IMERG", config)
                logger.info(f"--- IMERG {season} plots complete ---")
            else:
                logger.warning(f"  - No IMERG data found for {season}. Skipping IMERG plots.")

            logger.info(f"===== {season} Analysis Complete =====")

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