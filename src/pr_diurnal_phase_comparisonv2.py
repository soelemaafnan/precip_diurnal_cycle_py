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
- Uses a polar pcolormesh color wheel (no partial fill).
- Only processes MODEL data (IMERG removed from main()).
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
TSPD = 24  # Time steps per diurnal cycle (24 hours)
HOURS_UTC_24 = np.arange(0, TSPD)  # For 1-hourly interpolation
MODEL_TIME_SHIFT = -3.0  # For 6-hourly time-mean model data

# Title experiment label and fixed year for titles
TITLE_EXPERIMENT = "SPEAR-MED"
TITLE_YEAR       = "2015"

# NCL plot shows 0hr=Blue(0.66), 12hr=Yellow(0.16).
# This implies a (0.66) or 240-degree offset, not 180.
HUE_OFFSET = 0.66  # (240 / 360)

# ---------------------------------------------------------------
# 2. LOGGER SETUP
# ---------------------------------------------------------------
logger = logging.getLogger()  # Get the root logger


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

def get_fourier_components(dcycle_composite):
    """
    Performs a Fourier analysis to get phase and amplitude.
    Returns amplitude and PHASE IN DEGREES (like NCL).
    """
    logger.info("    - Calculating Fourier transform...")
    N = len(dcycle_composite.time)  # Number of time steps (24)

    # Compute the Fast Fourier Transform along the time axis
    fft_result = np.fft.fft(dcycle_composite.values, axis=0)

    # Amplitude is (2 * abs(fft[1])) / N
    amplitude_data = (np.abs(fft_result[1, :, :]) * 2) / N

    # Phase is the angle of the complex number (radians → degrees)
    phase_rad_data = np.angle(fft_result[1, :, :])
    phase_deg_data = np.degrees(phase_rad_data)

    amplitude = xr.DataArray(
        amplitude_data,
        coords={'lat': dcycle_composite.lat, 'lon': dcycle_composite.lon},
        dims=['lat', 'lon']
    )

    phase_deg = xr.DataArray(
        phase_deg_data,
        coords={'lat': dcycle_composite.lat, 'lon': dcycle_composite.lon},
        dims=['lat', 'lon']
    )

    return amplitude, phase_deg


def convert_phase_to_local_time(phase_deg, lon_coords, time_shift):
    """
    Converts GMT phase (in DEGREES) to local solar time (in hours).
    """
    logger.info("    - Converting phase to local solar time...")

    # 1. phase (deg) → GMT hours (0–24)
    phase_gmt_hours = phase_deg / (360.0 / TSPD)  # phase_deg / 15.0

    # 2. longitude → time offset (hours)
    lon_offset_hours = (lon_coords / 360.0) * TSPD

    # 3. subtract longitude offset + model-specific time shift
    phase_local = phase_gmt_hours - lon_offset_hours - time_shift

    # 4. wrap to [0,24)
    phase_local_hours = phase_local % TSPD

    return phase_local_hours


# ---------------------------------------------------------------
# 4. DATA LOADING
# ---------------------------------------------------------------

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

    da = ds_filtered[pr_var] * 86400.0  # Convert to mm/day
    da = da.sel(lat=slice(LAT_S, LAT_N), lon=slice(LON_W, LON_E))  # Crop

    # Ensure standard dimension order (time, lat, lon)
    da = da.transpose('time', 'lat', 'lon')

    return da


# ---------------------------------------------------------------
# 5. PLOTTING HELPERS
# ---------------------------------------------------------------

def add_color_wheel(fig, vmin, vmax, subplot_position=[0.81, 0.18, 0.16, 0.55]):
    """
    Evans-style color wheel using the same phase→hue and
    amplitude→(saturation, value) mapping as the main map.

    vmin, vmax: amplitude range (e.g., AMPLITUDE_VMIN, AMPLITUDE_VMAX)
    subplot_position: [left, bottom, width, height] in figure coords
    """
    ax = fig.add_axes(subplot_position, projection='polar')

    # Resolution of the wheel
    n_theta = 72   # angular sectors (every 5°)
    n_r = 40       # radial bands

    # Cell edges in polar coordinates
    theta_edges = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    r_edges = np.linspace(0.0, 1.0, n_r + 1)

    TH, RR = np.meshgrid(theta_edges, r_edges)

    # Cell centers
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    THc, Rc = np.meshgrid(theta_centers, r_centers)

    # amplitude → S, V mapping (same logic as plot_phase_amplitude_map)
    n_sat_levels = 8
    MIN_VALUE = 0.8

    sat_ramp = np.linspace(0.1, 1.0, n_sat_levels - 3)  # 5 values
    val_ramp1 = np.linspace(MIN_VALUE, 1.0, n_sat_levels - 3)
    val_ramp2 = np.linspace(1.0, 0.6, 4)                # 4 values
    full_val_ramp = np.concatenate([val_ramp1, val_ramp2[1:]])  # 5 + 3 = 8

    def amp_to_sv(amp):
        # Normalize amplitude to [0,1]
        amp_norm = (amp - vmin) / (vmax - vmin)
        amp_norm = np.clip(amp_norm, 0.0, 1.0)

        # Saturation
        s = np.interp(
            amp_norm,
            np.linspace(0.0, 1.0, n_sat_levels - 3),
            sat_ramp
        )

        # Value
        v = np.interp(
            amp_norm,
            np.linspace(0.0, 1.0, n_sat_levels),
            full_val_ramp
        )

        s = np.clip(s, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        return s, v

    # Amplitude as a function of radius (0–1 → vmin–vmax)
    amp = vmin + Rc * (vmax - vmin)
    s, v = amp_to_sv(amp)

    # Phase/hour → hue
    phase_hr = (THc / (2.0 * np.pi)) * 24.0
    hue = (phase_hr / 24.0 + HUE_OFFSET) % 1.0

    # Convert HSV → RGB
    hsv = np.stack([hue, s, v], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)

    # Build a colormap with one color per cell
    rgb_flat = rgb.reshape(-1, 3)
    cmap = mcolors.ListedColormap(rgb_flat)

    # Dummy index field just to drive pcolormesh
    Z = np.arange(rgb_flat.shape[0]).reshape(n_r, n_theta)

    # Draw the colored wheel
    ax.pcolormesh(TH, RR, Z, cmap=cmap, shading='auto')
    ax.set_ylim(0.0, 1.0)

    # Radial ticks in amplitude units
    n_r_labels = 5
    r_tick_vals = np.linspace(0.0, 1.0, n_r_labels + 1)[1:]  # skip center
    amp_ticks = vmin + r_tick_vals * (vmax - vmin)

    ax.set_yticks(r_tick_vals)
    ax.set_yticklabels([f"{a:.1f}" for a in amp_ticks], fontsize=8)

    # Angular ticks in hours (0,3,6,...,21)
    hour_labels = np.arange(0, 24, 3)
    theta_ticks = (hour_labels / 24.0) * 2.0 * np.pi

    ax.set_xticks(theta_ticks)
    ax.set_xticklabels([f"{h}hr" for h in hour_labels], fontsize=9)

    ax.set_title("Phase (hr) &\nAmplitude (mm/day)", fontsize=10)


def plot_phase_amplitude_map(phase, amplitude, season_name, data_source, config_dict):
    """
    Creates the 2D 'Evans plot' (phase/amplitude map).
    """
    logger.info(f"  - Plotting {data_source} Phase/Amplitude map for {season_name}...")

    fig = plt.figure(figsize=(14, 7))  # Wide figure

    # 1. Phase → Hue
    hue = (phase / 24.0 + HUE_OFFSET) % 1.0

    # 2. Amplitude → Saturation & Value (NCL-like ramps)
    MIN_VALUE = 0.8
    n_sat_levels = 8  # must match wheel

    saturation = (amplitude - AMPLITUDE_VMIN) / (AMPLITUDE_VMAX - AMPLITUDE_VMIN)
    saturation = np.clip(saturation, 0.0, 1.0)
    sat_ramp = np.linspace(0.1, 1, n_sat_levels - 3)
    saturation = np.interp(saturation, np.linspace(0, 1, n_sat_levels - 3), sat_ramp)
    saturation[saturation > 1.0] = 1.0

    value = (amplitude - AMPLITUDE_VMIN) / (AMPLITUDE_VMAX - AMPLITUDE_VMIN)
    value = np.clip(value, 0.0, 1.0)
    val_ramp1 = np.linspace(MIN_VALUE, 1.0, n_sat_levels - 3)
    val_ramp2 = np.linspace(1.0, 0.6, 4)
    full_val_ramp = np.concatenate([val_ramp1, val_ramp2[1:]])
    value = np.interp(value, np.linspace(0, 1, n_sat_levels), full_val_ramp)

    # 3. HSV → RGB
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)

    # 4. Plot
    if CARTOPY_AVAILABLE:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        ax.imshow(
            rgb,
            origin='lower',
            extent=[LON_W, LON_E, LAT_S, LAT_N],
            transform=ccrs.PlateCarree()
        )
        ax.coastlines(color='0.3', linewidth=0.5)

        gl = ax.gridlines(
            draw_labels=True,
            linestyle=':',
            color='gray',
            alpha=0.5,
            ylocs=np.arange(-40, 51, 20)
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12, 'rotation': 90}
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(
            rgb,
            origin='lower',
            extent=[LON_W, LON_E, LAT_S, LAT_N],
            aspect='auto'
        )
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)
        logger.warning("  - 'cartopy' not found. Plotting a simple image.")

    # Add the color wheel
    add_color_wheel(fig, AMPLITUDE_VMIN, AMPLITUDE_VMAX)

    # Title: SPEAR-MED JJA/DJF 2015 - Phase and Amplitude
    title_prefix = f"{TITLE_EXPERIMENT} {season_name} {TITLE_YEAR}"
    ax.set_title(f"{title_prefix} - Phase and Amplitude", fontsize=16)

    # Layout: leave room for wheel
    fig.subplots_adjust(left=0.05, right=0.78, bottom=0.1, top=0.9)

    # Save
    work_dir = config_dict.get('work_dir', '.')
    output_dir = os.path.join(work_dir, "model", "PS")
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"pr_diurnal_phase_{data_source.lower()}_{season_name}.png"
    output_path = os.path.join(output_dir, plot_filename)

    plt.savefig(output_path, dpi=150)
    logger.info(f"  - Plot saved to {output_path}")
    plt.close(fig)


def plot_variance_map(variance, season_name, data_source, config_dict):
    """
    Creates the 'Variance Explained' map.
    """
    logger.info(f"  - Plotting {data_source} Variance Explained map for {season_name}...")

    fig = plt.figure(figsize=(14, 7))

    if CARTOPY_AVAILABLE:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        plot_obj = variance.plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            levels=np.arange(0, 101, 10),
            cmap='YlOrRd',
            add_colorbar=False
        )
        ax.coastlines(color='0.3', linewidth=0.5)

        gl = ax.gridlines(
            draw_labels=True,
            linestyle=':',
            color='gray',
            alpha=0.5,
            ylocs=np.arange(-40, 51, 20)
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12, 'rotation': 90}

        cbar = fig.colorbar(
            plot_obj,
            ax=ax,
            orientation='vertical',
            shrink=0.8,
            aspect=30,
            pad=0.03
        )
        cbar.set_label('%', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    else:
        ax = fig.add_subplot(1, 1, 1)
        variance.plot.contourf(
            ax=ax,
            levels=np.arange(0, 101, 10),
            cmap='YlOrRd',
            cbar_kwargs={'label': '%', 'shrink': 0.8, 'aspect': 30}
        )
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)

    # Title: SPEAR-MED JJA/DJF 2015 - Variance
    title_prefix = f"{TITLE_EXPERIMENT} {season_name} {TITLE_YEAR}"
    ax.set_title(f"{title_prefix} - Variance", fontsize=16)

    work_dir = config_dict.get('work_dir', '.')
    output_dir = os.path.join(work_dir, "model", "PS")
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"pr_diurnal_variance_{data_source.lower()}_{season_name}.png"
    output_path = os.path.join(output_dir, plot_filename)

    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9)
    plt.savefig(output_path, dpi=150)
    logger.info(f"  - Plot saved to {output_path}")
    plt.close(fig)


def plot_mean_precip_map(mean_precip, season_name, data_source, config_dict):
    """
    Creates the 'Mean Precipitation' map.
    """
    logger.info(f"  - Plotting {data_source} Mean Precipitation map for {season_name}...")

    fig = plt.figure(figsize=(14, 7))

    if CARTOPY_AVAILABLE:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        plot_obj = mean_precip.plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            levels=np.arange(0, 21, 2),
            cmap='GnBu',
            add_colorbar=False
        )
        ax.coastlines(color='0.3', linewidth=0.5)

        gl = ax.gridlines(
            draw_labels=True,
            linestyle=':',
            color='gray',
            alpha=0.5,
            ylocs=np.arange(-40, 51, 20)
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12, 'rotation': 90}

        cbar = fig.colorbar(
            plot_obj,
            ax=ax,
            orientation='vertical',
            shrink=0.8,
            aspect=30,
            pad=0.03
        )
        cbar.set_label('mm/day', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    else:
        ax = fig.add_subplot(1, 1, 1)
        mean_precip.plot.contourf(
            ax=ax,
            levels=np.arange(0, 21, 2),
            cmap='GnBu',
            cbar_kwargs={'label': 'mm/day', 'shrink': 0.8, 'aspect': 30}
        )
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)

    # Title: SPEAR-MED JJA/DJF 2015 - Mean Precipitation (mm/day)
    title_prefix = f"{TITLE_EXPERIMENT} {season_name} {TITLE_YEAR}"
    ax.set_title(f"{title_prefix} - Mean Precipitation (mm/day)", fontsize=16)

    work_dir = config_dict.get('work_dir', '.')
    output_dir = os.path.join(work_dir, "model", "PS")
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"pr_diurnal_mean_{data_source.lower()}_{season_name}.png"
    output_path = os.path.join(output_dir, plot_filename)

    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9)
    plt.savefig(output_path, dpi=150)
    logger.info(f"  - Plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------
# 6. CONFIG LOADING
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
# 7. MAIN
# ---------------------------------------------------------------
def main():
    """
    Main driver function to orchestrate the analysis for the model data.
    """
    parser = argparse.ArgumentParser(
        description="Run diurnal cycle phase/amplitude analysis for model data from a config file."
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
        # Process model data once
        model_da_full = load_and_process_model_data(config)

        # Loop over seasons for MODEL ONLY
        for season in ['JJA', 'DJF']:
            logger.info(f"\n===== Starting {season} Diurnal Cycle Analysis (MODEL) =====")

            if model_da_full is not None:
                logger.info(f"--- Calculating Model {season} Diurnal Cycle ---")
                model_ds_season = model_da_full.where(
                    model_da_full['time'].dt.season == season,
                    drop=True
                )

                if model_ds_season.time.size == 0:
                    logger.warning(f"  - No model data found for {season}. Skipping model plots.")
                else:
                    logger.info("  - Grouping model data by hour...")
                    dcycle_grouped_model = model_ds_season.groupby(
                        model_ds_season.time.dt.hour
                    ).mean(dim='time')

                    logger.info("  - Interpolating model data to 24-hourly cycle...")
                    dcycle_composite_model = dcycle_grouped_model.interp(
                        hour=HOURS_UTC_24,
                        kwargs={"fill_value": "extrapolate"}
                    ).rename({'hour': 'time'})

                    mean_precip_model = dcycle_composite_model.mean(dim='time')
                    total_variance_model = dcycle_composite_model.var(dim='time')
                    amplitude_model, phase_deg_model = get_fourier_components(dcycle_composite_model)
                    phase_local_model = convert_phase_to_local_time(
                        phase_deg_model,
                        dcycle_composite_model.lon,
                        MODEL_TIME_SHIFT
                    )
                    harmonic_variance_model = (amplitude_model ** 2) / 2.0
                    variance_explained_model = (harmonic_variance_model / total_variance_model) * 100.0
                    variance_explained_model = variance_explained_model.clip(0, 100)

                    # Plots
                    plot_phase_amplitude_map(
                        phase_local_model,
                        amplitude_model,
                        season,
                        "MODEL",
                        config
                    )
                    plot_variance_map(
                        variance_explained_model,
                        season,
                        "MODEL",
                        config
                    )
                    plot_mean_precip_map(
                        mean_precip_model,
                        season,
                        "MODEL",
                        config
                    )
                    logger.info(f"--- Model {season} plots complete ---")
            else:
                logger.warning("  - Model data is None. Skipping model plots.")

            logger.info(f"===== {season} Analysis Complete (MODEL) =====")

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