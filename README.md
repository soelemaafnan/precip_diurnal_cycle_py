# Diurnal Cycle of Precipitation Analysis

This repository contains a robust Python-based analytical framework to evaluate the diurnal cycle of precipitation, translating and modernizing legacy NCL climate diagnostics. 

The workflow compares simulated precipitation from climate models (e.g., GFDL SPEAR-MED) against high-resolution satellite observations (GPM IMERG). By applying a Fast Fourier Transform (FFT) along the temporal axis, the algorithm extracts the first harmonic to isolate the 24-hour daily cycle. From this, it calculates two primary metrics: the **amplitude** (the intensity of the daily cycle) and the **phase** (the local solar time of peak rainfall). The toolset generates spatial Evans plots—mapping phase to color hue and amplitude to color saturation—alongside variance and mean precipitation maps, enabling rigorous benchmarking of convective patterns.

> **Disclaimer:** This repository and its default configurations are currently set up as a demonstration for the year **2015 only**. To expand this analysis to include multiple years, please see the "Expanding to Multiple Years" instructions at the bottom of this document.

---

## Directory Structure

```text
diurnal-cycle-analysis/
├── .gitignore                # Tells git to ignore large data files
├── README.md                 # This file
├── environment.yml           # Conda environment definition
├── config.yaml               # User settings and file paths
└── src/                      # Source code
    ├── preprocess.py         # Dask-based IMERG raw data processing
    ├── analyze_cycle.py      # Generates regional diurnal cycle line plots
    └── analyze_phase.py      # Generates phase, amplitude, and variance maps
```

*(Note: Data and output directories should be kept locally and ignored by Git).*

---

## Data Acquisition

### 1. Satellite Observations (GPM IMERG)
The observational data used in this analysis is the GPM IMERG Final Run (Half-Hourly, 0.1° resolution, `3B-HHR`). The raw `.HDF5` files must be acquired from NASA's Goddard Earth Sciences Data and Information Services Center (GES DISC) using `wget`.

To download the satellite dataset:
1. **Create an Earthdata Account:** Register for a free [NASA Earthdata login](https://urs.earthdata.nasa.gov/).
2. **Configure Authentication:** Create a `.netrc` file in your home directory:
   ```bash
   echo "machine urs.earthdata.nasa.gov login <YOUR_USERNAME> password <YOUR_PASSWORD>" >> ~/.netrc
   chmod 0600 ~/.netrc
   ```
3. **Get the File List:** Use the [GES DISC search portal](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_06/summary) to filter for your desired dates (e.g., JJA and DJF of 2015) and download the generated URL list (e.g., `subset_GPM_3IMERGHH_06.txt`).
4. **Download via `wget`:** Run the following command in your raw data directory:
   ```bash
   wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -i subset_GPM_3IMERGHH_06.txt
   ```

### 2. Model Output (SPEAR)
The model data utilized is the 6-hourly time-mean precipitation from the GFDL SPEAR climate model. Ensure your `config.yaml` points to the correct NetCDF output file for your specific model run.

---

## Installation

This project relies on `xarray`, `dask`, and `cartopy`. To ensure reproducibility, create the Conda environment using the provided `environment.yml` file:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate diurnal_cycle
```

---

## Usage

### 1. Configuration
Edit the `config.yaml` file in the root directory to define your analysis period and set the paths to your raw data, processed data, and output directories.

### 2. Preprocessing (Run Once)
The satellite data is natively 30-minute resolution. To match the legacy TRMM 3-hourly standards and make it mathematically comparable to the 6-hourly model data, run the preprocessing script. This uses a Dask `LocalCluster` to memory-efficiently average the raw HDF5 files into seasonal 3-hourly composite days.

```bash
python src/preprocess.py -c config.yaml
```

### 3. Generate Line Plots (Diurnal Cycle)
This script interpolates the data to a 1-hourly grid, corrects for local solar time, and plots the regional diurnal cycle comparisons (SPEAR vs. IMERG) across 6 predefined global regions.

```bash
python src/analyze_cycle.py -c config.yaml
```

### 4. Generate Spatial Maps (Phase & Amplitude)
This script performs the Fourier analysis and generates three separate map files for each dataset and season: Phase/Amplitude (Evans Plot), Variance Explained (%), and Mean Precipitation.

```bash
python src/analyze_phase.py -c config.yaml
```

---

## Expanding to Multiple Years

The preprocessing and analysis scripts are fully capable of handling multi-year climatologies. To run this analysis over a longer time period (e.g., 2015–2020):

1. **Download Additional Satellite Data:** Go back to the GES DISC portal, generate a new URL list spanning your full multi-year date range, and download the new `.HDF5` files into your `raw_satellite_dir`.
2. **Provide Multi-Year Model Data:** Ensure the model NetCDF file referenced in `pr_file` contains data covering the entire desired time span.
3. **Update `config.yaml`:** Modify the `analysis_period` -> `start` and `end` dates to reflect the new, expanded timeframe.
4. **Re-run the Preprocessor:** Run `python src/preprocess.py -c config.yaml`. The Dask-powered script will automatically ingest all the new `.HDF5` files and crunch them into a single, multi-year seasonal composite, which the analysis scripts will then use automatically.
