# AGENTS.md

## Project

**caneblu** — Python package for satellite/geospatial data processing (vegetation indices, geocoding, Sentinel Hub harvesting, weather APIs). Focused on fire risk assessment.

## Package structure

Single package at repo root. `setup.py` maps `package_dir={"caneblu": "."}` — the `caneblu/` directory *is* the package root.

7 submodules, all lazy-loaded via `__getattr__` in `__init__.py`:

| File | Purpose |
|---|---|
| `composite_indices.py` | NDVI, NDWI, NBR, NDMI, NDCI, NDSI, MSI |
| `geocoding.py` | Geocoding, polygon splitting (SentinelHub `UtmZoneSplitter`), coordinate utils |
| `harvester.py` | `SentinelHub` class — data download orchestration |
| `satellite_image_utils.py` | GeoTIFF I/O, RGB/grayscale conversions, feature creation |
| `evalscript_generator_utils.py` | Sentinel Hub evalscript strings |
| `open_meteo_api.py` | Open-Meteo archive/forecast API wrapper |
| `utils.py` | Shared helpers (file I/O, time, dedup) |

## Important quirks
- **Hardcoded paths** in `geocoding.py:349` (`/home/edoardo/DATA/confini_amministrativi/...`) and `open_meteo_api.py:260` (`/home/edoardo/DATA/SEFORE/...`).
- **Sentinel Hub credentials** expected at `json/.sentinelhub_credentials.json` (relative to CWD at runtime).
- **No tests, no CI, no lint/format/typecheck config.** Repo has zero verification pipeline.

## Setup

```sh
python -m venv venv && source venv/bin/activate
pip install -e .
```

## Key dependencies

`rasterio`, `geopandas`, `sentinelhub`, `earthengine-api`, `shapely`, `numpy`, `streamlit`, `open-meteo` (via HTTP).

## Two undocumented submodules

`evalscript_generator_utils` and `open_meteo_api` are in `_SUBMODULES` but **not listed in README**. They are accessible via lazy import.
