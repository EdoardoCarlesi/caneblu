# AGENTS.md

## Project

**caneblu** — Python package for satellite/geospatial data processing (vegetation indices, geocoding, Sentinel Hub harvesting, weather APIs). Focused on fire risk assessment.

## Package structure

Single package at repo root. `setup.py` maps `package_dir={"caneblu": "."}` — the `caneblu/` directory *is* the package root.

7 submodules, all lazy-loaded via `__getattr__` in `__init__.py`:

| File | Purpose |
|---|---|
| `composite_indices.py` | NDVI, NDWI, NBR, NDMI, NDCI, NDSI, MSI |
| `geocoding.py` | Geocoding, coordinate utils, Italian admin boundaries (`get_admin_boundaries`) and square grids (`generate_squares_for_admin_area`, via SentinelHub `UtmZoneSplitter`) |
| `harvester.py` | `SentinelHub` class — data download orchestration. `provider="sentinelhub"` (default) or `"earth_search"` (key-less Sentinel-2). `download_area(name, level, side, time_interval, ..., sar=False)` downloads a square range over a regione/provincia/comune; also the CLI (`python -m caneblu.harvester -a CT -l provincia ...`) |
| `earth_search.py` | Earth Search STAC client: key-less Sentinel-2 search and COG reads (used when `provider="earth_search"`) |
| `satellite_image_utils.py` | GeoTIFF I/O, RGB/grayscale conversions, feature creation |
| `evalscript_generator_utils.py` | Sentinel Hub evalscript strings |
| `open_meteo_api.py` | Open-Meteo archive/forecast API wrapper |
| `utils.py` | Shared helpers (file I/O, time, dedup) |

## Important quirks
- **Bundled data**: ISTAT 2021 boundaries (regioni, province, comuni) live in `caneblu/data/` and ship via `package_data`.
- **Runtime caches are relative to CWD**: `generate_squares_for_admin_area` writes `data/{level}_{name}_squares_{side}.shp` and `download_area` writes `tmp/[sar_]{level}_{name}_{side}_*.npy`; existing files are reused, not recomputed.
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
