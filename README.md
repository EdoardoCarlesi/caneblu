# caneblu

Utilities for satellite and geospatial data: vegetation and burn indices, Sentinel Hub
downloads, geocoding and polygon tools, GeoTIFF helpers, and Open-Meteo weather data.
The package was built for fire-risk assessment but every module is usable on its own.

- **Plain functions** – most helpers are stateless and take NumPy arrays, shapely shapes or file paths.
- **Lazy imports** – `import caneblu` loads nothing heavy; each submodule loads on first use, and optional dependencies (Earth Engine, scikit-learn) load only inside the functions that need them.
- **One class for remote data** – `caneblu.harvester.SentinelHub` handles catalog search and downloads.

## Contents

- [Installation](#installation)
- [Credentials](#credentials)
- [Quick start](#quick-start)
- [Modules at a glance](#modules-at-a-glance)
- [Band order (important)](#band-order-important)
- [Module guide](#module-guide)
  - [composite_indices](#composite_indices--vegetation-and-burn-indices)
  - [evalscript_generator_utils](#evalscript_generator_utils--sentinel-hub-evalscripts)
  - [harvester](#harvester--sentinel-hub-downloads)
  - [geocoding](#geocoding--polygons-coordinates-and-squares)
  - [satellite_image_utils](#satellite_image_utils--rasters-and-features)
  - [open_meteo_api](#open_meteo_api--weather-data)
  - [utils](#utils--general-helpers)
- [End-to-end workflow](#end-to-end-workflow)
- [Command-line usage](#command-line-usage)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [License](#license)

---

## Installation

Requires Python 3.10 or newer.

```bash
# from a checkout of this repository
pip install -e .

# or straight from GitHub
pip install git+https://github.com/EdoardoCarlesi/caneblu.git
```

The pinned versions used during development are in `requirements.txt`. The main dependencies are
`numpy`, `pandas`, `geopandas`, `shapely`, `rasterio`, `tifffile`, `pyproj`, `sentinelhub`,
`requests`, `arrow`, `matplotlib`, `pillow`, and `tqdm`. Optional: `earthengine-api` (only for
`geocoding.bbox_from_address`) and `scikit-learn` (only for `OpenMeteoAPI.recover_humidity`).

Check the installation:

```python
import caneblu
from caneblu import composite_indices, geocoding, harvester, satellite_image_utils
from caneblu import open_meteo_api, utils, evalscript_generator_utils

print(caneblu.__version__)
```

---

## Credentials

Only the Sentinel Hub downloads (`harvester`) need credentials. Get a client ID and secret from the
Sentinel Hub dashboard, then provide them in one of three ways, in this order of precedence:

1. Arguments: `SentinelHub(client_id="...", client_secret="...")`
2. Environment variables (recommended): `SH_CLIENT_ID` and `SH_CLIENT_SECRET`
3. A JSON file (default path `json/.sentinelhub_credentials.json`, relative to the working directory):

```json
{
    "CLIENT_ID": "your-client-id",
    "CLIENT_SECRET": "your-client-secret"
}
```

Open-Meteo needs no credentials for its free API. A commercial API key is optional and is read from
`OPEN_METEO_API_KEY` (or passed as `OpenMeteoAPI(..., api_key=...)`). See the `open_meteo_api` section.

Never commit credentials to version control. Use environment variables or a file that is git-ignored.

```bash
export SH_CLIENT_ID="your-client-id"
export SH_CLIENT_SECRET="your-client-secret"
```

---

## Quick start

Compute a vegetation index from an array of reflectance values:

```python
import numpy as np
from caneblu.composite_indices import NDVI

# A 2x2 image with 4 bands in the order [red, green, blue, nir]
img = np.array([
    [[0.10, 0.12, 0.08, 0.50], [0.20, 0.18, 0.15, 0.40]],
    [[0.05, 0.07, 0.04, 0.60], [0.30, 0.28, 0.25, 0.30]],
])

ndvi = NDVI(img, i_red=0, i_nir=3)
print(ndvi.shape)       # (2, 2)
print(ndvi.round(3))
```

Convert a bounding box to a polygon and split a large area into squares:

```python
from caneblu.geocoding import bbox_to_polygon

poly = bbox_to_polygon((15.0, 37.0, 15.5, 37.5))   # (min_lon, min_lat, max_lon, max_lat)
print(poly.bounds)      # (15.0, 37.0, 15.5, 37.5)
```

Download Sentinel-2 data (requires credentials and network):

```python
# requires network and credentials
from caneblu.harvester import SentinelHub

sh = SentinelHub()      # reads SH_CLIENT_ID / SH_CLIENT_SECRET
data, files = sh.download_roi(
    bbox=(15.0, 37.0, 15.5, 37.5),
    resolution=10,
    time_interval=("2024-06-01", "2024-07-30"),
    cloud_cover=20,
    data_type="all_bands_10m",
    base_path="tmp/area_",
)
print(len(data), "acquisitions;", files[0])
```

---

## Modules at a glance

| Module | What it does | Main entry points |
|---|---|---|
| `composite_indices` | Spectral indices on NumPy arrays | `NDVI`, `NDWI`, `MNDWI`, `NBR`, `NDMI`, `NDCI`, `NDSI`, `MSI` |
| `evalscript_generator_utils` | Sentinel Hub evalscript strings | `get_evalscript(name, bands=None)` |
| `harvester` | Sentinel Hub catalog search and downloads | `SentinelHub`, `download_province`, `format_for_fire_risk` |
| `geocoding` | Geometry, coordinates, squares, overlap checks | `split_into_squares`, `bbox_to_polygon`, `check_polygons_overlap`, `generate_squares_for_province` |
| `satellite_image_utils` | GeoTIFF I/O, grayscale, feature tables, cropping | `tiff_to_png`, `create_features`, `save_numpy_to_geotiff`, `crop_tiff_with_polygon` |
| `open_meteo_api` | Open-Meteo archive and forecast wrapper | `OpenMeteoAPI` |
| `utils` | Filenames, JSON/GeoJSON I/O, time conversions | `npy_filename`, `read_json_file`, `time_to_utc` |

---

## Band order (important)

Index functions take **band positions**, not band names. The position depends on the evalscript
(the `data_type` argument) that produced the array. Use this table to pick the right indices:

| `data_type` | Band order (index 0, 1, 2, …) | Typical call |
|---|---|---|
| `all_bands_10m` (default for `download_roi`) | 0 red (B04), 1 green (B03), 2 blue (B02), 3 NIR (B08) | `NDVI(img, i_red=0, i_nir=3)` |
| `all_bands` | same as `all_bands_10m` | same as above |
| `fire_risk_bands_60m` | 0 B02, 1 B03, 2 B04 (red), 3 B08 (NIR), 4 B11 (SWIR1), 5 B12 (SWIR2) | `NDMI(img, i_swir=4, i_nir=3)` |
| `bands_60m_rgb` | 0 B02, 1 B03, 2 B04 | — |
| `bands_60m_ir` | 0 B08, 1 B11, 2 B12 | — |
| `false_color` | 0 B08 (NIR), 1 B04 (red), 2 B03 (green) | `NDVI(img, i_red=1, i_nir=0)` |
| `ndvi` | 0 NDVI (already computed on the server) | — |
| `cloud_mask` | 0 CLM (cloud mask) | — |

Sentinel-2 band reference: B02 blue, B03 green, B04 red, B08 NIR, B11 SWIR1, B12 SWIR2.
When in doubt, check `get_evalscript(name)` for the exact order.

---

## Module guide

### `composite_indices` – vegetation and burn indices

Every function takes an image shaped `(height, width, bands)` and returns a 2-D float array
`(height, width)`. Band indices are passed explicitly, so one function works with any band layout.
A small constant (`eps = 0.0001`) prevents division by zero.

| Function | Formula | Arguments |
|---|---|---|
| `NDVI` | (NIR − Red) / (NIR + Red) | `i_red`, `i_nir` |
| `NDWI` | (Green − NIR) / (Green + NIR) | `i_green`, `i_nir` |
| `MNDWI` | (Green − SWIR) / (Green + SWIR) | `i_green`, `i_swir` |
| `NBR` | (NIR − SWIR) / (NIR + SWIR) | `i_swir`, `i_nir` |
| `NDMI` | (NIR − SWIR) / (NIR + SWIR) (moisture) | `i_swir`, `i_nir` |
| `NDCI` | (VNIR − NIR) / (VNIR + NIR) (chlorophyll) | `i_vnir`, `i_nir` |
| `NDSI` | (SWIR − VNIR) / (SWIR + VNIR) (snow) | `i_vnir`, `i_swir` |
| `MSI` | SWIR / NIR (moisture stress) | `i_swir`, `i_nir` |

**Example 1 – NDVI on a real-looking raster with a 4-band layout**

```python
import numpy as np
from caneblu import composite_indices as ci

rng = np.random.default_rng(0)
img = rng.uniform(0.02, 0.6, size=(100, 120, 4))   # random reflectances, [red, green, blue, nir]

ndvi = ci.NDVI(img, i_red=0, i_nir=3)
print("shape:", ndvi.shape)                          # (100, 120)
print("range:", round(ndvi.min(), 3), round(ndvi.max(), 3))  # between -1 and 1
print("share above 0.3:", round((ndvi > 0.3).mean(), 3))
```

**Example 2 – burn severity: difference of NBR before and after a fire**

Burn severity is usually measured as dNBR = NBR_pre − NBR_post. Values above roughly 0.44
indicate high severity (common thresholds vary by study).

```python
import numpy as np
from caneblu import composite_indices as ci

# fire_risk_bands_60m layout: 0 B02, 1 B03, 2 B04, 3 B08 (NIR), 4 B11 (SWIR1), 5 B12 (SWIR2)
pre  = np.zeros((50, 50, 6));  pre[..., 3] = 0.45;  pre[..., 5] = 0.15   # healthy vegetation
post = np.zeros((50, 50, 6));  post[..., 3] = 0.12; post[..., 5] = 0.30  # burned

nbr_pre  = ci.NBR(pre,  i_swir=5, i_nir=3)   # NBR uses SWIR2 (B12) here
nbr_post = ci.NBR(post, i_swir=5, i_nir=3)

dnbr = nbr_pre - nbr_post
print("mean dNBR:", round(dnbr.mean(), 3))
print("high severity share:", round((dnbr > 0.44).mean(), 3))
```

**Example 3 – water and moisture indices on the same stack**

```python
import numpy as np
from caneblu import composite_indices as ci

img = np.random.default_rng(1).uniform(0.05, 0.5, size=(10, 10, 6))

# fire_risk_bands_60m layout
water_ndwi = ci.NDWI(img, i_green=1, i_nir=3)
water_mndwi = ci.MNDWI(img, i_green=1, i_swir=4)
moisture = ci.NDMI(img, i_swir=4, i_nir=3)
msi = ci.MSI(img, i_swir=4, i_nir=3)

for name, arr in [("NDWI", water_ndwi), ("MNDWI", water_mndwi), ("NDMI", moisture), ("MSI", msi)]:
    print(f"{name:6s} mean={arr.mean():+.3f}")
```

**Example 4 – stack the indices for many dates**

```python
import numpy as np
from caneblu import composite_indices as ci

dates_stack = np.random.default_rng(2).uniform(0.02, 0.6, size=(3, 64, 64, 6))  # 3 dates

ndvi_series = np.stack([ci.NDVI(dates_stack[t], i_red=2, i_nir=3) for t in range(3)])
print(ndvi_series.shape)            # (3, 64, 64)
print("change first->last:", round((ndvi_series[-1] - ndvi_series[0]).mean(), 4))
```

---

### `evalscript_generator_utils` – Sentinel Hub evalscripts

`get_evalscript(script, bands=None)` returns the JavaScript evalscript as a string. Pass one of the
predefined names, or `"custom_bands"` together with a list of band names.

Predefined names: `all_bands`, `all_bands_10m`, `fire_risk_bands_60m`, `bands_60m_rgb`,
`bands_60m_ir`, `ndvi`, `cloud_mask`, `false_color`.

**Example 1 – list the evalscript for a predefined layout**

```python
from caneblu.evalscript_generator_utils import get_evalscript

script = get_evalscript("fire_risk_bands_60m")
print("B02" in script and "B12" in script)   # True: six input bands
print("bands: 6" in script)                   # True: six output bands
```

**Example 2 – a custom band set**

```python
from caneblu.evalscript_generator_utils import get_evalscript

script = get_evalscript("custom_bands", bands=["B08", "B11"])
print('bands: ["B08", "B11"]' in script)      # True: the input band list
print("return [sample.B08, sample.B11];" in script)   # True: the evaluatePixel output
```

**Example 3 – unknown names fail loudly**

```python
from caneblu.evalscript_generator_utils import get_evalscript

try:
    get_evalscript("does_not_exist")
except ValueError as err:
    print("caught:", str(err)[:40], "...")
```

**Example 4 – pass the script to the harvester**

`SentinelHub.get_raw_bands(script)` is a thin wrapper around `get_evalscript`:

```python
from caneblu.harvester import SentinelHub

sh = SentinelHub(client_id="dummy", client_secret="dummy")   # no network call is made here
print(sh.get_raw_bands("ndvi").strip().splitlines()[0])
```

---

### `harvester` – Sentinel Hub downloads

`SentinelHub` wraps the Sentinel Hub catalog and processing API.

| Method | Purpose |
|---|---|
| `SentinelHub(...)` | Create a client (credentials, data collection, retry count) |
| `get_catalog(bbox, time_interval, cloud_cover)` | List matching acquisitions (network) |
| `request_box(bbox, ...)` | Download one array for one bounding box and time range (network) |
| `download_roi(bbox, ...)` | Download one array per acquisition, with local caching of `.npy` files (network) |

Bounding boxes can be a 4-tuple `(min_lon, min_lat, max_lon, max_lat)` in EPSG:4326, a list of
polygon vertices, a shapely `Polygon`, or a `sentinelhub.BBox`.

`download_roi` skips any acquisition whose `.npy` file already exists, so re-running a script
resumes instead of re-downloading.

**Example 1 – create a client and check how bounding boxes are converted (offline)**

```python
from shapely.geometry import Polygon
from sentinelhub import BBox
from caneblu.harvester import SentinelHub

sh = SentinelHub(client_id="dummy", client_secret="dummy")

from_tuple = sh._to_bbox((15.0, 37.0, 15.5, 37.5))
from_poly  = sh._to_bbox(Polygon([(15.0, 37.0), (15.5, 37.0), (15.5, 37.5), (15.0, 37.5)]))
from_verts = sh._to_bbox([(15.0, 37.0), (15.5, 37.0), (15.5, 37.5), (15.0, 37.5), (15.0, 37.0)])
from_bbox  = sh._to_bbox(BBox((15.0, 37.0, 15.5, 37.5), crs="EPSG:4326"))

print(from_tuple == from_poly == from_verts == from_bbox)   # True
```

**Example 2 – find acquisitions for a box (network)**

```python
# requires network and credentials
from caneblu.harvester import SentinelHub

sh = SentinelHub()   # SH_CLIENT_ID / SH_CLIENT_SECRET from the environment
catalog = sh.get_catalog(
    bbox=(15.0, 37.0, 15.5, 37.5),
    time_interval=("2024-06-01", "2024-06-30"),
    cloud_cover=10,           # percent
)
for item in catalog[:5]:
    print(item["properties"]["datetime"], item["properties"]["eo:cloud_cover"])
```

**Example 3 – download the full acquisition series for one box (network)**

```python
# requires network and credentials
from caneblu.harvester import SentinelHub

sh = SentinelHub()
arrays, files = sh.download_roi(
    bbox=(15.0, 37.0, 15.5, 37.5),
    base_path="tmp/area_",      # files: tmp/area__0000_<datetime>_10.npy
    index=0,
    resolution=10,
    time_interval=("2024-06-01", "2024-07-30"),
    cloud_cover=20,
    mosaic=False,               # True: one least-cloudy image per day
    save_numpy=True,
    data_type="all_bands_10m",  # see the band-order table
)

print("acquisitions:", len(arrays))
print("first array shape (H, W, bands):", arrays[0].shape)
```

**Example 4 – one mosaic per day, cached on disk (network)**

```python
# requires network and credentials
from caneblu.harvester import SentinelHub

sh = SentinelHub()
arrays, files = sh.download_roi(
    bbox=(15.0, 37.0, 15.5, 37.5),
    base_path="tmp/mosaic_",
    resolution=60,
    time_interval=("2025-06-15", "2025-06-30"),
    cloud_cover=10,
    mosaic=True,
    data_type="fire_risk_bands_60m",
)
# Run it again: existing .npy files are loaded instead of downloaded.
```

**Example 5 – download one square of a province (network, needs a boundaries shapefile)**

`download_province` splits a province into square tiles (see `geocoding.generate_squares_for_province`)
and downloads the requested tile range. Pass your own shapefile with `geo_file`. It must contain a
`SIGLA` column with province codes.

```python
# requires network, credentials and a province boundaries shapefile
from caneblu.harvester import download_province

results = download_province(
    prov="CT",
    side=3600,                              # square side in metres
    time_interval=("2025-06-15", "2025-06-30"),
    square_init=0,
    square_end=2,                           # tiles 0 and 1
    geo_file="data/ProvCM_WGS84.shp",
    resolution=60,
    cloud_cover=10,
    data_type="fire_risk_bands_60m",
    mosaic=True,
)
for square, (arrays, files) in results.items():
    print("square", square, "->", len(arrays), "images")
```

**Example 6 – turn downloaded arrays into fire-risk layers (offline)**

`format_for_fire_risk` expects the `fire_risk_bands_60m` layout (six bands) and returns four lists
of 2-D arrays for the first `n_dates` acquisitions:

```python
import numpy as np
from caneblu.harvester import format_for_fire_risk

rng = np.random.default_rng(3)
# three acquisitions, six bands: B02, B03, B04, B08, B11, B12
data = [rng.uniform(0.02, 0.5, size=(40, 40, 6)) for _ in range(3)]

ndvis, ndmis, rgbs_gray, nbrs = format_for_fire_risk(data, n_dates=3)
print(len(ndvis), ndvis[0].shape)     # 3 (40, 40)
```

**Example 7 – build the feature table used by the pipeline (offline)**

```python
import numpy as np
from caneblu.harvester import format_for_fire_risk
from caneblu.satellite_image_utils import create_features

rng = np.random.default_rng(4)
data = [rng.uniform(0.02, 0.5, size=(40, 40, 6)) for _ in range(3)]

ndvis, ndmis, rgbs_gray, nbrs = format_for_fire_risk(data, n_dates=3)
df = create_features(ndvis, ndmis, rgbs_gray, nbrs, n_dates=3)
print(df.shape)                       # (1600, 12): 4 features x 3 dates
print(list(df.columns)[:4])           # ['NDVI_0', 'NDMI_0', 'RGB_0', 'NBR_0']
```

**Example 8 – the same pipeline from the command line**

See [Command-line usage](#command-line-usage).

---

### `geocoding` – polygons, coordinates and squares

**Polygons and bounding boxes (offline)**

```python
from shapely.geometry import Polygon
from shapely.geometry import Point
from caneblu.geocoding import bbox_to_polygon, invert_polygon_latlon, find_point_within_polygon

box = bbox_to_polygon((15.0, 37.0, 15.5, 37.5))   # (min_lon, min_lat, max_lon, max_lat)
print(box.area)                         # 0.25 (degrees squared)

# swap lon/lat, e.g. to go from (lat, lon) data to (lon, lat) shapes
swapped = invert_polygon_latlon(Polygon([(37.0, 15.0), (37.5, 15.0), (37.5, 15.5), (37.0, 15.5)]))
print(swapped.bounds)                   # (15.0, 37.0, 15.5, 37.5)

# a point guaranteed to lie inside the polygon (safe for concave shapes)
x, y = find_point_within_polygon(box)
print(box.contains(Point(x, y)))        # True
```

**Overlap between two shapes from different sources (offline)**

```python
from shapely.geometry import Polygon
from caneblu.geocoding import check_polygons_overlap

official = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
digitised = Polygon([(0.2, 0.1), (10.1, 0.1), (10.1, 10.0), (0.2, 10.0)])

# True when each polygon covers at least 75 % of the other (defaults)
print(check_polygons_overlap(official, digitised))                      # True
print(check_polygons_overlap(official, digitised, overlap_min=0.99))    # False
```

**Random points inside a polygon (offline, seeded)**

```python
import random
from shapely.geometry import Polygon
from caneblu.geocoding import resample_within_polygon

random.seed(42)
ring = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
p = resample_within_polygon(ring)
print(ring.contains(p))                                                 # True
```

**Zones within a radius (offline)**

`find_zones_within_radius` measures distances in metres, so `crs` must be a **projected (metric)**
CRS such as `EPSG:32633` (UTM 33N, for Sicily). Convert the zones to that CRS first, and give the
point in the same CRS. Passing `EPSG:4326` would buffer by degrees and select everything.

```python
import geopandas as gpd
from shapely.geometry import Polygon
from caneblu.geocoding import find_zones_within_radius

zones = gpd.GeoDataFrame(
    {"name": ["near", "far"]},
    geometry=[
        Polygon([(15.00, 37.00), (15.01, 37.00), (15.01, 37.01), (15.00, 37.01)]),
        Polygon([(16.00, 38.00), (16.01, 38.00), (16.01, 38.01), (16.00, 38.01)]),
    ],
    crs="EPSG:4326",
).to_crs("EPSG:32633")

centre = zones.geometry.iloc[0].centroid        # a point in the same metric CRS
hits = find_zones_within_radius(point=centre, radius=5000, geo_data=zones, crs="EPSG:32633")
print(list(hits["name"]))                       # ['near']
```

**Split a province into squares (network-free, writes a file)**

`split_into_squares` uses the Sentinel Hub UTM splitter and writes a shapefile. Run it on your own
polygon:

```python
from shapely.geometry import box
from caneblu.geocoding import split_into_squares

area = box(15.0, 37.0, 15.3, 37.3)   # a polygon in EPSG:4326
squares = split_into_squares(shape=area, file_out="data/area_squares.shp", size=5000)
print(len(squares), "squares written")
```

**Province squares with a boundaries file (needs your shapefile)**

```python
# requires a province boundaries shapefile with a SIGLA column (your own file)
from caneblu.geocoding import generate_squares_for_province

gdf = generate_squares_for_province(
    prov="CT",
    side=7200,
    geo_file="data/ProvCM01012021_g_WGS84.shp",   # default path is the author's machine
)
# Result is cached to data/CT_squares_7200.shp; later calls read the cache.
print(len(gdf), "squares")
```

**Addresses and reverse geocoding (network)**

```python
# requires network (ArcGIS geocoder via geopandas)
from shapely.geometry import Point
from caneblu.geocoding import get_address_from_gps, bbox_from_address

addresses = get_address_from_gps([(37.5, 15.1)])        # (lat, lon) pairs or Points
print(addresses)

# Earth Engine region around an address; needs earthengine-api and ee.Initialize()
# region, point = bbox_from_address("Piazza Duomo, Catania", radius=2000)
```

---

### `satellite_image_utils` – rasters and features

**Grayscale and PNG preview (offline)**

```python
import numpy as np
from caneblu.satellite_image_utils import gray_to_rgb

index = np.linspace(-0.2, 0.9, 36).reshape(6, 6)    # e.g. an NDVI map
rgb = gray_to_rgb(index)                             # six-class colour map, uint8
print(rgb.shape, rgb.dtype)                          # (6, 6, 3) uint8
```

**Save and reload a GeoTIFF with georeferencing (offline, writes to the working directory)**

```python
import numpy as np
import rasterio
from shapely.geometry import box
from caneblu.satellite_image_utils import save_numpy_to_geotiff

arr = np.random.default_rng(5).uniform(0, 1, size=(20, 30, 3)).astype("float32")  # (H, W, bands)
roi = box(15.0, 37.0, 15.5, 37.5)                    # bounds of the array, EPSG:4326

save_numpy_to_geotiff(arr, "demo_roi.tif", roi=roi, crs="EPSG:4326")
with rasterio.open("demo_roi.tif") as src:
    print(src.count, src.width, src.height, src.bounds)
```

**Merge several GeoTIFFs into a mosaic (offline, uses the file above)**

```python
import numpy as np
from shapely.geometry import box
from caneblu.satellite_image_utils import save_numpy_to_geotiff, merge_georeferenced_tiffs

# two adjacent tiles, side by side in longitude
left = np.full((20, 30, 3), 0.2, dtype="float32")
right = np.full((20, 30, 3), 0.8, dtype="float32")
save_numpy_to_geotiff(left, "tile_left.tif", roi=box(15.0, 37.0, 15.5, 37.5), crs="EPSG:4326")
save_numpy_to_geotiff(right, "tile_right.tif", roi=box(15.5, 37.0, 16.0, 37.5), crs="EPSG:4326")

merge_georeferenced_tiffs(["tile_left.tif", "tile_right.tif"], "mosaic.tif")
```

**Crop a raster to a polygon (offline)**

`crop_tiff_with_polygon` returns `True` only if the crop is larger than 120 pixels on its side.

```python
import numpy as np
from shapely.geometry import box
from caneblu.satellite_image_utils import save_numpy_to_geotiff, crop_tiff_with_polygon

big = np.random.default_rng(6).uniform(0, 1, size=(400, 400, 3)).astype("float32")
save_numpy_to_geotiff(big, "big.tif", roi=box(15.0, 37.0, 16.0, 38.0), crs="EPSG:4326")

ok = crop_tiff_with_polygon("big.tif", "crop.tif", box(15.2, 37.2, 15.8, 37.8))
print("cropped:", ok)     # True
```

**Normalise, bin and brighten bands (offline)**

```python
import numpy as np
from caneblu.satellite_image_utils import normalize, bin_values, brighten

band = np.array([0.0, 2.0, 4.0, 8.0])
print(normalize(band))                    # [0.    0.25  0.5   1.   ]
print(bin_values(np.array([0.05, 0.3, 0.7, 0.99]), n_bins=5))   # [0.   0.25 0.75 1.  ]
print(brighten(np.array([100.0, 2000.0])))  # clipped to 0-255
```

**Burned-area mask from a dNBR map (offline)**

```python
import numpy as np
from caneblu.satellite_image_utils import burned_mask

dnbr = np.array([[0.1, 0.8], [0.5, 1.2]])
print(burned_mask(dnbr, threshold=0.75))   # 1.0 where dNBR > 0.75, otherwise 0.0
```

**Feature table from index stacks (offline)**

```python
import numpy as np
from caneblu.satellite_image_utils import create_features

rng = np.random.default_rng(7)
shape = (10, 10)
ndvis = [rng.uniform(-0.2, 0.8, shape) for _ in range(3)]
ndmis = [rng.uniform(-0.2, 0.6, shape) for _ in range(3)]
rgbs  = [rng.uniform(0, 255, shape) for _ in range(3)]
nbrs  = [rng.uniform(-0.2, 0.7, shape) for _ in range(3)]

df = create_features(ndvis, ndmis, rgbs, nbrs, n_dates=3)       # normalised per band
print(df.shape)                                                 # (100, 12)

# Add a binary label for supervised training
labels = (rng.uniform(size=100) > 0.9).astype(int)
df_labelled = create_features(ndvis, ndmis, rgbs, nbrs, dnbr_mask=labels.reshape(shape), n_dates=3)
print(df_labelled.columns[-1])                                  # BURNED
```

**Sort acquisitions around a fire date (offline)**

`sort_before_after` splits dates into those before and after an event, nearest first.

```python
from caneblu.satellite_image_utils import sort_before_after

dates = ["2024-06-01", "2024-06-10", "2024-07-05", "2024-07-20"]
before, after = sort_before_after(dates, "2024-07-01")
print(before)   # [(2024, 6, 10), (2024, 6, 1)]  nearest first
print(after)    # [(2024, 7, 5), (2024, 7, 20)]
```

**Compare two label maps (offline)**

```python
import numpy as np
from caneblu.satellite_image_utils import compare_arrays

pred = np.array([1, 0, 1, 1])
true = np.array([1, 1, 1, 0])
n_diff, pct = compare_arrays(pred, true)
print(n_diff, pct)      # 2 50.0
```

**Display several images side by side (interactive)**

```python
import numpy as np
from caneblu.satellite_image_utils import show_images_side_by_side

imgs = [np.random.default_rng(i).uniform(size=(32, 32)) for i in range(3)]
show_images_side_by_side(imgs, titles=["t0", "t1", "t2"], cmap="viridis")   # opens a window
```

**Write RGB images from a folder of Sentinel-2 TIFFs (needs your folder layout)**

```python
# requires your own folder layout: {base_path}/{sub_path}/SENTINEL2_NO_CLOUDS/*.tiff
from caneblu.satellite_image_utils import create_rgb_imgs

images = create_rgb_imgs(base_path="data", sub_paths=["CT_0001"], pxmax=11056)
print(len(images), "RGB images written")
```

---

### `open_meteo_api` – weather data

`OpenMeteoAPI` wraps the Open-Meteo archive (history) and forecast endpoints. Creating an instance
does not make a network call; calls happen when you use the data methods.

| Method | Purpose |
|---|---|
| `OpenMeteoAPI(lat, lon, n_days, date_end, ...)` | Set the location and window (`date_end` defaults to now) |
| `call_api(frequency, call_type, ...)` | Raw request to the archive or forecast endpoint (network) |
| `history()` | Hourly history for the window (network) |
| `forecast(tmp_path, forecast_days)` | Hourly forecast, cached on disk for one hour (network) |
| `recover_humidity()` | Estimate humidity with a random forest (needs scikit-learn, network) |

**Using an API key (commercial plan)**

Without a key, requests go to the free hosts (`archive-api.open-meteo.com`, `api.open-meteo.com`).
With a key, they go to the commercial hosts (`customer-archive-api.open-meteo.com`,
`customer-api.open-meteo.com`) and the key is sent as an `apikey` query parameter. The key is never
written into the URL, so it does not appear in timeout log messages. The key is taken from the
argument if given, otherwise from `OPEN_METEO_API_KEY`:

```bash
export OPEN_METEO_API_KEY="your-open-meteo-key"
```

```python
# requires network and an Open-Meteo API key
from caneblu.open_meteo_api import OpenMeteoAPI

# picks up OPEN_METEO_API_KEY from the environment
om = OpenMeteoAPI(lat=37.5, lon=15.1, date_end="2024-07-31", n_days=10)

# or pass the key explicitly
om_explicit = OpenMeteoAPI(lat=37.5, lon=15.1, date_end="2024-07-31", n_days=10,
                           api_key="your-open-meteo-key")

daily = om.call_api(frequency="daily", call_type="archive")      # customer-archive-api
hourly = om.history()                                            # customer-archive-api
forecast = om.forecast(tmp_path="tmp/", forecast_days=2)         # customer-api, cached 1 h
```

Get a key from the [Open-Meteo pricing page](https://open-meteo.com/en/pricing).

**Example 1 – construct and inspect (offline)**

```python
from caneblu.open_meteo_api import OpenMeteoAPI

om = OpenMeteoAPI(lat=37.5, lon=15.1, date_end="2024-07-31", n_days=10)
print(om.latitude(), om.longitude())        # 37.5 15.1
print(om.date_start.format("YYYY-MM-DD"))   # 2024-07-21
om.info()                                   # prints the date and location
```

**Example 2 – daily archive for a fire date (network)**

```python
# requires network
from caneblu.open_meteo_api import OpenMeteoAPI

om = OpenMeteoAPI(lat=37.5, lon=15.1, date_end="2024-07-31", n_days=10)
daily = om.call_api(frequency="daily", call_type="archive")
print(daily["time"][:3], daily["temperature_2m_max"][:3])
```

**Example 3 – hourly history before an event (network)**

```python
# requires network
from caneblu.open_meteo_api import OpenMeteoAPI

om = OpenMeteoAPI(lat=37.5, lon=15.1, date_end="2024-07-31", n_days=3)
hourly = om.history()                 # hourly temperature, wind, humidity, ET0
print(len(hourly["time"]), "hours")
```

**Example 4 – forecast with a one-hour disk cache (network)**

```python
# requires network
from caneblu.open_meteo_api import OpenMeteoAPI

om = OpenMeteoAPI(lat=37.5, lon=15.1, prov="CT", index=0, side=3600)
forecast = om.forecast(tmp_path="tmp/", forecast_days=2)
# The second call within an hour reads tmp/CT_0_3600_forecast_cache.json.
```

**Example 5 – many locations, one object each (offline)**

```python
from caneblu.open_meteo_api import OpenMeteoAPI

sites = [
    {"lat": 37.5, "lon": 15.1, "date": "2024-07-31"},
    {"lat": 38.0, "lon": 14.7, "date": "2024-08-12"},
]
objects = [
    OpenMeteoAPI(lat=s["lat"], lon=s["lon"], date_end=s["date"], n_days=10) for s in sites
]
print([o.date_start.format("YYYY-MM-DD") for o in objects])
```

---

### `utils` – general helpers

**Filenames used by the downloader (offline)**

```python
from caneblu import utils

print(utils.npy_filename("tmp/area_", 0, "2024-06-01", res=10))
# tmp/area__0000_2024-06-01_10.npy
print(utils.npy_mosaic_filename("tmp/area_", 0, "2024-06-01", res=10))
# tmp/area__0000_2024-06-01_10_mosaic.npy
```

**Remove duplicate catalog entries (offline)**

Sentinel Hub often returns the same acquisition twice. The helper keeps the first occurrence of each
ID suffix, in order:

```python
from caneblu.utils import remove_duplicates_from_list

items = [
    {"id": "S2A_MSIL2A_20240601_T33SVB_123"},
    {"id": "S2B_MSIL2A_20240601_T33SVB_123"},   # same suffix "123": dropped
    {"id": "S2A_MSIL2A_20240605_T33SVB_456"},
]
print([i["id"] for i in remove_duplicates_from_list(items)])
```

**JSON and GeoJSON (offline, writes a file)**

```python
from caneblu.utils import dump_geojson, read_json_file

feature_collection = {"type": "FeatureCollection", "features": []}
dump_geojson(feature_collection, "areas.geojson")
print(read_json_file("areas.geojson")["type"])     # FeatureCollection
print(read_json_file("missing.json"))              # None, and a message is printed
```

**Time conversions (offline)**

```python
from caneblu.utils import time_to_utc, utc_to_time, modis_to_celsius, fix_time

ts = time_to_utc("2024-07-01")           # integer timestamp, local midnight
print(utc_to_time(ts))                   # 2024-07-01 00:00:00 (datetime)

print(fix_time(["2024-07-01T10:00:00Z", "2024-07-02T11:30:00Z"]))   # ['2024-07-01', '2024-07-02']
print(round(modis_to_celsius(140000), 2))                            # 6.85  (140000 * 0.002 - 273.15)
```

**Cache age check (offline, uses a file created above)**

```python
from caneblu.utils import dump_geojson, is_file_older_than

dump_geojson({"type": "FeatureCollection", "features": []}, "fresh.geojson")
print(is_file_older_than("fresh.geojson", hours=1))   # False: the file was just written
```

---

## End-to-end workflow

A complete offline-ready pipeline for one tile, once data has been downloaded. Replace the random
arrays with real downloads from `download_province` to run it on real data.

```python
import numpy as np
import pandas as pd

from caneblu import composite_indices as ci
from caneblu.harvester import format_for_fire_risk
from caneblu.satellite_image_utils import create_features, bin_values

# 1. Input: three acquisitions of a 60 m tile, layout fire_risk_bands_60m
rng = np.random.default_rng(8)
acquisitions = [rng.uniform(0.02, 0.5, size=(60, 60, 6)) for _ in range(3)]

# 2. Indices and grayscale layers for the first three dates
ndvis, ndmis, rgbs_gray, nbrs = format_for_fire_risk(acquisitions, n_dates=3)

# 3. Feature table: one row per pixel, four features per date
features = create_features(ndvis, ndmis, rgbs_gray, nbrs, n_dates=3)

# 4. Turn the change in NDVI into a coarse 5-level risk class per pixel
ndvi_change = np.abs(ndvis[-1] - ndvis[0])
risk = bin_values(ndvi_change.ravel() / max(ndvi_change.max(), 1e-9), n_bins=5)
features["RISK_CLASS"] = risk

summary = features.groupby("RISK_CLASS").size()
print(summary)
print(features.head(3))
```

Add the Sentinel Hub download before step 1 when you have credentials:

```python
# requires network and credentials
from caneblu.harvester import SentinelHub

sh = SentinelHub()
acquisitions, files = sh.download_roi(
    bbox=(15.0, 37.0, 15.5, 37.5),
    resolution=60,
    time_interval=("2025-06-15", "2025-06-30"),
    cloud_cover=10,
    mosaic=True,
    data_type="fire_risk_bands_60m",
)
```

---

## Command-line usage

The harvester has a small CLI that downloads squares of a province:

```bash
python -m caneblu.harvester -p CT -s 3600 -cc 10 -di 2025-06-15 -de 2025-06-30 \
    -r 60 -m true -si 0 -se 2 -dt fire_risk_bands_60m
```

| Flag | Meaning | Default |
|---|---|---|
| `-p, --province` | Province code | `CT` |
| `-s, --side` | Square side in metres | `3600` |
| `-cc, --cloud_cover` | Maximum cloud cover in percent | `50` |
| `-di, --date_init` | Start date | `2021-07-01` |
| `-de, --date_end` | End date | `2021-08-30` |
| `-r, --resolution` | Resolution in metres | `10` |
| `-m, --mosaic` | `true`/`false`: one least-cloudy image per day | `false` |
| `-t, --time_span` | Passed through to `download_roi` | `10` |
| `-si, --square_init` | First square index (inclusive) | `0` |
| `-se, --square_end` | Last square index (exclusive) | `1` |
| `-dt, --data_type` | Evalscript name | `all_bands_10m` |

The CLI reads the boundaries file from the default path in `geocoding.DEFAULT_PROVINCE_FILE`. Edit that
constant, or call `download_province(..., geo_file=...)` from Python, to point it at your own file.

---

## Troubleshooting

- **`Warning: no Sentinel Hub credentials`** – set `SH_CLIENT_ID` and `SH_CLIENT_SECRET`, or pass them
  to `SentinelHub(...)`.
- **`FileNotFoundError` for a province shapefile** – the default path belongs to the author's machine.
  Pass `geo_file=` to `generate_squares_for_province` or `download_province`.
- **Index values look wrong** – check the band-order table. `NDVI(img, i_red=2, i_nir=3)` is correct for
  `fire_risk_bands_60m`, but for `all_bands_10m` the red band is at index 0.
- **`ImportError` for `ee`** – only `geocoding.bbox_from_address` needs `earthengine-api`.
- **`ImportError` for `sklearn`** – only `OpenMeteoAPI.recover_humidity` needs scikit-learn.
- **Downloads resume slowly or repeat** – `download_roi` reuses existing `.npy` files only when the
  filename matches (same `base_path`, `index`, date, and resolution). Changing any of them re-downloads.
- **`ValueError: Unknown script`** – use one of the names in
  [evalscript_generator_utils](#evalscript_generator_utils--sentinel-hub-evalscripts), or `custom_bands`
  with `bands=[...]`.

---

## Project layout

```
caneblu/                       repository root
  caneblu/                     the Python package
    __init__.py                lazy submodule loading, __version__
    composite_indices.py       spectral indices
    evalscript_generator_utils.py  Sentinel Hub evalscripts
    geocoding.py               polygons, coordinates, squares
    harvester.py               SentinelHub client, download_province, CLI
    open_meteo_api.py          Open-Meteo wrapper
    satellite_image_utils.py   rasters, features, cropping
    utils.py                   filenames, JSON, time helpers
  example.py                   minimal download script (reads credentials from the environment)
  setup.py, requirements.txt, LICENSE, README.md
```

Install in editable mode with `pip install -e .` to edit the code and see the changes immediately.

---

## License

MIT License. See the `LICENSE` file for details.
