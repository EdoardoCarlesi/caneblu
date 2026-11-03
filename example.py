"""
Minimal end-to-end example: download Sentinel-2 data for one bounding box.

Credentials are read from the environment (never hard-code them):

    export SH_CLIENT_ID="your-client-id"
    export SH_CLIENT_SECRET="your-client-secret"
    python example.py
"""

from caneblu.harvester import SentinelHub

sh = SentinelHub()  # picks up SH_CLIENT_ID / SH_CLIENT_SECRET from the environment

# (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
bbox = (15.0, 37.0, 15.5, 37.5)

data, files = sh.download_roi(
    bbox=bbox,
    base_path="tmp/area_",
    index=0,
    resolution=10,
    time_interval=("2024-06-01", "2024-07-30"),
    cloud_cover=20,
    mosaic=False,
    save_numpy=True,
    data_type="all_bands_10m",
)

print(f"Downloaded {len(data)} acquisitions, saved {len(files)} files")
