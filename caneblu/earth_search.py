"""
Key-less satellite data from the Earth Search STAC API (Element 84, AWS Open Data).

Supported collections: Sentinel-2 L1C ("sentinel-2-l1c") and L2A ("sentinel-2-l2a").
Assets are Cloud-Optimised GeoTIFFs read directly from S3 with unsigned requests, so no
account or key is needed.

Not supported here: Landsat (the usgs-landsat bucket is requester-pays) and Sentinel-1 GRD
(uncalibrated DN with ground control points, so it needs calibration and geocoding first).

Output arrays follow the same layout as `SentinelHub.request_box`: (height, width, bands),
with the same band order as the Sentinel Hub evalscripts.
"""

import math

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"

# caneblu collection name (as used by SentinelHub(data_collection=...)) -> Earth Search collection
COLLECTIONS = {
    "Sentinel2L1C": "sentinel-2-l1c",
    "Sentinel2L2A": "sentinel-2-l2a",
}

# Sentinel-2 band name -> Earth Search asset key
S2_ASSETS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
}

# data_type -> band names, in the same order as the Sentinel Hub evalscripts
DATA_TYPE_BANDS = {
    "all_bands_10m": ["B04", "B03", "B02", "B08"],
    "all_bands": ["B04", "B03", "B02", "B08"],
    "fire_risk_bands_60m": ["B02", "B03", "B04", "B08", "B11", "B12"],
    "bands_60m_rgb": ["B02", "B03", "B04"],
    "bands_60m_ir": ["B08", "B11", "B12"],
    "false_color": ["B08", "B04", "B03"],
}


def _datetime_range(time_interval) -> str:
    start, end = time_interval
    return f"{start}T00:00:00Z/{end}T23:59:59Z"


def _utm_crs(bbox) -> str:
    """UTM CRS of the bounding box centre."""
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def search(collection, bbox, time_interval, cloud_cover=None) -> list:
    """
    Search Earth Search for items intersecting `bbox` (min_lon, min_lat, max_lon, max_lat).

    :param collection: Earth Search collection id, e.g. "sentinel-2-l2a".
    :param cloud_cover: Optional maximum cloud cover in percent (strict "<").
    :return: list of dicts with keys "id", "properties" and "assets", the same shape
             that `SentinelHub.get_catalog` returns for the fields it needs.
    """
    body = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": _datetime_range(time_interval),
        "limit": 100,
    }
    if cloud_cover is not None:
        body["query"] = {"eo:cloud_cover": {"lt": cloud_cover}}

    items = []
    while True:
        response = requests.post(f"{EARTH_SEARCH_URL}/search", json=body, timeout=60)
        response.raise_for_status()
        page = response.json()
        items.extend(page.get("features", []))

        next_link = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if next_link is None:
            break
        body = {**body, **next_link.get("body", {})}

    return [
        {"id": item["id"], "properties": item["properties"], "assets": item["assets"]}
        for item in items
    ]


def request_box(collection, bbox, data_type="all_bands_10m", resolution=10,
                time_interval=("2020-06-12", "2020-06-13"), mosaic=False) -> np.ndarray:
    """
    Read one array for `bbox` and the time range, mosaicking every item that covers it.

    Pixels are filled from the first item that has valid data there: least cloudy first when
    `mosaic=True`, most recent first otherwise (the same choice `SentinelHub.request_box` makes).

    :return: uint16 array of shape (height, width, bands) on a UTM grid at `resolution` metres.
    """
    if data_type not in DATA_TYPE_BANDS:
        raise ValueError(
            f"data_type '{data_type}' is not available from earth_search. "
            f"Valid options: {sorted(DATA_TYPE_BANDS)}."
        )
    bands = DATA_TYPE_BANDS[data_type]
    keys = [S2_ASSETS[band] for band in bands]

    items = search(collection, bbox, time_interval)
    if mosaic:
        items.sort(key=lambda it: it["properties"].get("eo:cloud_cover", 100))
    else:
        items.sort(key=lambda it: it["properties"]["datetime"], reverse=True)

    crs = _utm_crs(bbox)
    west, south, east, north = transform_bounds("EPSG:4326", crs, *bbox)
    width = max(1, math.ceil((east - west) / resolution))
    height = max(1, math.ceil((north - south) / resolution))
    transform = from_origin(west, north, resolution, resolution)

    out = np.zeros((height, width, len(bands)), dtype=np.uint16)
    valid = np.zeros((height, width), dtype=bool)

    # Unsigned S3 reads; skipping directory listings keeps each COG open fast
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for item in items:
            if not all(key in item["assets"] for key in keys):
                continue

            stack = np.stack(
                [_read_warped(item["assets"][key]["href"], crs, transform, width, height)
                 for key in keys],
                axis=-1,
            )
            fill = (stack > 0).all(axis=-1) & ~valid
            out[fill] = stack[fill]
            valid |= fill
            if valid.all():
                break  # every pixel is filled; remaining items would change nothing

    return out


def _read_warped(href, crs, transform, width, height) -> np.ndarray:
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=width, height=height,
                       resampling=Resampling.nearest, nodata=0) as vrt:
            return vrt.read(1)


__all__ = ["EARTH_SEARCH_URL", "COLLECTIONS", "S2_ASSETS", "DATA_TYPE_BANDS", "search", "request_box"]
