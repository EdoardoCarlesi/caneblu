import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
from sentinelhub import (
    BBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions)

from shapely.geometry import Polygon

from . import utils
from . import composite_indices as ci
from . import satellite_image_utils as siu
from .evalscript_generator_utils import get_evalscript


class SentinelHub:
    """
    Sentinel Hub client: search the catalog and download satellite data for a bounding box.

    Credentials come from `client_id`/`client_secret`, the environment variables
    `SH_CLIENT_ID`/`SH_CLIENT_SECRET`, or a JSON file with CLIENT_ID/CLIENT_SECRET keys.

    Example:
        >>> from caneblu.harvester import SentinelHub
        >>> sh = SentinelHub()
        >>> data, files = sh.download_roi((15.0, 37.0, 15.5, 37.5), resolution=10,
        ...                               time_interval=("2024-06-01", "2024-07-30"))
    """

    def __init__(
        self,
        instance_id="default-instance-id",
        max_download_attempts=5,
        crs="EPSG:4326",
        credentials_file_json="json/.sentinelhub_credentials.json",
        data_collection="Sentinel2L1C",
        client_id=None,
        client_secret=None,
    ):

        self.crs = crs
        self.cred_file = credentials_file_json

        # Default
        self.data_collection = DataCollection.SENTINEL2_L1C

        if data_collection == "Sentinel2L1C":
            self.data_collection = DataCollection.SENTINEL2_L1C

        self.config = SHConfig()
        self.config_auto(client_id, client_secret)
        self.config.instance_id = instance_id
        self.config.max_download_attempts = max_download_attempts
        self.catalog = SentinelHubCatalog(config=self.config)

    def config_auto(self, client_id=None, client_secret=None):

        if client_id and client_secret:
            self.config.sh_client_id = client_id
            self.config.sh_client_secret = client_secret
            return

        try:
            with open(self.cred_file, "r") as credentials:
                cred = json.load(credentials)
                self.config.sh_client_id = cred["CLIENT_ID"]
                self.config.sh_client_secret = cred["CLIENT_SECRET"]

        except FileNotFoundError:
            # SHConfig() already reads SH_CLIENT_ID / SH_CLIENT_SECRET from the environment
            if not self.config.sh_client_id:
                print(
                    "Warning: no Sentinel Hub credentials. Pass client_id/client_secret, "
                    "set SH_CLIENT_ID and SH_CLIENT_SECRET, or create the credentials file."
                )

    def _to_bbox(self, bounding_box):
        """Convert various bounding box formats to a SentinelHub BBox object.

        Accepts:
        - BBox: returned as-is
        - Polygon: extracts bounds and creates BBox
        - tuple/list of 4 values: (min_lon, min_lat, max_lon, max_lat)
        - tuple/list of 5+ values (polygon coords): uses first and last point's min/max
        """
        if isinstance(bounding_box, BBox):
            return bounding_box

        if isinstance(bounding_box, Polygon):
            return BBox(bounding_box.bounds, crs=self.crs)

        if isinstance(bounding_box, (tuple, list)):
            if len(bounding_box) == 4:
                return BBox(bounding_box, crs=self.crs)
            if len(bounding_box) >= 4:
                lons = [p[0] for p in bounding_box]
                lats = [p[1] for p in bounding_box]
                return BBox((min(lons), min(lats), max(lons), max(lats)), crs=self.crs)

        raise TypeError(f"Unsupported bounding box type: {type(bounding_box)}")

    def get_raw_bands(self, script="all_bands_10m"):
        return get_evalscript(script)

    def get_catalog(
        self, bounding_box, time_interval=("2020-06-12", "2020-06-13"), cloud_cover=0.2
    ):

        bbox = self._to_bbox(bounding_box)
        search_iterator = self.catalog.search(
            self.data_collection,
            bbox=bbox,
            time=time_interval,
            filter=f"eo:cloud_cover < {cloud_cover}",
            fields={
                "include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
                "exclude": [],
            },
        )

        return list(search_iterator)

    def get_config(self):
        return self.config

    def request_box(
        self,
        bounding_box,
        mosaic=False,
        data_type="all_bands_10m",
        resolution=10,
        time_interval=("2020-06-12", "2020-06-13"),
    ):

        bbox = self._to_bbox(bounding_box)
        size = bbox_to_dimensions(bbox, resolution=resolution)
        script = self.get_raw_bands(script=data_type)

        if mosaic:
            sentinel_input_data = [
                SentinelHubRequest.input_data(
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                    data_collection=self.data_collection,
                    time_interval=time_interval,
                )
            ]
        else:
            sentinel_input_data = [
                SentinelHubRequest.input_data(
                    data_collection=self.data_collection,
                    time_interval=time_interval
                )
            ]

        self.sentinel_request = SentinelHubRequest(
            evalscript=script,
            input_data=sentinel_input_data,
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=self.config,
        )

        self.data = self.sentinel_request.get_data()

        # To avoid weird array dimensions later on
        return self.data[0]

    def download_roi(
        self,
        bbox,
        base_path='tmp/area_',
        index=0,
        resolution=10,
        time_interval=("2024-06-01", "2024-07-30"),
        cloud_cover=0.1,
        mosaic=False,
        save_numpy=True,
        time_span=10,
        data_type="all_bands_10m",
    ) -> tuple:
        """
        Wrapper to extract all the historical data from a province, once it's divided into squares
        """

        # Local data paths containing necessary data. Define all the variables read from the cfg file here.
        resolution = int(resolution)
        mosaic = bool(mosaic)
        time_span = int(time_span)

        # Generate a sentinel hub object for the kind of data required
        self.all_data = []
        self.all_output_files = []

        # For consistency throughout the program, the filename is generated in the data_io module
        catalog = self.get_catalog(
            bbox, time_interval=time_interval, cloud_cover=cloud_cover
        )
    
        catalog = utils.remove_duplicates_from_list(catalog)

        for entry in catalog:
            date = entry["properties"]["datetime"]
            try:
                formatted_date = datetime.fromisoformat(date).strftime("%Y-%m-%d")  # Format the date
            except:
                date = date.replace('Z', '+00:00')
                formatted_date = datetime.fromisoformat(date).strftime("%Y-%m-%d")  # Format the date

            day_before = (datetime.fromisoformat(date) - timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )  # Day before
            day_after = (datetime.fromisoformat(date) + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )  # Day after

            this_time_interval = (day_before, day_after)
            
            if mosaic:
                output_file = utils.npy_mosaic_filename(base_path, index, date, res=resolution)
            else:
                output_file = utils.npy_filename(base_path, index, date, res=resolution)
            
            if os.path.exists(output_file):
                print(f"File {output_file} found")
                data = np.load(output_file)

            else:
                print(f"Harvesting Sentinel Hub data for {base_path} on {formatted_date}")
                data = self.request_box(
                    bbox,
                    data_type=data_type,
                    resolution=resolution,
                    time_interval=this_time_interval,
                    mosaic=mosaic,
                )

                if save_numpy:
                    print(f"Saving to: {output_file}, sanity check on data shape: {data.shape}\n")
                    np.save(output_file, data)

            self.all_output_files.append(output_file)
            self.all_data.append(data)

        return self.all_data, self.all_output_files


def format_for_fire_risk(data, n_dates=3) -> tuple:
    """
    Compute NDVI, NDMI, NBR and grayscale RGB for the first `n_dates` images.

    :param data: List of Sentinel-2 arrays (bands B02, B03, B04, B08, B11, B12 as produced by
                 `SentinelHub.download_roi`), indexed by date.
    :return: (ndvis, ndmis, rgbs_gray, nbrs), each a list of 2D arrays.
    """

    i_red, i_nir, i_swir, i_swir2 = 2, 3, 4, 5
    ndvis, ndmis, rgbs_gray, nbrs = [], [], [], []

    for i in range(0, n_dates):
        rgb_gray = siu.rgb_to_gray(data[i][:, :, 0:3])
        ndvi = ci.NDVI(data[i], i_nir, i_red)
        ndmi = ci.NDMI(data[i], i_swir, i_nir)
        nbr = ci.NBR(data[i], i_swir2, i_nir)

        rgbs_gray.append(rgb_gray)
        ndvis.append(ndvi)
        ndmis.append(ndmi)
        nbrs.append(nbr)

    return (ndvis, ndmis, rgbs_gray, nbrs)


def parse_params() -> dict:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="Sentinel Hub Data Harvesting")

    # List of arguments to be parsed
    parser.add_argument("-p", "--province", type=str, dest="prov", default="CT")
    parser.add_argument("-s", "--side", type=int, dest="side", default=3600)
    parser.add_argument(
        "-cc", "--cloud_cover", type=int, dest="cloud_cover", default=50
    )
    parser.add_argument(
        "-di", "--date_init", type=str, dest="date_init", default="2021-07-01"
    )
    parser.add_argument(
        "-de", "--date_end", type=str, dest="date_end", default="2021-08-30"
    )
    parser.add_argument("-r", "--resolution", type=int, dest="resolution", default=10)
    parser.add_argument("-m", "--mosaic", type=_str2bool, dest="mosaic", default=False)
    parser.add_argument("-t", "--time_span", type=int, dest="time_span", default=10)
    parser.add_argument("-si", "--square_init", type=int, dest="square_init", default=0)
    parser.add_argument("-se", "--square_end", type=int, dest="square_end", default=1)
    parser.add_argument(
        "-dt", "--data_type", type=str, dest="data_type", default="all_bands_10m"
    )

    args = parser.parse_args()

    return vars(args)


def download_province(
    prov,
    side,
    time_interval,
    square_init=0,
    square_end=1,
    sh=None,
    geo_file=None,
    **download_kwargs,
) -> dict:
    """
    Download Sentinel Hub data for a range of squares covering a province.

    The province is split into `side` x `side` metre squares (see
    `geocoding.generate_squares_for_province`); squares `square_init` (inclusive)
    to `square_end` (exclusive) are downloaded with `SentinelHub.download_roi`.

    :param prov: Province code, e.g. "CT".
    :param side: Square side length in metres.
    :param time_interval: (start, end) date strings, "YYYY-MM-DD".
    :param square_init: First square index to download.
    :param square_end: One past the last square index to download.
    :param sh: Optional existing `SentinelHub` instance (created if None).
    :param geo_file: Optional path to the province boundaries shapefile.
    :param download_kwargs: Extra keyword arguments forwarded to `download_roi`
                            (e.g. resolution, cloud_cover, mosaic, data_type, base_path).
    :return: {square_index: (list_of_arrays, list_of_npy_files)}
    """
    from . import geocoding

    squares = geocoding.generate_squares_for_province(prov=prov, side=side, geo_file=geo_file)
    sh = sh or SentinelHub()

    results = {}
    for index in range(square_init, square_end):
        bbox = squares.geometry.iloc[index].bounds  # (min_lon, min_lat, max_lon, max_lat)
        results[index] = sh.download_roi(
            bbox,
            base_path=f"tmp/{prov}_{side}_",
            index=index,
            time_interval=time_interval,
            **download_kwargs,
        )

    return results


def _str2bool(value) -> bool:
    """argparse helper: accepts 'true/false', 'yes/no', '1/0'."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def parse_start():
    """Command-line entry point: `python -m caneblu.harvester -p CT -s 3600 ...`"""

    params = parse_params()

    download_province(
        prov=params["prov"],
        side=params["side"],
        time_interval=(params["date_init"], params["date_end"]),
        square_init=params["square_init"],
        square_end=params["square_end"],
        resolution=params["resolution"],
        cloud_cover=params["cloud_cover"],
        data_type=params["data_type"],
        mosaic=params["mosaic"],
        time_span=params["time_span"],
    )


def param_start(prov, sh=None):
    """
    Example pipeline: harvest one square of a province at 60 m resolution and
    build the fire-risk feature table.

    :return: pandas DataFrame of features (see `satellite_image_utils.create_features`).
    """
    n_dates = 3

    results = download_province(
        prov=prov,
        side=3600,
        time_interval=("2025-06-15", "2025-06-30"),
        square_init=0,
        square_end=1,
        sh=sh,
        resolution=60,
        cloud_cover=10,
        data_type="fire_risk_bands_60m",
        mosaic=True,
    )
    data, _files = results[0]

    ndvis, ndmis, rgbs_gray, nbrs = format_for_fire_risk(data, n_dates=n_dates)
    return siu.create_features(ndvis, ndmis, rgbs_gray, nbrs, n_dates=n_dates)


__all__ = [
    "SentinelHub",
    "download_province",
    "format_for_fire_risk",
    "parse_params",
    "parse_start",
    "param_start",
]


if __name__ == "__main__":
    parse_start()
