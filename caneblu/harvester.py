import argparse
import json
import os
import pickle
import numpy as np
from datetime import datetime, timedelta

import geopandas as gpd
from sentinelhub import (
    BBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions)

from . import utils
from . import composite_indices as ci
from . import satellite_image_utils as siu
from .evalscript_generator_utils import get_evalscript


class SentinelHub:
    """
    A mock class to represent the SentinelHub functionalities referenced in the script.
    """

    def __init__(
        self,
        instance_id="default-instance-id",
        max_download_attempts=5,
        crs="EPSG:4326",
        credentials_file_json="json/.sentinelhub_credentials.json",
        data_collection="Sentinel2L1C",
    ):

        self.crs = crs
        self.cred_file = credentials_file_json

        # Default
        self.data_collection = DataCollection.SENTINEL2_L1C

        if data_collection == "Sentinel2L1C":
            self.data_collection = DataCollection.SENTINEL2_L1C

        self.config = SHConfig()
        self.config_auto()
        self.config.instance_id = instance_id
        self.config.max_download_attempts = max_download_attempts
        self.catalog = SentinelHubCatalog(config=self.config)

    def config_auto(self):

        try:
            with open(self.cred_file, "r") as credentials:
                cred = json.load(credentials)
                self.config.sh_client_id = cred["CLIENT_ID"]
                self.config.sh_client_secret = cred["CLIENT_SECRET"]

        except FileNotFoundError:
            print("Error: sentinelhub_credentials.json file not found.")

    def get_raw_bands(self, script="all_bands_10m"):
        return get_evalscript(script)

    def get_catalog(
        self, bounding_box, time_interval=("2020-06-12", "2020-06-13"), cloud_cover=0.2
    ):

        bbox = BBox(bounding_box.bounds, crs=self.crs)
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

        bbox = BBox(bounding_box.bounds, crs=self.crs)
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

        string = self.catalog.get_collections()
        json_str = json.dumps(string, indent=4)
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
        :param: res, the resolution (in meters) for this
        :param: data_type
        :param: prov
        :param: time_interval
        :param: n_square_init
        :param: n_square_end
        """

        # Local data paths containing necessary data. Define all the variables read from the cfg file here.
        resolution = int(resolution)
        cloud_cover = int(cloud_cover)
        mosaic = bool(mosaic)
        time_span = int(time_span)

        print(f"Downloading tile: {index} for {prov}, {time_interval}")

        # Generate a sentinel hub object for the kind of data required
        self.all_data = []
        self.all_output_files = []

        # For consistency throughout the program, the filename is generated in the data_io module
        catalog = self.get_catalog(
            bbox, time_interval=time_interval, cloud_cover=cloud_cover
        )
    
        catalog = utils.remove_duplicates_from_list(catalog)
        catalog_str = json.dumps(catalog, indent=4)

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
                output_file = utils.npy_mosaic_filename(prov, index, date)
            else:
                output_file = utils.npy_filename(prov, index, date)
            
            if os.path.exists(output_file):
                print(f"File {output_file} found")
                data = np.load(output_file)

            else:
                print(f"Harvesting Sentinel Hub data for {prov} on {formatted_date}")
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

    i_blue, i_green, i_red, i_nir, i_swir, i_swir2 = 0, 1, 2, 3, 4, 5
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
    parser.add_argument("-m", "--mosaic", type=bool, dest="mosaic", default=False)
    parser.add_argument("-t", "--time_span", type=int, dest="time_span", default=10)
    parser.add_argument("-si", "--square_init", type=str, dest="square_init", default=0)
    parser.add_argument("-se", "--square_end", type=str, dest="square_end", default=1)
    parser.add_argument(
        "-dt", "--data_type", type=str, dest="data_type", default="all_bands_10m"
    )

    args = parser.parse_args()

    return vars(args)


def parse_start():

    # First read, parse and format the input parameters in a dictionary
    params = parse_params()

    sentinel_download_province(
        resolution=params["resolution"],
        time_interval=(params["date_init"], params["date_end"]),
        prov=params["prov"],
        side=params["side"],
        time_span=params["time_span"],
        square_init=params["square_init"],
        square_end=params["square_end"],
        cloud_cover=params["cloud_cover"],
        data_type=params["data_type"],
        mosaic=params["mosaic"],
    )


def param_start(prov):

    dinit, dend = "2025-06-15", "2025-06-30"
    square_init = 0
    square_end = square_init + 1
    resolution = 60
    data_type = "fire_risk_bands_60m"
    do_harvest = True
    mosaic = True
    side = 3600
    cloud_cover = 10
    n_dates = 3

    if do_harvest:
        data, files = sentinel_download_province(
            resolution=resolution,
            time_interval=(dinit, dend),
            prov=prov,
            side=side,
            square_init=square_init,
            square_end=square_end,
            data_type=data_type,
            cloud_cover=cloud_cover,
            mosaic=mosaic,
        )

    (ndvis, ndmis, rgbs_gray, nbrs) = format_for_fire_risk(data)
    data_df = siu.create_features(ndvis, ndmis, rgbs_gray, nbrs, n_dates=n_dates)

    print(data_df.head())


if __name__ == "__main__":
    """Go!"""

    param_start(prov="CT")
