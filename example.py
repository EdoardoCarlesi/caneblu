from shapely.geometry import Polygon
from sentinelhub import BBox
from caneblu.harvester import SentinelHub

sh = SentinelHub(
    client_id="REDACTED_CLIENT_ID",
    client_secret="REDACTED_SECRET"
)

bbox_tuple = (15.0, 37.0, 15.5, 37.5)
bbox_list = [(15.0, 37.0), (15.5, 37.0), (15.5, 37.5), (15.0, 37.5), (15.0, 37.0)]
bbox_polygon = Polygon(bbox_list)
bbox_sh = BBox(bbox=(15.0, 37.0, 15.5, 37.5), crs="EPSG:4326")

data, files = sh.download_roi(
    bbox=bbox_tuple,
    base_path="tmp/area_",
    index=0,
    resolution=10,
    time_interval=("2024-06-01", "2024-07-30"),
    cloud_cover=20,
    mosaic=False,
    save_numpy=True,
    #data_type="all_bands_10m"
    data_type="all_bands_10m"
)
