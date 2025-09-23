# Do the reverse geocoding operation - for each center of the list above get
import os
import random

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from sentinelhub import BBox, UtmZoneSplitter
from shapely.geometry import MultiPolygon, Point, Polygon
from tqdm import tqdm


def split_into_squares(
    shape: Polygon, file_out: str, size=5000, crs="epsg:4326", wgs84=True
) -> gpd.GeoDataFrame:
    """
    Input:
    :param: shape, shapely polygon of an area to be split into squares

    :param: size, float - meters for the size of the square to be taken

    Output:
    :param: gdf_bboxes, GeoDataFrame containing all the squares into which the area is divided
    """

    print("Splitting polygon into squares...")
    translated_shape, deg_shift = check_utm_box(shape)

    bbox_splitter = UtmZoneSplitter([translated_shape], crs, size)
    bbox_list = np.array(bbox_splitter.get_bbox_list())
    info_list = np.array(bbox_splitter.get_info_list())
    print(info_list)

    bboxes_gps = []

    # Loop over the bounding boxes
    for one_box in tqdm(bbox_list):
        if wgs84:
            this_box = bbox_utm_to_wgs84(one_box)

        bboxes_gps.append(this_box)

    gdf_bboxes = gpd.GeoDataFrame(geometry=bboxes_gps, crs=crs)
    gdf_bboxes["geometry"] = gdf_bboxes.translate(xoff=-deg_shift)
    gdf_bboxes.to_file(file_out)
    print(f"Saved to: {file_out}")

    return gdf_bboxes


def bbox_utm_to_wgs84(bbox: BBox) -> Polygon:
    """Convert bounding box from UTM to EPSG:4326"""

    lon0 = bbox.min_x
    lat0 = bbox.min_y
    lon1 = bbox.max_x
    lat1 = bbox.max_y

    poly = Polygon([(lon0, lat0), (lon0, lat1), (lon1, lat1), (lon1, lat0)])
    gdf_tmp = gpd.GeoDataFrame(geometry=[poly], crs="epsg:32633")
    gdf_new = gdf_tmp.to_crs("epsg:4326")

    return gdf_new["geometry"].values[0]


def bbox_to_polygon(bbox=None) -> Polygon:
    """
    Convert a bounding box to a shapely polygon

    Input:
    :param: bbox, bounding box with two pairs: (lon, lat) (lon, lat)

    Output:

    :param: polybox, shapely polygon obtained from the input bbox
    """

    lon0, lat0, lon1, lat1 = tuple(bbox)

    pt0 = [lon0, lat0]
    pt1 = [lon0, lat1]
    pt2 = [lon1, lat0]
    pt3 = [lon1, lat1]

    polybox = Polygon([pt0, pt1, pt2, pt3])

    return polybox


def invert_polygon_latlon(polygon=None) -> Polygon:
    """
    Given a polygon/multipolygon, invert lat and long coordinates

    Input:
    :param: polygon, Polygon/MultiPolygon input polygon(s)

    Output:
    :param: polygon_new, list of Polygon/MultiPolygon with reversed lat/lon
    """

    # Ensure that the object we will be dealing with is a list of polygons
    if isinstance(polygon, Polygon):
        polygons = [polygon]

    elif isinstance(polygon, MultiPolygon):
        polygons = list(polygon)

    else:
        print(
            "Error. Must deal with polygons only for coordinate inversion. Exiting..."
        )

    # Convert the polygons to the new coordinate system, point by point
    polygons_tmp = []
    for poly in polygons:
        points = []
        x_utm, y_utm = poly.exterior.coords.xy

        for x, y in zip(x_utm, y_utm):
            points.append(Point(y, x))

        poly_new = Polygon(points)
        polygons_tmp.append(poly_new)

    # Convert the polygon(s) to MultiPolygon or return a simple Polygon obj
    if len(polygons_tmp) > 1:
        polygons_new = MultiPolygon(polygons_tmp)
    else:
        polygons_new = polygons_tmp[0]

    return polygons_new


def get_address_from_gps(coords=None) -> list:
    """This is a wrapper for gpd tools reverse geocode, we put it here so if at some point the reverse geocoding service needs to change we only change it here."""

    addresses = []

    # Loop over all the zones in the province and find their addresses
    for coord in coords:

        # Make sure lat and lon are encoded in the right order
        if not isinstance(coord, Point):
            coord = Point(np.array((coord[1], coord[0])))

        address = gpd.tools.reverse_geocode(coord, provider="arcgis")["address"][0]
        addresses.append(address)

    return addresses


def find_point_within_polygon(polygon=None) -> tuple:
    """
    Given a polygon, ensure that we will get a point that lies within it.
    This is basically a wrapper for the function representative_point() but we might want to use other functions, so we'd only need to change this here.

    Input:
    :param: polygon: shapely Polygon

    Output:
    :param: x,y: tuple of float values
    """

    point = polygon.representative_point()
    x, y = point.x, point.y

    if not polygon.contains(point):
        print(f"Point {x}, {y} does not lie within the polygon.")

    return x, y


def check_utm_box(shape, crs="epsg:4326") -> tuple:
    """
    When partitioning the provinces into squares, there is an error for those squares at lat > 18 or lat < 12.
    We need to rescale their position appropriately with a 3 or 6deg correction, depending on the position of the
    polygon (whether it is completely before 12 or beyond 18, or it lies on the edge).
    """

    deg_max = 18
    deg_min = 12

    lon0, lat0, lon1, lat1 = tuple(shape.bounds)
    deg_shift = 0.0

    # We can shift by the whole 6 degrees those provinces whose boundaries lie completely < 12
    if lon0 < deg_min:
        deg_shift = 6.0

    # Since only a very small part of Italy (provincia di Lecce only) is > 18, we don't need to shift it all by 6 deg
    if lon1 > deg_max:
        deg_shift = -3.0

    # If the province is right on the edge of the 12 deg long then shift it by less than 6 to avoid placing the right boundaries
    if (lon1 > deg_min) and (lon0 < deg_min):
        deg_shift = 3.0

    gshp = gpd.GeoDataFrame([deg_shift], geometry=[shape], crs=crs)
    gshp["geometry"] = gshp.translate(xoff=deg_shift)

    return gshp["geometry"].values[0], deg_shift


def unpack_address(address=None) -> str:
    """
    Unpack a dictionary containing various address features to a single string
    """

    ad_str = ""

    for key in address:
        ad_str += address[key] + " "

    return ad_str


def check_polygons_overlap(
    poly1=None, poly2=None, overlap_min=0.75, area_max=0.3
) -> bool:
    """
    Check if two polygons overlap enough to be considered the same area
    Sometimes the shapes of provinces / OMI zones / cities etc. taken from different sources might differ.
    This function helps checking whether they overlap by more than a given amount and can be then considered as the same.
    """

    # Initialize
    overlap = False

    if poly2.is_empty or poly1.is_empty:
        overlap = False

    # We need to check it both ways: if one of the two is much smaller, it might share 100% of its area with the other, which would share much less
    if poly1.is_valid and poly2.is_valid:
        intersect12 = poly1.intersection(poly2)
        intersect21 = poly2.intersection(poly1)

        if (poly1.area > 0.0) and (poly2.area > 0.0):
            overlap12 = intersect12.area / poly1.area
            overlap21 = intersect21.area / poly2.area

        # If the polygon has null area then there is no overlap and we're dealing with something wrong
        else:
            overlap12, overlap21 = 0.0, 0.0

        # print('Overlap area: ', overlap12, overlap21)

        if (overlap12 > overlap_min) and (overlap21 > overlap_min):
            overlap = True

    # If some polygons have a weird shape it is not possible to check the intersection but we might still check the area difference to determine whether it's the same polygon or not
    else:
        area_diff = abs(poly1.area - poly2.area) / poly1.area
        print("Area difference between polygons: ", area_diff)

        if area_diff < area_max:
            overlap = True

    return overlap


def find_zones_within_radius(
    point=None, radius=None, geo_data=None, crs=None
) -> gpd.GeoDataFrame:
    """
    Given a GPS point (Point shapely type) a radius (float) find all the zones that intersect the area in a GeoDataFrame (geo_data)
    """

    # Here we append all the rows
    rows_select = []

    # First convert all the data to CRS=EPSG:32634 to work with metric units instead of degrees
    epsg_metric = crs
    point_metric = (
        gpd.GeoDataFrame(crs=crs, geometry=[point]).to_crs(epsg_metric).values[0][0]
    )
    geo_data_metric = geo_data.to_crs(epsg_metric)

    # Generate a circle using the buffer method
    circle = point_metric.buffer(radius)

    # Now loop over the geodataframe and look for intersecting areas
    for i, row in geo_data_metric.iterrows():

        geo = row["geometry"]

        if circle.intersects(geo):
            rows_select.append(row)

    # Create a new geodataframe with the matching rows
    geo_data_select = gpd.GeoDataFrame(rows_select, crs=epsg_metric)

    # Convert the selected geodata to the original CRS (here they are still in the epsg_metric CRS)
    geo_data_select.to_crs(crs=crs)

    # Print some information on the selected data
    print(
        f"Found {len(geo_data_select)} OMI zones within {radius} meters from the center."
    )

    return geo_data_select


def resample_within_polygon(poly: Polygon) -> Point:
    """
    Get a (random) point within the polygon
    """

    x_min, y_min, x_max, y_max = poly.bounds
    x_rand, y_rand = -1, -1
    point_rand = Point([x_rand, y_rand])

    for i in range(0, 10000):
        x_rand = random.uniform(x_min, x_max)
        y_rand = random.uniform(y_min, y_max)
        point_rand = Point([x_rand, y_rand])

        if poly.contains(point_rand):
            print("Found randomly resampled point: ", x_rand, y_rand)
            break

    return point_rand


def bbox_from_address(address: str, radius: float) -> tuple:
    """
    Take a string with an address and get its gps coordinates
    """

    coords = gpd.tools.geocode(address)
    poi = coords["geometry"]
    region = poi.buffer(radius)[0]
    bbox = region.bounds
    region_ee = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])
    poi_ee = ee.Geometry.Point(poi[0].x, poi[0].y)

    return region_ee, poi_ee


def generate_squares_for_province(prov: str, side: int) -> gpd.GeoDataFrame:
    """
    Generate a set of squares for a given province name and square size in meters.

    :param prov: Province code (e.g., 'CT').
    :param side: Size of each square in meters.
    :return: GeoDataFrame containing the squares.
    """
    geo_file = "/home/edoardo/DATA/confini_amministrativi/Limiti01012021_g/ProvCM01012021_g/ProvCM01012021_g_WGS84.shp"
    file_out = f"data/{prov}_squares_{side}.shp"

    if not os.path.exists(file_out):
        gdf = gpd.read_file(geo_file)
        gdf = gdf.to_crs("epsg:4326")
        gdf_sel = gdf[gdf["SIGLA"] == prov]

        this_shape = gdf_sel["geometry"].values[0]
        gdf_boxes = split_into_squares(shape=this_shape, file_out=file_out, size=side)
        gdf_boxes.to_file(file_out)
    else:
        gdf_boxes = gpd.read_file(file_out)

    return gdf_boxes


def test_generation(size=7200):
    """Main, used for testing other function of this file"""
    regione = "Sicilia"
    provinces = ["CT", "PA", "RG", "AG", "SR", "ME", "TP", "EN", "CL"]
    gdfs = []

    with st.spinner(f"Partitioning the man in {size} km side squares..."):
        for select in tqdm(provinces):
            gdf_boxes = generate_squares_for_province(prov=select, side=size)
            gdfs.append(gdf_boxes)

    file_reg_out = f"data/{regione}_squares_{size}.shp"
    full_gpd = pd.concat(gdfs)
    full_gpd.to_file(file_reg_out)
    print(
        f"File for {regione} saved to {file_reg_out}. Total number of squares: {len(full_gpd)}"
    )


if __name__ == "__main__":
    """MAIN WRAPPER"""

    generate_squares_for_province(prov="CT", side=7200)
