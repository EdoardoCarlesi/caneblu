import glob
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import mapping
from tifffile import imread, imwrite
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.mask import mask

# Local import
from . import utils
from . import composite_indices as ci


def gray_to_rgb(image_data, color_map=None):

    # Normalize the image data for PNG conversion (scale to 0-255)
    image_data = image_data.astype(float)  # Convert to float for scaling
    image = ((image_data - image_data.min()) /
        (image_data.max() - image_data.min()) * 255).astype(np.uint8)

    # Define 6-level color map if not provided
    if color_map is None:
        color_map = {
            0: (255, 255, 255),  # White
            1: (215, 252, 0),  # Bright Green
            2: (0, 255, 0),  # Green
            3: (255, 255, 0),  # Yellow
            4: (255, 165, 0),  # Orange
            5: (255, 0, 0)  # Red
        }

    # Define 6 equally spaced bins
    bins = np.linspace(image.min(), image.max(), 7)  # 6 bins => 7 edges

    # Digitize the image to assign bin indices (0 to 5)
    binned = np.digitize(image, bins) - 1  # Bin indices range from 0 to 5

    # Create a blank RGB image
    rgb_image = np.zeros((*image.shape, 3), dtype=np.uint8)

    # Map bin indices to RGB colors
    for bin_index, color in color_map.items():
        rgb_image[binned == bin_index] = color

    return rgb_image


def tiff_to_png(tiff_path, output_png_path=None, color_map=None):
    """
    Convert a GeoTIFF file to PNG format.

    :param tiff_path: Path to the input GeoTIFF file.
    :param output_png_path: Path to save the output PNG file.
    """
    # Open the GeoTIFF file using rasterio
    with rasterio.open(tiff_path) as src:
        # Read the first band (assuming it's a grayscale image)
        image_data = src.read()

        if src.count == 1:
            rgb_image = gray_to_rgb(image_data[0])

        if src.count == 3:
            rgb_image = np.dstack([image_data[0], image_data[1], image_data[2]])

        image = Image.fromarray(rgb_image)

        if not output_png_path == None:
            image.save(output_png_path)
            threshold = 250
            img = Image.open(output_png_path).convert("RGBA")
            datas = img.getdata()
            new_data = []
            
            for r, g, b, a in datas:
                if r >= threshold and g >= threshold and b >= threshold:
                    new_data.append((r, g, b, 0))
                else:
                    new_data.append((r, g, b, a))

            img.putdata(new_data)
            img.save(output_png_path, "PNG")

    return image        


def generate_bounding_box(latitude, longitude, size_meters):
    """
    Generates a bounding box around a given GPS coordinate.

    Args:
        latitude: Latitude of the center point.
        longitude: Longitude of the center point.
        size_meters: Size of the bounding box in meters (square box).

    Returns:
        A GeoDataFrame with the bounding box polygon.
    """

    # Earth's radius in meters
    earth_radius = 6378137

    # Calculate latitude and longitude offsets
    lat_offset = (size_meters / 1.35) / earth_radius * (180 / 3.14159)
    lon_offset = (
        (size_meters / 2)
        / (earth_radius * math.cos(math.pi * latitude / 180))
        * (180 / 3.14159)
    )

    # Define bounding box coordinates
    bbox = [
        (latitude + lat_offset, longitude - lon_offset),  # Upper left
        (latitude + lat_offset, longitude + lon_offset),  # Upper right
        (latitude - lat_offset, longitude + lon_offset),  # Lower right
        (latitude - lat_offset, longitude - lon_offset),  # Lower left
        (latitude + lat_offset, longitude - lon_offset),  # Close polygon
    ]

    # Create a polygon from the bounding box coordinates
    polygon = Polygon(bbox)

    # Create a GeoDataFrame from the polygon
    return polygon


def create_rgb_imgs(base_path, sub_paths, sub_dir="SENTINEL2_NO_CLOUDS", sub_dir_target="RGB_DATA", sub_sub_dir_target="RGB", rgbs=[], pxmax=11056):

    for sp in sub_paths:
        images_orig = glob.glob(f"{base_path}/{sp}/{sub_dir}/*.tiff")
        dir0 = base_path + "/" + sp + "/" + sub_dir_target
        dir1 = base_path + "/" + sp + "/" + sub_dir_target + "/" + sub_sub_dir_target

        if not os.path.exists(dir0):
            os.mkdir(dir0)
        if not os.path.exists(dir1):
            os.mkdir(dir1)

        for img in images_orig:
            io = imread(img)[:, :, 0:3]
            io = np.array((io / pxmax) * 255, dtype=int)
            rgbs.append(io)
            f_name = img.split("/")[-1]
            f_name = f"{dir1}/{f_name}"
            imwrite(f_name, io)


def bin_values(vector, n_bins=5):
    """Bins a vector of real values between 0 and 1 into 5 value brackets.

    Args:
      vector: A 1D numpy array of real values between 0 and 1.

    Returns:
      A 1D numpy array of the same size as vector, with each element replaced by its
      corresponding bin value
    """

    # Define bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Assign each element of vector to a bin using `digitize`
    binned_vector = np.digitize(vector, bin_edges, right=False) - 1

    # Ensure values stay within the valid range of bins
    binned_vector = np.clip(binned_vector, 0, n_bins - 1) / (n_bins - 1)

    return binned_vector


def extract_date(fname):

    fname = fname.split("/")[-1]
    dates = fname.split("_")

    if "SENTINEL2" in dates:
        year, month, day = dates[4], dates[3], dates[2]
    elif "NDVI" in dates:
        year, month, day = dates[3].split(".")[0], dates[2], dates[1]
    else:
        year, month, day = dates[2], dates[1], dates[0]

    return f"{year}-{month}-{day}"


def get_files(path, sub_dirs, sub_sub_dirs, all_indices):

    indices = dict()

    for index in all_indices:

        indices[index] = dict()

        if index == "NDVI":
            index_path = f"{sub_dirs[0]}/{sub_sub_dirs[0]}"
        elif index == "NBR":
            index_path = f"{sub_dirs[1]}/{sub_sub_dirs[2]}"
        elif index == "NDMI":
            index_path = f"{sub_dirs[2]}/{sub_sub_dirs[4]}"
        elif index == "RGB":
            index_path = f"{sub_dirs[3]}/{sub_sub_dirs[6]}"
        elif index == "NDVI_5YR":
            index_path = f"{sub_dirs[0]}/{sub_sub_dirs[-1]}"
        else:
            index_path = "-"
            print("Error: Index Path {index_path} does not exist!")

        full_path = f"{path}/{index_path}/"
        files = sorted(glob.glob(f"{full_path}*tiff"))

        for f in files:
            d = extract_date(f)
            indices[index][d] = f

    return indices


def get_fire_date(path):
    for p in path.split("/"):
        if "FIRMS" in p:
            p = p.split("_")
            return p[3], p


def str_to_int(date):

    if isinstance(date, str):
        return int(date[0:4]), int(date[5:7]), int(date[8:10])
    else:
        date = date[1]
        date = date[4].split("-")
        return int(date[0]), int(date[1]), int(date[2])


def int_to_str(date):
    return datetime(*date).strftime("%Y-%m-%d")


def sort_before_after(dates, fire_date):

    diffs = []
    diff2date = dict()
    fd = str_to_int(fire_date)
    fd = datetime(fd[0], fd[1], fd[2])
    dates_before = []
    dates_after = []

    for d in dates:
        d = str_to_int(d)
        diff = (datetime(d[0], d[1], d[2]) - fd).days
        diffs.append(diff)
        diff2date[diff] = d

    dsort = np.argsort(abs(np.array(diffs)))

    for id in dsort:
        diffid = list(diff2date.keys())[id]
        d2d = diff2date[diffid]

        if diffid < 0:
            dates_before.append(d2d)
        else:
            dates_after.append(d2d)

    return dates_before, dates_after


def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return (band - band_min) / ((band_max - band_min))


def brighten(band):
    alpha = 0.13
    beta = 0
    return np.clip(alpha * band + beta, 0, 255)


def rgb_to_gray(img):
    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]


def gen_diff(index):
    list_diff = []

    for il in range(1, len(index)):
        diff = index[il].astype(float) - index[il - 1].astype(float)
        list_diff.append(diff)

    return list_diff


def check_pred(y_pred, y_test):
    true = len(y_pred) - np.sum(abs(y_pred - y_test))
    false = len(y_pred) - true

    return true, false


def load_indices_in_area(num_file, base_path, sub_paths, all_indices, sub_dirs, sub_sub_dirs, n_dates=3):

    path = f"{base_path}/{sub_paths[num_file]}"
    files = get_files(path, sub_dirs, sub_sub_dirs, all_indices)
    fire_date = get_fire_date(path)
    before, after = sort_before_after(files["NDVI"].keys(), fire_date)
    sub_paths[num_file]

    dates = []
    dates.append(int_to_str(after[0]))
    ndvis, nbrs, ndmis, rgbs = [], [], [], []

    for d in range(0, n_dates):
        dates.append(int_to_str(before[d]))

    for d in dates:
        ndvis.append(imread(files["NDVI"][d]))
        rgbs.append(imread(files["RGB"][d]))
        ndmis.append(imread(files["NDMI"][d]))
        nbrs.append(imread(files["NBR"][d]))

    ndvis_5yr = []

    for d in sorted(files["NDVI_5YR"].keys()):
        ndvis_5yr.append(imread(files["NDVI_5YR"][d]))

    return ndvis, nbrs, ndmis, rgbs, ndvis_5yr


def burned_mask(dnbr, threshold=0.75):
    burned_pts = dnbr > threshold
    normal_pts = dnbr <= threshold
    dnbr_mask = dnbr.copy()
    dnbr_mask[burned_pts] = 1.0
    dnbr_mask[normal_pts] = 0.0

    return dnbr_mask


def show_images_side_by_side(images, titles=None, cmap="viridis"):
    """Displays an arbitrary number of images side-by-side.

    Args:
      images: A list of images to display. Each image can be a NumPy array
        or a PIL Image object.
      titles: An optional list of titles for the images.
      cmap: The colormap to use for grayscale images. Defaults to 'viridis'.
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    for i, image in enumerate(images):
        ax = axes[i]
        ax.imshow(image, cmap=cmap)
        ax.axis("off")  # Turn off axis ticks and labels

        if titles:
            ax.set_title(titles[i])

    plt.tight_layout()  # Adjust subplot parameters for a tight layout
    plt.show()  # Display the figure
    
    return plt


def format_for_fire_risk(data, n_dates=3) -> tuple:
    i_blue, i_green, i_red, i_nir, i_swir, i_swir2 = 0, 1, 2, 3, 4, 5

    ndvis, ndmis, rgbs_gray, nbrs, rgbs = [], [], [], [], []

    for i in range(0, n_dates):
        rgb = enhance_rgb_bands(data[0][i][:, :, [2, 1, 0]], factor=2.0)
        rgb_gray = rgb_to_gray(data[0][i][:, :, 0:3])
        ndvi = ci.NDVI(data[0][i], i_nir, i_red)
        ndmi = ci.NDMI(data[0][i], i_swir, i_nir)
        nbr = ci.NBR(data[0][i], i_swir2, i_nir)

        rgbs.append(rgb)
        rgbs_gray.append(rgb_gray)
        ndvis.append(ndvi)
        ndmis.append(ndmi)
        nbrs.append(nbr)

    return (ndvis, ndmis, rgbs_gray, nbrs, rgbs)


def enhance_rgb_bands(image, factor=2.5):
    """Enhances an RGB image band-by-band with fixed numerical factors.

    Args:
      image: A NumPy array representing the RGB image (shape: height, width, 3).
      factors: A list or tuple of three numerical factors, one for each band (R, G, B).

    Returns:
      A NumPy array representing the enhanced RGB image.
    """

    # Enhance each band
    enhanced_image = image.copy()
    for i in range(0, 3):
        enhanced_image[:, :, i] = enhanced_image[:, :, i] * factor

    return enhanced_image


def rgbs_to_grays(rgb_images):
    """Converts a list of RGB images to grayscale.

    Args:
      rgb_images: A list of RGB images, where each image is a NumPy array.

    Returns:
      A list of grayscale images, where each image is a NumPy array.
    """

    gray_images = []
    for rgb_image in rgb_images:
        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(rgb_image.astype("uint8"), "RGB")

        # Convert to grayscale
        gray_image = pil_image.convert("L")

        # Convert back to NumPy array
        gray_images.append(np.array(gray_image))

    return gray_images


def compare_arrays(array1, array2):
    """Compares two arrays and returns the number and percentage of different entries.

    Args:
      array1: The first array.
      array2: The second array.

    Returns:
      A tuple containing:
        - The total number of different entries.
        - The percentage of different entries.
    """

    # Compare the arrays element-wise
    comparison_array = np.where(array1 == array2, 0, 1)

    # Calculate the total number of different entries
    total_different = np.sum(comparison_array)

    # Calculate the percentage of different entries
    percentage_different = (total_different / len(array1)) * 100

    return total_different, percentage_different


def reshape_to_2d(flat_array, shape):
    """Reshapes a 1-dimensional flattened array into a 2-dimensional matrix.

    Args:
      flat_array: The 1-dimensional flattened array.
      shape: A tuple specifying the desired shape of the 2-dimensional matrix (rows, columns).

    Returns:
      The reshaped 2-dimensional matrix.
    """
    try:
        reshaped_array = np.reshape(flat_array, shape)
        return reshaped_array
    except ValueError:
        raise ValueError(
            "Cannot reshape array with given shape. Ensure the number of elements matches."
        )


def create_features(ndvis, ndmis, rgbs_gray, nbrs, dnbr_mask=None, n_dates=3, norm=True):

    fire_df = pd.DataFrame()

    for id in range(0, n_dates):
        ndvi_line = np.ndarray.flatten(ndvis[id])
        ndmi_line = np.ndarray.flatten(ndmis[id])
        rgb_line = np.ndarray.flatten(rgbs_gray[id])
        nbr_line = np.ndarray.flatten(nbrs[id])

        if norm:
            ndvi_line = normalize(ndvi_line)
            ndmi_line = normalize(ndmi_line)
            rgb_line = normalize(rgb_line)
            nbr_line = normalize(nbr_line)

        fire_df[f"NDVI_{id}"] = ndvi_line
        fire_df[f"NDMI_{id}"] = ndmi_line
        fire_df[f"RGB_{id}"] = rgb_line
        fire_df[f"NBR_{id}"] = nbr_line

    if not (dnbr_mask is None):
        burned_feats = np.ndarray.flatten(dnbr_mask)
        fire_df["BURNED"] = burned_feats

    return fire_df


def save_numpy_to_geotiff(array, output_file, roi, crs="EPSG:4326"):
    """
    Save a NumPy array to a georeferenced TIFF file.

    :param array: 2D NumPy array representing the data to save.
    :param output_file: Path to save the GeoTIFF file.
    :param lon_min: Longitude of the top-left corner of the image.
    :param lat_max: Latitude of the top-left corner of the image.
    :param pixel_size: Size of each pixel (spatial resolution).
    :param crs: Coordinate Reference System (default is EPSG:4326).
    """

    pixel_x, pixel_y, dim = array.shape[0], array.shape[1], array.shape[2]
    min_x, min_y, max_x, max_y = roi.bounds

    # Create an affine transform for georeferencing
    transform = from_bounds(min_x, min_y, max_x, max_y, pixel_x, pixel_y)
    
    # Define metadata for the GeoTIFF
    meta = {
        "driver": "GTiff",
        "dtype": array.dtype,
        "nodata": None,  # Use None or a specific no-data value
        "width": array.shape[1],  # Number of columns (width)
        "height": array.shape[0],  # Number of rows (height)
        "count": dim,  # Single-band image
        "crs": crs,  # Coordinate Reference System
        "transform": transform,  # Affine transformation
    }

    # Write the array to a GeoTIFF file
    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(array.transpose(2, 0, 1))  # Write the first (and only) band

    print(f"Array has been saved to GeoTIFF file: {output_file}")

    return output_file


def append_random_subset(source_df, target_df, fraction=0.2):
    """
    Creates a random subset of rows from source_df and appends it to target_df.

    Args:
      source_df: The source dataframe.
      target_df_name: The name of the target dataframe.
      fraction: The fraction of rows to include in the subset (default: 0.2).

    Returns:
      The updated target dataframe.

    # Check if target_df exists and create if not
    if target_df_name in locals() or target_df_name in globals():
      target_df = globals()[target_df_name]  # Or locals()[target_df_name]
      print(f"Dataframe '{target_df_name}' already exists.")
    else:
      target_df = pd.DataFrame()  # Create an empty dataframe if it doesn't exist
      print(f"Dataframe '{target_df_name}' created.")
    """

    # Create a random subset
    num_rows = int(len(source_df) * fraction)
    subset_df = source_df.sample(n=num_rows)

    # Append the subset to target_df
    target_df = pd.concat([target_df, subset_df], ignore_index=True)

    return target_df


def merge_georeferenced_tiffs(tiff_file_list, output_file):
    """
    Merges a list of georeferenced TIFF files into one GeoTIFF.

    :param tiff_file_list: List of file paths to georeferenced TIFF files.
    :param output_file: Path to save the resulting merged GeoTIFF.
    """
    # Check if the file list is not empty
    if not tiff_file_list:
        raise ValueError("The list of TIFF files is empty.")

    # Open the individual TIFF files as datasets
    src_files_to_merge = [rasterio.open(tiff_file) for tiff_file in tiff_file_list]

    # Merge the TIFF files
    mosaic, out_transform = merge(src_files_to_merge)

    # Copy metadata from one of the input files
    out_meta = src_files_to_merge[0].meta.copy()

    # Update metadata for the resulting merged file
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "count": mosaic.shape[0],  # Number of bands
    })

    # Save the merged GeoTIFF file
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close the input files
    for src in src_files_to_merge:
        src.close()

    print(f"Merged GeoTIFF saved at: {output_file}")


def replace_black_with_white(arr):
    """
    Replace black pixels (0) with white (255) in an RGBA or RGB image array.
    """
    arr = arr.copy()
    if arr.shape[0] == 3:  # RGB
        mask = np.all(arr == 0, axis=0)
        arr[:, mask] = 255
    elif arr.shape[0] == 4:  # RGBA
        mask = np.all(arr[:3] == 0, axis=0) & (arr[3] == 0)  # transparent/black
        arr[:3, mask] = 255
        arr[3, mask] = 255  # make opaque
    return arr


def crop_tiff_with_polygon(input_path, output_path, polygon, polygon_crs='EPSG:4326'):

    with rasterio.open(input_path) as src:
        # Ensure polygon is in GeoJSON-like dict format
        geojson_geom = [mapping(polygon)]
        
        raster_crs = src.crs  # CRS of the GeoTIFF

        # Reproject polygon if CRS differs
        if raster_crs and str(raster_crs) != polygon_crs:
            project = pyproj.Transformer.from_crs(polygon_crs, raster_crs, always_xy=True).transform
            polygon = transform(project, polygon)

        # Mask (crop) the raster
        try:
            out_image, out_transform = mask(src, geojson_geom, crop=True)
            out_image = replace_black_with_white(out_image)

            # Copy metadata and update
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "bounds": polygon.bounds,
                "transform": out_transform
            })

            if out_image.shape[1] > 120: # check if it is larger than at least 2 squares per side
                #   Save cropped raster
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
             
                return True
            else:
                return False

        except:
            return False


def crop_province(prov, size=3600, res=60):
 
    file_provs = utils.PROVINCE_FILE_PATH
    gdf = gpd.read_file(file_provs)
    gdf = gdf.to_crs('EPSG:4326')

    # FIXME: this only saves the mainland information and neglects the small islands which should de saved as separate files
    file_tiff = utils.geotiff_ml_filename(prov) 
    file_png = utils.png_ml_filename(prov)
    file_tiff_clip = utils.geotiff_clip_ml_filename(prov)
    file_png_clip = utils.png_clip_ml_filename(prov)
    
    gdf_pv = gdf[gdf['SIGLA'] == prov]
    crop = gdf_pv['geometry'].iloc[0]

    if isinstance(crop, Polygon):
        has_edges = crop_tiff_with_polygon(file_tiff, file_tiff_clip, crop)
        
        if has_edges:
            edges = crop.bounds

    elif isinstance(crop, MultiPolygon):
        for i, p in enumerate(crop.geoms):
            has_edges = crop_tiff_with_polygon(file_tiff, file_tiff_clip, p)
                
            if has_edges:
                edges = p.bounds

    tiff_to_png(file_tiff_clip, file_png_clip)
    print(prov, edges)
    return edges 


if __name__ == "__main__":
    """ Wrap """

    for prov in utils.provs:
        crop_province(prov)


