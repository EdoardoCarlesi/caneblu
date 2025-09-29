from datetime import datetime, timedelta, timezone
import json
import os
import threading
import time

from fontTools.mtiLib import bucketizeRules
from minio import Minio
from minio.error import S3Error


# Variables and settings
provs = ["CT", "PA", "RG", "AG", "SR", "ME", "TP", "EN", "CL"]
FIRE_RISK_SCHEDULE_HOURS = 8 # How often should we run the FRI
HOURS_CACHE = 12 # Meteo API calls (make a new one if the data is older than this)

# Local paths
TMP_PATH = "tmp/" # For caching data mainly
LOG_PATH = "logs/" # For the logging files
DATA_PATH = 'data/' # Data necessary to run the application
MODEL_PATH = "models/RF_thr0.1_estimators100_depth10_samples20_dates_2_acc0.85.pkl"  # Path to the pre-trained ML model
FRI_OUTPUT_PATH = 'output/FRI/'  # Fire Risk Index output folder
ML_OUTPUT_PATH = "output/ML/"  # Fire Risk Index ML output folder
MSI_OUTPUT_PATH = "output/MSI/"  # Multi spectral images are dumped here
PROVINCE_FILE_PATH = f'{DATA_PATH}ConfiniProvince/ProvCM01012021_g_WGS84.shp'

# Bucket paths on MinIO
MINIO_FRI_PATH = "FireRisk/FRI/"    # MinIO Path for the FRI
MINIO_ML_PATH = "FireRisk/ML/"  # MinIO Path for the ML FRI

##########################################################
# DEFINE ALL FILENAMES HERE AS FUNCTIONS FOR CONSISTENCY #
##########################################################


def shp_square_filename(prov, size=3600):
    path = f"{DATA_PATH}{prov}_squares_{size}.shp"
    return path


def npy_filename(prov, index, date, res=60):
    path = f"{MSI_OUTPUT_PATH}{prov}_{index:04}_{date}_{res}.npy"
    return path

    
def npy_mosaic_filename(prov, index, date):
    path = f"{MSI_OUTPUT_PATH}{prov}_{index:04}_{date}_{res}_mosaic.npy"
    return path


def geotiff_square_ml_filename(prov, index, res=60, size=3600):
    path = f"{ML_OUTPUT_PATH}{prov}_{index:04}_res{resolution}m_risk_map_ml_{size}.tiff"
    return path


def geotiff_ml_filename(prov, size=3600, res=60):
    path = f'{ML_OUTPUT_PATH}MOSAIC_{prov}_res{res}_full_risk_map_ml{size}.tiff'
    return path


def geotiff_clip_ml_filename(prov, size=3600, res=60):
    path = f'{ML_OUTPUT_PATH}MOSAIC_{prov}_res{res}_full_risk_map_ml{size}_clip.tiff'
    return path


def png_clip_ml_filename(prov, size=3600, res=60):
    path = f'{ML_OUTPUT_PATH}MOSAIC_{prov}_res{res}_full_risk_map_ml{size}_clip.png'
    return path


def png_ml_filename(prov, size=3600, res=60):
    path = f'{ML_OUTPUT_PATH}MOSAIC_{prov}_res{res}_full_risk_map_ml{size}.png'
    return path


def geojson_fri_filename(prov, size=3600):
    path = f"{FRI_OUTPUT_PATH}{province}_risk_map_{size}.json"
    return path


def geojson_fri_minio(prov, size=3600):
    path = f"{MINIO_FRI_PATH}{prov}_risk_map_{size}.json"
    return path 


def geotiff_ml_minio(prov, size=3600, res=60):
    path = f'{MINIO_ML_PATH}MOSAIC_{prov}_res{res}_full_risk_map_ml{size}.tiff'
    return path


def geotiff_clip_ml_minio(prov, size=3600, res=60):
    path = f'{MINIO_ML_PATH}MOSAIC_{prov}_res{res}_full_risk_map_ml{size}_clip.tiff'
    return path


def png_clip_ml_minio(prov, size=3600, res=60):
    path = f'{MINIO_ML_PATH}MOSAIC_{prov}_res{res}_full_risk_map_ml{size}_clip.png'
    return path




######################################################################
################## OTHER FUNCTIONS ###################################
######################################################################


def get_minio_client(json_file='json/.minio.json', logger=None) -> Minio:
    """
    Create a Minio client to interact with an S3-compatible MinIO bucket.

    :param endpoint: MinIO server URL, e.g., 'play.min.io:9000'
    :param access_key: Your MinIO access key
    :param secret_key: Your MinIO secret key
    :param secure: Use HTTPS if True, HTTP if False
    :return: Minio client instance
    """

    try:
        with open(json_file, 'r') as f:
            config = json.load(f)

        endpoint = config['url']
        access_key = config['accessKey']
        secret_key = config['secretKey']

        if logger is None:
            print(f"Found configuration file: {json_file} for endpoint: {endpoint}")
        else:
            logger.info(f"Found configuration file: {json_file} for endpoint: {endpoint}")

    except FileNotFoundError:
        # Fallback to environment variables
        if logger is None:
            print(f"Config file '{json_file}' not found. Trying to load from environment variables.")
        else: 
            logger.info(f"Config file '{json_file}' not found. Trying to load from environment variables.")
        
        endpoint = os.environ.get('MINIO_URL')
        access_key = os.environ.get('MINIO_ACCESS_KEY')
        secret_key = os.environ.get('MINIO_SECRET_KEY')
        
        if not all([endpoint, access_key, secret_key]):
            raise RuntimeError(
                "MinIO credentials not found in environment variables. "
                "Set MINIO_URL, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY."
            )
        else:
            if logger is None:
                print("MINIO_URL, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY found in the environment variables")
            else:
                logger.info("MINIO_URL, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY found in the environment variables")
    try:
        client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
            )

    except Exception as e:
        if logger is None:
            print(f"Client error: {e}")
        else:
            logger.info(f"Client error: {e}")

    return client


def remove_duplicates_from_list(data):
    """
    Removes duplicate dictionaries from a list while preserving the order.

    :param data: List of dictionaries to deduplicate.
    :return: A deduplicated list of dictionaries.
    """
    deduplicated_list = []
    seen = set()

    for item in data:
        item_id = item['id'].split('_')[-1]

        if item_id not in seen:
            deduplicated_list.append(item)
            seen.add(item_id)

    return deduplicated_list


def upload_fri_to_minio(prov, side=3600):

    bucket_name = 'sefore'
    file_loc = geojson_fri_filename(prov)
    file_minio = geojson_fri_minio(prov)
    client = get_minio_client()
    client.fput_object(bucket_name, file_minio, file_loc)

    return file_minio


def get_fri_from_minio(prov, side=3600):

    bucket_name = 'sefore'
    file_minio = geojson_fri_minio(prov)
    client = get_minio_client()
    response = client.get_object(bucket_name, file_minio)
    file_fri = response.read()
    file_fri = json.loads(file_fri)
    
    return file_fri


def upload_ml_to_minio(prov, side=3600, res=60):

    bucket_name = 'sefore'
    
    # Local filenames
    tiff = geotiff_ml_filename(prov)
    tiff_clip = geotiff_clip_ml_filename(prov)
    png = png_ml_filename(prov)
    png_clip = png_clip_ml_filename(prov)
    
    # Remote filenames
    tiff_minio = geotiff_ml_minio(prov)
    tiff_clip_minio = geotiff_clip_ml_minio(prov)
    png_minio = png_ml_minio(prov)
    png_clip_minio = png_clip_ml_minio(prov)

    client = get_minio_client()
    client.fput_object(bucket_name, tiff_minio, tiff)
    client.fput_object(bucket_name, tiff_clip_minio, tiff_clip)
    client.fput_object(bucket_name, png_minio, png)
    client.fput_object(bucket_name, png_clip_minio, png_clip)


def get_ml_from_minio(prov, side=3600):

    bucket_name = 'sefore'

    # Only retrieve the CLIPPED version of the files
    tiff_minio = geotiff_clip_ml_minio(prov)
    png_minio = png_clip_ml_minio(prov)

    # Local files
    tiff = geotiff_clip_ml_filename(prov)
    png = png_clip_ml_filename(prov)

    client = get_minio_client()
    client.fget_object(bucket_name, tiff_minio, tiff)
    client.fget_object(bucket_name, png_minio, png)
    


def is_file_older_than(file_path : str, hours : int, size=3600) -> bool:
    """
    Checks if the file at the given path is older than the specified number of hours.

    :param file_path: Path to the file.
    :param hours: The number of hours to check against.
    :return: True if the file is older than the given number of hours, False otherwise.
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Get the file's last modification time in seconds since epoch
    file_mod_time = os.path.getmtime(file_path)

    # Get the current time in seconds since epoch
    current_time = time.time()

    # Convert hours to seconds
    time_difference_limit = hours * 3600

    # Check if the file is older
    return (current_time - file_mod_time) > time_difference_limit


def is_minio_file_older_than(object_name : str, hours=HOURS_CACHE, logger=None, size=3600) -> bool:
    """
    Checks if a file on MinIO is older than the specified number of hours.

    :param bucket_name: Name of the MinIO bucket.
    :param object_name: Name of the file/object in the bucket.
    :param hours: The number of hours to check against.
    :return: True if the file is older than the specified number of hours, False otherwise.
    """

    client = get_minio_client()
    bucket_name = 'sefore'

    try:
        # Get the object's metadata to retrieve last modification time
        obj_metadata = client.stat_object(bucket_name, object_name)
    except S3Error as e:
        if logger is None:
            print(f"Error retrieving metadata for {object_name} in bucket {bucket_name}: {e}")
        else:
            logger.info(f"Error retrieving metadata for {object_name} in bucket {bucket_name}: {e}")
        return False

    # Extract the last_modified time and convert to UTC
    last_modified = obj_metadata.last_modified.replace(tzinfo=timezone.utc)
    current_time = datetime.now(timezone.utc)

    # Calculate the time difference limit
    time_difference_limit = timedelta(hours=hours)

    # Check if the file is older than the specified time
    return (current_time - last_modified) > time_difference_limit


def minio_file_last_updated(object_name : str, logger=None) -> bool:
    """
    Checks if a file on MinIO is older than the specified number of hours.

    :param bucket_name: Name of the MinIO bucket.
    :param object_name: Name of the file/object in the bucket.
    :param hours: The number of hours to check against.
    :return: True if the file is older than the specified number of hours, False otherwise.
    """

    client = get_minio_client()
    bucket_name = 'sefore'

    try:
        # Get the object's metadata to retrieve last modification time
        obj_metadata = client.stat_object(bucket_name, object_name)
    except S3Error as e:
        if logger is None:
            print(f"Error retrieving metadata for {object_name} in bucket {bucket_name}: {e}")
        else:
            logger.info(f"Error retrieving metadata for {object_name} in bucket {bucket_name}: {e}")
        return False

    # Extract the last_modified time and convert to UTC
    last_modified = obj_metadata.last_modified.replace(tzinfo=timezone.utc)

    # Check if the file is older than the specified time
    return last_modified


def read_json_file(file_path, logger=None):
    """
    Reads and parses a JSON file.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON data as a Python dictionary or list.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)  # Parse the JSON file
            return data
    except FileNotFoundError:
        if logger is None:
            print(f"Error: File not found at {file_path}!!")
        else:
            logger.info(f"Error: File not found at {file_path}!!")
        return None
    except json.JSONDecodeError as e:
        if logger is None:
            print(f"Error: Failed to decode JSON. {e}")
        else:
            logger.info(f"Error: Failed to decode JSON. {e}")

        return None


def dump_geojson(data, file_path, logger=None):
    """
    Dumps a dictionary into a GeoJSON file.

    :param data: A Python dictionary representing GeoJSON data.
    :param file_path: Path where the GeoJSON file will be saved.
    """
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)  # Save JSON with indentation
            # print(f"GeoJSON data has been written to {file_path}")
    except Exception as e:
        if logger is None:
            print(f"Error: Could not write GeoJSON data. {e}")
        else:
            logger.info(f"Error: Could not write GeoJSON data. {e}")


def modis_to_celsius(t):
    """
    Simple conversion
    """

    return t * 0.002 - 273.15


def fix_time(times):
    """
    Format time to the right string
    """
    new_times = []
    for t in times:
        new_times.append(t[0:10])

    return new_times


def time_to_utc(date_str: str, hour=0, minute=0) -> tuple:
    """
    AgroMonitoring API get as an input the UTC time, so we need to convert input strings properly
    """

    time_tmp = time.strptime(date_str, "%Y-%m-%d")
    year, month, day = time_tmp.tm_year, time_tmp.tm_mon, time_tmp.tm_mday
    date_dt = datetime(year, month, day, hour, minute)
    date_utc = int(time.mktime(date_dt.timetuple()))

    return date_utc


def utc_to_time(date_utc: int) -> tuple:
    """
    Make UTC time readable
    """

    return datetime.fromtimestamp(date_utc)


if __name__ == "__main__":
    """
    Execute stuff from here
    """

    for prov in provs:
        #upload_ml_to_minio(prov)   
        get_ml_from_minio(prov)
