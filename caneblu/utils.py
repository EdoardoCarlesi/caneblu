from datetime import datetime, timedelta, timezone
import json
import os
import threading
import time
from fontTools.mtiLib import bucketizeRules


##########################################################
# DEFINE ALL FILENAMES HERE AS FUNCTIONS FOR CONSISTENCY #
##########################################################


def npy_filename(base_path, index, date, res=60):
    path = f"{base_path}_{index:04}_{date}_{res}.npy"
    return path

    
def npy_mosaic_filename(base_path, index, date, res=60):
    path = f"{base_path}_{index:04}_{date}_{res}_mosaic.npy"
    return path


######################################################################
################## OTHER FUNCTIONS ###################################
######################################################################


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
