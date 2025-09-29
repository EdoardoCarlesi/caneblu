import numpy as np

# Prevent division by zero
eps = 0.0001


def NDVI(img: np.array, i_red: int, i_nir: int) -> np.array:
    """Normalized Difference Vegetation Index"""

    red = img[:, :, i_red].astype(float)
    nir = img[:, :, i_nir].astype(float)
    ndvi = (nir - red) / (nir + red + eps)

    return ndvi


def NDWI(img: np.array, i_green: int, i_nir: int) -> np.array:
    """Normalized Difference Water Index"""

    green = img[:, :, i_green].astype(float)
    nir = img[:, :, i_nir].astype(float)
    ndwi = (green - nir) / (nir + green + eps)

    return ndwi


# TODO
def MNDWI(img: np.array, i_green: int, i_swir: int) -> np.array:
    """MODIFIED Normalized Difference Water Index"""

    green = img[:, :, i_green].astype(float)
    swir = img[:, :, i_swir].astype(float)
    ndwi = (green - swir) / (swir + green + eps)

    return ndwi


def NBR(img: np.array, i_swir: int, i_nir: int) -> np.array:
    """Normalized Burned Ratio"""

    swir = img[:, :, i_swir].astype(float)
    nir = img[:, :, i_nir].astype(float)
    nbr = (nir - swir) / (nir + swir + eps)

    return nbr


def NDMI(img: np.array, i_swir: int, i_nir: int) -> np.array:
    """Normalized Difference Moisture Index"""

    swir = img[:, :, i_swir].astype(float)
    nir = img[:, :, i_nir].astype(float)
    ndmi = (nir - swir) / (nir + swir + eps)

    return ndmi


def NDCI(img: np.array, i_vnir: int, i_nir: int) -> np.array:
    """Normalized Difference Chlorophyll Index
    NDCI is an index that aims to predict the plant chlorophyll content.
    S2: It is calculated using the red spectral band B04 with the red edge spectral band B05.
    """

    vnir = img[:, :, i_vnir].astype(float)
    nir = img[:, :, i_nir].astype(float)
    ndci = (vnir - nir) / (vnir + nir + eps)

    return ndci


def NDSI(img: np.array, i_vnir: int, i_swir: int) -> np.array:
    """Normalized Difference Snow Index
    The Sentinel-2 normalized difference snow index is a ratio of two bands: one in the VIR (Band 3) and one in the SWIR (Band 11).
    Values above 0.42 are usually snow
    """

    vnir = img[:, :, i_vnir].astype(float)
    swir = img[:, :, i_swir].astype(float)
    ndsi = (swir - vnir) / (vnir + swir + eps)

    return ndsi


def MSI(img: np.array, i_red: int, i_nir: int) -> np.array:
    """Moisture Index
    The MSI is a reflectance measurement, sensitive to increases in leaf water content.
    """

    swir = img[:, :, i_red].astype(float)
    nir = img[:, :, i_nir].astype(float)
    msi = swir / (nir + eps)

    return msi
