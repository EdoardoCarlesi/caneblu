# setup.py
from pathlib import Path
from setuptools import setup

ROOT = Path(__file__).parent
readme = (ROOT / "README.md")
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name="caneblu",
    version="0.1.0",
    description="Utilities for satellite data processing",
    long_description=long_description,
    long_description_content_type="text/markdown" if readme.exists() else "text/plain",
    author="Edoardo Carlesi",
    author_email="ecarlesi83@gmail.com",
    url="https://github.com/EdoardoCarlesi/caneblu",
    license="MIT",
    # Map the repository root to the 'satdat_utils' package
    packages=["caneblu"],
    python_requires=">=3.10",
    install_requires=[
        "earthengine-api>=1.1.3",
        "geopandas>=1.0.1",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "streamlit>=1.38.0",
        "sentinelhub>=3.10.2",
        "shapely>=2.0.6",
        "tqdm>=4.66.5",
        "matplotlib>=3.9.0",
        "pillow>=10.3.0",
        "tifffile>=2024.8.30",
        "rasterio>=1.3.10",
        "pyproj>=3.6.1",
        "fonttools>=4.53.1",
        "minio>=7.2.7",
        "arrow>=1.3.0",
        "scikit-learn>=1.5.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
