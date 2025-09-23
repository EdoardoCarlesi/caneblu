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
    url="https://github.com/EdoardoCarlesi/caneblu"
    license="MIT",
    # Map the repository root to the 'satdat_utils' package
    packages=["caneblu"],
    package_dir={"caneblu": "."},
    python_requires=">=3.10",
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
