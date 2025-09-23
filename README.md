CANEBLU
============

Utilities for working with satellite and geospatial data. This package provides a set of modular helpers for processing imagery, building composites and indices, basic geocoding workflows, harvesting pipelines, and general utilities.

- Clean, modular API (submodules for specific tasks)
- Lightweight imports with lazy-loading submodules
- Designed to be extended and integrated into larger pipelines

Features
--------

- Composite and index helpers for satellite imagery workflows
- Geocoding helpers and coordinate utilities
- Data harvesting/orchestration utilities
- General-purpose utilities used across modules
- Import only what you need via submodules


Installation
------------

Minimum Python version:

- Python 3.10+


Quick start
-----------

Import the package and submodules as needed:

   import caneblu as cb

Import specific submodules directly:

   from caneblu import geocoding, composite_indices


Modules
-------

- ``composite_indices``: Helpers for building image composites and indices
- ``geocoding``: Geocoding and coordinate utilities
- ``harvester``: Utilities for harvesting/orchestrating data workflows
- ``satellite_image_utils``: Common imagery helpers (I/O, transforms, etc.)
- ``utils``: Shared helpers used across the package

Each module includes docstrings with function-level usage notes.


Project layout
--------------

   caneblu/
     setup.py
     LICENSE
     (package modules at repository root)

When installed, the package is importable as ``caneblu``.

- Submit a pull request describing your changes


License
-------

This project is licensed under the MIT License. See the ``LICENSE`` file for details.
