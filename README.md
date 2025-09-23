CANEBLU
============

Utilities for working with satellite and geospatial data. This package provides a set of modular helpers for processing imagery, building composites and indices, basic geocoding workflows, harvesting pipelines, and general utilities.

- Clean, modular API (submodules for specific tasks)
- Lightweight imports with lazy-loading submodules
- Designed to be extended and integrated into larger pipelines


Table of contents
-----------------

- Features_
- Installation_
- Quick start_
- Modules_
- Project layout_
- Compatibility_
- Contributing_
- License_


Features
--------

- Composite and index helpers for satellite imagery workflows
- Geocoding helpers and coordinate utilities
- Data harvesting/orchestration utilities
- General-purpose utilities used across modules
- Import only what you need via submodules


Installation
------------

From PyPI (recommended once published):

.. code-block:: bash

   pip install caneblu

From source (in the repository root):

.. code-block:: bash

   pip install .

Install in editable/development mode:

.. code-block:: bash

   pip install -e .

Minimum Python version:

- Python 3.10+


Quick start
-----------

Import the package and submodules as needed:

.. code-block:: python

   import caneblu as cb

   # Submodules are available under the package namespace
   # Access a submodule explicitly:
   # sdu.composite_indices
   # sdu.geocoding
   # sdu.harvester
   # sdu.satellite_image_utils
   # sdu.utils

Import specific submodules directly:

.. code-block:: python

   from caneblu import geocoding, composite_indices

   # Use submodule functionality as appropriate for your workflow.
   # (Refer to inline docstrings in each module for details.)

Note: Submodules are imported lazily the first time you access them from the package namespace.


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

.. code-block:: text

   caneblu/
     setup.py
     LICENSE
     (package modules at repository root)

When installed, the package is importable as ``caneblu``.


Compatibility
-------------

- Python 3.8+
- Platform: OS-independent


Contributing
------------

Contributions are welcome!

- Fork the repository
- Create a feature branch
- Add/update tests and docs where applicable
- Submit a pull request describing your changes

Before submitting, please ensure:

- Code is formatted and linted consistently
- New functionality includes minimal examples or docstrings


License
-------

This project is licensed under the MIT License. See the ``LICENSE`` file for details.
