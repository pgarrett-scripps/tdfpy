"""
Package for working with TDF (Bruker Data File) data.

This package provides classes and functions for reading and manipulating TDF data using pandas DataFrames and
the TimsData format.

Modules:
- pandas_tdf: Contains the PandasTdf class for working with TDF data using pandas DataFrames.
- timsdata: Contains the TimsData class for working with TimsData format.

Attributes:
- __version__ (str): The current version of the package.
"""

from .pandas_tdf import PandasTdf
from .timsdata import TimsData

__version__ = '0.1.7'
