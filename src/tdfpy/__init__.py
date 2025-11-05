"""
Package for working with TDF (Bruker Data File) data.

This package provides classes and functions for reading and manipulating TDF data using pandas DataFrames and
the TimsData format.

Modules:
- pandas_tdf: Contains the PandasTdf class for working with TDF data using pandas DataFrames.
- timsdata: Contains the TimsData class for working with TimsData format.
- spectra: Contains higher-level Pythonic API for working with MS1 spectrum data.

Attributes:
- __version__ (str): The current version of the package.
"""

from .pandas_tdf import PandasTdf
from .timsdata import TimsData, timsdata_connect
from .spectra import (
    Peak,
    Ms1Spectrum,
    merge_peaks,
    get_centroided_ms1_spectrum,
    get_centroided_ms1_spectra,
)

__version__ = "0.2.0"

__all__ = [
    "PandasTdf",
    "TimsData",
    "timsdata_connect",
    "Peak",
    "Ms1Spectrum",
    "merge_peaks",
    "get_centroided_ms1_spectrum",
    "get_centroided_ms1_spectra",
]
