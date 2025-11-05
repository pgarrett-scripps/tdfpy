# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024

Major architectural refactor with high-level API and modern Python packaging.

### Added
- **High-level Pythonic API** (`spectra.py`):
  - `Peak` and `Ms1Spectrum` NamedTuples for immutable data structures
  - `get_centroided_ms1_spectrum()` for extracting single centroided MS1 spectra
  - `get_centroided_ms1_spectra()` generator for memory-efficient bulk processing
  - `merge_peaks()` function with configurable m/z and ion mobility tolerances
  - Support for both ppm/dalton m/z tolerances and relative/absolute ion mobility tolerances
  - Noise filtering with `min_peaks` parameter
- **Type annotations** throughout the codebase (Python 3.8+ compatible)
- **Comprehensive test suite** using pytest:
  - `test_spectra.py` for high-level API
  - `test_timsdata.py` for low-level bindings
  - `test_pandas_tdf.py` for DataFrame interface
  - Test data included in repository (`tests/data/200ngHeLaPASEF_1min.d/`)
- **Modern build system**:
  - Migrated to `pyproject.toml` (PEP 517/518)
  - Switched to `uv` package manager
  - Hatchling build backend
  - `Makefile` with common development commands
- **Documentation**:
  - Comprehensive README with multiple examples
  - `.github/copilot-instructions.md` for AI-assisted development
  - Improved docstrings throughout

### Changed
- **Project structure**: Migrated to src-based layout (`src/tdfpy/`)
- **Dependencies**: Relaxed strict version pinning for better compatibility
- **API design**: Generator-based functions for large datasets (memory efficient)
- **Return time units**: High-level API returns minutes (low-level still uses seconds)
- **Peak merging algorithm**: Uses intensity-weighted averaging with binary search optimization

### Removed
- Strict numpy/pandas version dependencies (now flexible ranges)
- Unicode import from numpy (no longer needed in modern versions)

## [0.1.7] - 2023

### Added
- PRM (Parallel Reaction Monitoring) related database tables
- `is_dda` and `is_prm` properties to distinguish acquisition modes
- GitHub Actions workflows for pytest and pylint

## [0.1.6] - 2023

### Changed
- Updated numpy and pandas version requirements

### Removed
- Unicode import from numpy (deprecated)

## [0.1.3] - 2022

### Added
- Logging support throughout the package
- Test data moved into repository for easier testing
- Updated numpy and pandas dependencies

## [0.1.2] - 2022

### Added
- Context manager support (`timsdata_connect()`) for automatic resource cleanup
- `with` statement support for `TimsData` class

## [0.1.0] - 2021

Initial release with basic functionality.

### Added
- `TimsData` class for low-level access to Bruker `.tdf` and `.tdf_bin` files
- `PandasTdf` class for DataFrame interface to SQLite metadata
- ctypes bindings to Bruker's native libraries
- Cross-platform support (Windows DLL, Linux SO)
- Basic reading of frames, scans, and PASEF MS/MS data
