"""
Higher-level Pythonic API for working with MS1 spectrum data from Bruker timsTOF files.

This module provides a cleaner interface using NamedTuples and convenience functions
for reading centroided MS1 spectra with optional peak merging.
"""

from typing import Generator, List, Literal, NamedTuple, Optional

import numpy as np

from .timsdata import TimsData


class Peak(NamedTuple):
    """Represents a single mass spec peak.

    Attributes:
        mz: Mass-to-charge ratio
        intensity: Peak intensity (area)
        ion_mobility: Ion mobility (1/K0) - note that for DLL-centroided spectra,
                     this represents the average ion mobility across the frame
    """
    mz: float
    intensity: float
    ion_mobility: float


class Ms1Spectrum(NamedTuple):
    """Represents a complete MS1 spectrum from a single frame.

    Attributes:
        spectrum_index: Sequential index of this spectrum
        frame_id: Original frame ID from the TDF file
        retention_time: Retention time in minutes
        num_peaks: Number of peaks in the spectrum
        peaks: List of Peak objects
    """
    spectrum_index: int
    frame_id: int
    retention_time: float
    num_peaks: int
    peaks: List[Peak]

    def __repr__(self) -> str:
        return (
            f"Ms1Spectrum(index={self.spectrum_index}, frame_id={self.frame_id}, "
            f"retention_time={self.retention_time:.2f} min, num_peaks={self.num_peaks})"
        )


def merge_peaks(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ion_mobility_array: np.ndarray,
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: Optional[int] = None
) -> List[Peak]:
    """Merge nearby peaks using m/z and ion mobility tolerances.

    This function implements a greedy clustering algorithm that merges peaks
    within specified m/z and ion mobility windows. Peaks are processed in
    descending order of intensity.

    Args:
        mz_array: Array of m/z values
        intensity_array: Array of intensity values
        ion_mobility_array: Array of ion mobility (1/K0) values
        mz_tolerance: Tolerance for m/z matching
        mz_tolerance_type: Type of m/z tolerance - "ppm" or "da" (daltons)
        im_tolerance: Tolerance for ion mobility matching
        im_tolerance_type: Type of ion mobility tolerance - "relative" or "absolute"
        min_peaks: Minimum number of nearby peaks required to form a cluster.
                  Set to 0 or 1 to keep all peaks.
        max_peaks: Maximum number of merged peaks to return (keeps highest intensity)

    Returns:
        List of merged Peak objects

    Example:
        >>> mz = np.array([100.0, 100.001, 200.0])
        >>> intensity = np.array([1000.0, 500.0, 2000.0])
        >>> im = np.array([0.8, 0.8, 0.9])
        >>> peaks = merge_peaks(mz, intensity, im, mz_tolerance=10, mz_tolerance_type="ppm")
    """
    if len(mz_array) == 0:
        return []

    # Pre-compute tolerances
    if mz_tolerance_type == "ppm":
        mz_tol_factor = mz_tolerance / 1e6
        mz_tol_abs = 0.0
    else:
        mz_tol_abs = mz_tolerance
        mz_tol_factor = 0.0

    if im_tolerance_type == "relative":
        mobility_tol_factor = im_tolerance
        mobility_tol_abs = 0.0
    else:
        mobility_tol_abs = im_tolerance
        mobility_tol_factor = 0.0

    # Sort by mz for binary search
    sort_idx = np.argsort(mz_array)
    mz_array = mz_array[sort_idx]
    intensity_array = intensity_array[sort_idx]
    ion_mobility_array = ion_mobility_array[sort_idx]

    # Sort by intensity for greedy clustering
    intensity_order = np.argsort(intensity_array)[::-1]

    # Use boolean mask for tracking used peaks
    used_mask = np.zeros(len(mz_array), dtype=bool)
    merged_peaks: List[Peak] = []

    for peak_idx in intensity_order:
        if used_mask[peak_idx]:
            continue

        mz_peak = float(mz_array[peak_idx])
        intensity_peak = float(intensity_array[peak_idx])
        mobility_peak = float(ion_mobility_array[peak_idx])

        # Calculate tolerances
        mz_tol = mz_peak * mz_tol_factor if mz_tolerance_type == "ppm" else mz_tol_abs
        mobility_tol = (
            mobility_peak * mobility_tol_factor if im_tolerance_type == "relative" else mobility_tol_abs
        )

        # Binary search for mz range
        left_mz = mz_peak - mz_tol
        right_mz = mz_peak + mz_tol
        left_idx = int(np.searchsorted(mz_array, left_mz, side="left"))
        right_idx = int(np.searchsorted(mz_array, right_mz, side="right"))

        # Only check mobility in the mz window
        mz_window_slice = slice(left_idx, right_idx)
        mobility_window = ion_mobility_array[mz_window_slice]
        intensity_window = intensity_array[mz_window_slice]
        mz_window = mz_array[mz_window_slice]
        used_window = used_mask[mz_window_slice]

        # Find nearby peaks in mobility dimension
        mobility_diff = np.abs(mobility_window - mobility_peak)
        nearby_mask = (mobility_diff <= mobility_tol) & ~used_window

        # Check minimum peaks requirement
        if min_peaks > 0 and np.sum(nearby_mask) < min_peaks:
            # Not enough nearby peaks, skip this peak
            used_mask[peak_idx] = True
            continue

        if not np.any(nearby_mask):
            # Edge case: no nearby peaks (shouldn't happen but be safe)
            merged_peaks.append(Peak(mz=mz_peak, intensity=intensity_peak, ion_mobility=mobility_peak))
            used_mask[peak_idx] = True
            continue

        # Merge peaks using weighted average
        nearby_intensities = intensity_window[nearby_mask]
        merged_intensity = float(np.sum(nearby_intensities))
        merged_mz = float(np.average(mz_window[nearby_mask], weights=nearby_intensities))
        merged_mobility = float(np.average(mobility_window[nearby_mask], weights=nearby_intensities))

        merged_peaks.append(Peak(mz=merged_mz, intensity=merged_intensity, ion_mobility=merged_mobility))

        # Mark as used (convert local indices to global)
        global_nearby_idx = np.where(nearby_mask)[0] + left_idx
        used_mask[global_nearby_idx] = True

        if max_peaks and len(merged_peaks) >= max_peaks:
            break

    return merged_peaks


def get_centroided_ms1_spectrum(
    td: TimsData,
    frame_id: int,
    spectrum_index: Optional[int] = None,
    include_ion_mobility: bool = True,
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: Optional[int] = None
) -> Ms1Spectrum:
    """Extract a centroided MS1 spectrum for a single frame.

    This function reads raw scans from the frame, converts indices to m/z values,
    collects all peaks with their ion mobility values, and optionally merges
    nearby peaks based on m/z and ion mobility tolerances.

    Args:
        td: TimsData instance connected to the analysis directory
        frame_id: Frame ID to extract
        spectrum_index: Optional index for this spectrum (defaults to frame_id)
        include_ion_mobility: If True, calculate and include ion mobility for each peak
        mz_tolerance: Tolerance for m/z matching when merging
        mz_tolerance_type: Type of m/z tolerance - "ppm" or "da" (daltons)
        im_tolerance: Tolerance for ion mobility matching when merging
        im_tolerance_type: Type of ion mobility tolerance - "relative" or "absolute"
        min_peaks: Minimum number of nearby peaks required to form a cluster (0 or 1 keeps all peaks)
        max_peaks: Maximum number of merged peaks to return

    Returns:
        Ms1Spectrum object containing peaks and metadata

    Raises:
        ValueError: If the frame_id doesn't exist or is not an MS1 frame
        RuntimeError: If the TimsData connection is not open

    Example:
        >>> with timsdata_connect('path/to/data.d') as td:
        ...     # Get spectrum with default merging
        ...     spectrum = get_centroided_ms1_spectrum(td, frame_id=1)
        ...     print(f"Found {spectrum.num_peaks} peaks")
        ...
        ...     # Get spectrum without merging
        ...     spectrum = get_centroided_ms1_spectrum(td, frame_id=1)
        ...
        ...     # Custom tolerances
        ...     spectrum = get_centroided_ms1_spectrum(
        ...         td, frame_id=1, mz_tolerance=10, im_tolerance=0.1
        ...     )
    """
    if td.conn is None:
        raise RuntimeError("TimsData connection is not open")

    # Get frame metadata from the database
    cursor = td.conn.cursor()
    cursor.execute(
        "SELECT Time, NumScans, MsMsType FROM Frames WHERE Id = ?",
        (frame_id,)
    )
    result = cursor.fetchone()

    if result is None:
        raise ValueError(f"Frame {frame_id} not found in database")

    retention_time_sec, num_scans, msms_type = result

    if msms_type != 0:
        raise ValueError(f"Frame {frame_id} is not an MS1 frame (MsMsType={msms_type})")

    retention_time_min = retention_time_sec / 60.0

    if num_scans == 0:
        return Ms1Spectrum(
            spectrum_index=spectrum_index if spectrum_index is not None else frame_id,
            frame_id=frame_id,
            retention_time=retention_time_min,
            num_peaks=0,
            peaks=[]
        )

    # Pre-compute ion mobility values for each scan if requested
    if include_ion_mobility:
        ion_mobility = td.scanNumToOneOverK0(frame_id, np.arange(0, num_scans))
    else:
        ion_mobility = np.zeros(num_scans, dtype=np.float64)

    # Read all scans at once
    results = td.readScans(frame_id, 0, num_scans)

    # Pre-allocate arrays with estimated size
    total_peaks = sum(len(idx) for idx, _ in results)

    if total_peaks == 0:
        return Ms1Spectrum(
            spectrum_index=spectrum_index if spectrum_index is not None else frame_id,
            frame_id=frame_id,
            retention_time=retention_time_min,
            num_peaks=0,
            peaks=[]
        )

    mz_array = np.empty(total_peaks, dtype=np.float64)
    intensity_array = np.empty(total_peaks, dtype=np.float64)
    ion_mobility_array = np.empty(total_peaks, dtype=np.float64)

    # Collect all peaks from all scans
    offset = 0
    for scan_index, (index_array, intensity_scan) in enumerate(results):
        n_peaks = len(index_array)
        if n_peaks == 0:
            continue

        # Convert indices to m/z in batch
        mz_values = td.indexToMz(frame_id, index_array)

        # Fill pre-allocated arrays
        mz_array[offset : offset + n_peaks] = mz_values
        intensity_array[offset : offset + n_peaks] = intensity_scan
        ion_mobility_array[offset : offset + n_peaks] = ion_mobility[scan_index]
        offset += n_peaks

    # Trim arrays to actual size
    mz_array = mz_array[:offset]
    intensity_array = intensity_array[:offset]
    ion_mobility_array = ion_mobility_array[:offset]

    # Apply peak merging if requested
    peaks = merge_peaks(
        mz_array=mz_array,
        intensity_array=intensity_array,
        ion_mobility_array=ion_mobility_array,
        mz_tolerance=mz_tolerance,
        mz_tolerance_type=mz_tolerance_type,
        im_tolerance=im_tolerance,
        im_tolerance_type=im_tolerance_type,
        min_peaks=min_peaks,
        max_peaks=max_peaks
    )

    # Apply max_peaks limit if specified
    if max_peaks and len(peaks) > max_peaks:
        # Sort by intensity and take top N
        peaks = sorted(peaks, key=lambda p: p.intensity, reverse=True)[:max_peaks]

    return Ms1Spectrum(
        spectrum_index=spectrum_index if spectrum_index is not None else frame_id,
        frame_id=frame_id,
        retention_time=retention_time_min,
        num_peaks=len(peaks),
        peaks=peaks
    )


def get_centroided_ms1_spectra(
    td: TimsData,
    frame_ids: Optional[List[int]] = None,
    include_ion_mobility: bool = True,
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: Optional[int] = None
) -> Generator[Ms1Spectrum, None, None]:
    """Extract centroided MS1 spectra for multiple frames.

    Convenience function to extract multiple MS1 spectra. If frame_ids is not specified,
    all MS1 frames in the file will be processed.

    Args:
        td: TimsData instance connected to the analysis directory
        frame_ids: Optional list of frame IDs to extract. If None, extracts all MS1 frames.
        include_ion_mobility: If True, calculate and include ion mobility for each peak
        mz_tolerance: Tolerance for m/z matching when merging
        mz_tolerance_type: Type of m/z tolerance - "ppm" or "da" (daltons)
        im_tolerance: Tolerance for ion mobility matching when merging
        im_tolerance_type: Type of ion mobility tolerance - "relative" or "absolute"
        min_peaks: Minimum number of nearby peaks required to form a cluster
        max_peaks: Maximum number of merged peaks to return per spectrum

    Returns:
        List of Ms1Spectrum objects, ordered by frame ID

    Example:
        >>> with timsdata_connect('path/to/data.d') as td:
        ...     # Get all MS1 spectra with default merging
        ...     spectra = get_centroided_ms1_spectra(td)
        ...     print(f"Found {len(spectra)} MS1 spectra")
        ...
        ...     # Get specific frames without merging
        ...     spectra = get_centroided_ms1_spectra(td, frame_ids=[1, 2, 3], merge=False)
        ...
        ...     # Custom tolerances
        ...     spectra = get_centroided_ms1_spectra(
        ...         td, mz_tolerance=10, im_tolerance=0.1
        ...     )
    """
    if td.conn is None:
        raise RuntimeError("TimsData connection is not open")

    cursor = td.conn.cursor()

    if frame_ids is None:
        # Get all MS1 frame IDs
        cursor.execute("SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id")
        frame_ids = [row[0] for row in cursor.fetchall()]

    for idx, frame_id in enumerate(frame_ids):
        try:
            spectrum = get_centroided_ms1_spectrum(
                td,
                frame_id=frame_id,
                spectrum_index=idx,
                include_ion_mobility=include_ion_mobility,
                mz_tolerance=mz_tolerance,
                mz_tolerance_type=mz_tolerance_type,
                im_tolerance=im_tolerance,
                im_tolerance_type=im_tolerance_type,
                min_peaks=min_peaks,
                max_peaks=max_peaks
            )
            yield spectrum
        except (ValueError, RuntimeError) as e:
            # Log warning but continue processing
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to extract spectrum for frame {frame_id}: {e}"
            )
            continue

