# TDFpy

Basic pip package to handle Bruker's .tdf and .tdfbin files with both low-level and high-level APIs.

## How to install
```bash
pip install tdfpy
```

or

```bash
git clone https://github.com/pgarrett-scripps/tdfpy.git
cd tdfpy
pip install .
```

## Quick Start Examples

### High-Level API (Recommended for MS1 Spectra)
```python
from tdfpy import timsdata_connect, get_centroided_ms1_spectrum, get_centroided_ms1_spectra

# Read a single MS1 spectrum with centroiding
with timsdata_connect('path/to/data.d') as td:
    spectrum = get_centroided_ms1_spectrum(td, frame_id=1)
    print(f"Found {spectrum.num_peaks} peaks at RT={spectrum.retention_time:.2f} min")

    for peak in spectrum.peaks[:5]:
        print(f"m/z: {peak.mz:.4f}, intensity: {peak.intensity:.2f}, mobility: {peak.ion_mobility:.4f}")

# Read all MS1 spectra
with timsdata_connect('path/to/data.d') as td:
    spectra = get_centroided_ms1_spectra(td)
    print(f"Loaded {len(spectra)} MS1 spectra")
```

### Low-Level Database Access
```python
from tdfpy import PandasTdf

pd_tdf = PandasTdf('path/to/analysis.tdf')
pd_tdf.precursors  # returns pd.DataFrame containing Precursors table
pd_tdf.frames      # returns pd.DataFrame containing Frames table
pd_tdf.properties  # returns pd.DataFrame containing Properties table
```

### Low-Level Binary Data Access
```python
from tdfpy import TimsData

td = TimsData('path/to/data.d')
td.readPasefMsMsForFrame(1)  # return msms spectra for first msms frame
td.close()
```


