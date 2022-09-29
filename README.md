# TDFpy

basic pip package to handle Bruker's .tdf and .tdfbin files. 
## How to install
- pip install tdfpy

or

- git clone https://github.com/pgarrett-scripps/tdfpy.git
- cd tdfpy
- pip install .

## How to access analysis.tdf db
- from tdfpy import PandasTdf
- pd_tdf = PandasTdf('path/to/analysis.tdf')
- pd_tdf.precursors -> returns pd.DataFrame containing Precursors table
- pd_tdf.frames -> returns pd.DataFrame containing Frames table
- pd_tdf.properties -> returns pd.DataFrame containing Properties table

## How to access analysis.tdf_bin db
- from tdfpy.timsdata import TimsData
- td = TimsData('path/to/dfolder')
- td.readPasefMsMsForFrame(1) -> return msms spectra for first msms frame


