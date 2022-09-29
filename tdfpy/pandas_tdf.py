import sqlite3
from dataclasses import dataclass

import pandas as pd

from tdfpy.constants import TableNames


def convert_table_to_df(db:str, table_name:str) -> pd.DataFrame:
    print(f'Fetching {table_name} from {db}')
    with sqlite3.connect(str(db)) as conn:
        df = pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
        return df


@dataclass
class PandasTdf:
    db_path: str

    @property
    def calibration_info(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.CALIBRATION_INFO.value)

    @property
    def dia_frame_msms_info(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.DIA_FRAME_MSMS_INFO.value)

    @property
    def dia_frame_msms_window_groups(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.DIA_FRAME_MSMS_WINDOW_GROUPS.value)

    @property
    def dia_frame_msms_windows(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.DIA_FRAME_MSMS_WINDOWS.value)

    @property
    def error_log(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.ERROR_LOG.value)

    @property
    def frame_msms_info(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.FRAME_MSMS_WINDOW.value)

    @property
    def frame_properties(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.FRAME_PROPERTIES.value)

    @property
    def frames(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.FRAMES.value)

    @property
    def global_metadata(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.GLOBAL_METADATA.value)

    @property
    def group_properties(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.GROUP_PROPERTIES.value)

    @property
    def mz_calibration(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.MZ_CALIBRATION.value)

    @property
    def pasef_frame_msms_info(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.PASEF_FRAME_MSMS_INFO.value)

    @property
    def precursors(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.PRECURSORS.value)

    @property
    def properties(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.PROPERTIES.value)

    @property
    def property_definitions(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.PROPERTY_DEFINITIONS.value)

    @property
    def property_groups(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.PROPERTY_GROUPS.value)

    @property
    def segments(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.SEGMENTS.value)

    @property
    def tims_calibration(self) -> pd.DataFrame:
        return convert_table_to_df(self.db_path, TableNames.TIMS_CALIBRATION.value)
