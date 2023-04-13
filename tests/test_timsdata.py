import unittest

from tdfpy import timsdata

TDF_PATH = r'200ngHeLaPASEF_1min.d'


class TestTimsData(unittest.TestCase):

    def test_timsdata(self):
        with timsdata.timsdata_connect(TDF_PATH) as td:
            self.assertTrue(td.conn is not None)
