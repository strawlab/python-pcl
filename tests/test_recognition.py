import os.path
import pickle
import shutil
import tempfile
import unittest

import pcl
import numpy as np

_data = [(i, 2 * i, 3 * i + 0.2) for i in range(5)]
_DATA = """0.0, 0.0, 0.2;
           1.0, 2.0, 3.2;
           2.0, 4.0, 6.2;
           3.0, 6.0, 9.2;
           4.0, 8.0, 12.2"""

# recognition


def suite():
    suite = unittest.TestSuite()
    # recognition
    # suite.addTests(unittest.makeSuite(...))
    return suite


if __name__ == '__main__':
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
