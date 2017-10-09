import os.path
import pickle
import shutil
import tempfile
import unittest

import pcl
import numpy as np


# surface
### ConcaveHull ###
class TestConcaveHull(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### MovingLeastSquares ###
class TestMovingLeastSquares(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


def suite():
    suite = unittest.TestSuite()
    # surface
    suite.addTests(unittest.makeSuite(TestConcaveHull))
    suite.addTests(unittest.makeSuite(TestMovingLeastSquares))
    return suite

if __name__ == '__main__':
    unittest.main()

