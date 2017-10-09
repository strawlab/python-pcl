import os.path
import pickle
import shutil
import tempfile
import unittest

import pcl
import numpy as np


# features
### DifferenceOfNormalsEstimation ###
class TestDifferenceOfNormalsEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### IntegralImageNormalEstimation ###
class TestIntegralImageNormalEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### MomentOfInertiaEstimation ###
class TestMomentOfInertiaEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### NormalEstimation ###
class TestNormalEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### RangeImageBorderExtractor ###
class TestRangeImageBorderExtractor(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### VFHEstimation ###
class TestVFHEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


def suite():
    suite = unittest.TestSuite()
    # features
    suite.addTests(unittest.makeSuite(DifferenceOfNormalsEstimation))
    suite.addTests(unittest.makeSuite(IntegralImageNormalEstimation))
    suite.addTests(unittest.makeSuite(MomentOfInertiaEstimation))
    suite.addTests(unittest.makeSuite(NormalEstimation))
    suite.addTests(unittest.makeSuite(RangeImageBorderExtractor))
    suite.addTests(unittest.makeSuite(VFHEstimation))
    return suite

if __name__ == '__main__':
    unittest.main()

