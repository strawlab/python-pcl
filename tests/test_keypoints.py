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


# keyPoints
### HarrisKeypoint3D ###
class TestHarrisKeypoint3D(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### NarfKeypoint ###
class TestNarfKeypoint(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### UniformSampling ###
class TestUniformSampling(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


def suite():
    suite = unittest.TestSuite()
    # keypoints
    suite.addTests(unittest.makeSuite(TestHarrisKeypoint3D))
    suite.addTests(unittest.makeSuite(TestNarfKeypoint))
    suite.addTests(unittest.makeSuite(TestUniformSampling))
    return suite


if __name__ == '__main__':
    unittest.main()
