import os.path
import pickle
import shutil
import tempfile
import unittest

import pcl
import numpy as np

# segmentation
### ConditionalEuclideanClustering(1.7.2) ###
class TestConditionalEuclideanClustering(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### EuclideanClusterExtraction ###
class TestEuclideanClusterExtraction(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### MinCutSegmentation(1.7.2) ###
class TestMinCutSegmentation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### ProgressiveMorphologicalFilter ###
class TestProgressiveMorphologicalFilter(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### Segmentation ###
class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### SegmentationNormal ###
class TestSegmentationNormal(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


def suite():
    suite = unittest.TestSuite()
    # segmentation
    suite.addTests(unittest.makeSuite(TestConditionalEuclideanClustering))
    suite.addTests(unittest.makeSuite(TestEuclideanClusterExtraction))
    suite.addTests(unittest.makeSuite(TestMinCutSegmentation))
    suite.addTests(unittest.makeSuite(TestProgressiveMorphologicalFilter))
    suite.addTests(unittest.makeSuite(TestSegmentation))
    suite.addTests(unittest.makeSuite(TestSegmentationNormal))
    return suite


if __name__ == '__main__':
    unittest.main()

