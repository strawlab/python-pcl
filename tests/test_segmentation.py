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
        self.segment = pcl.ConditionalEuclideanClustering()


### EuclideanClusterExtraction ###
class TestEuclideanClusterExtraction(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.segment = pcl.EuclideanClusterExtraction()


### MinCutSegmentation(1.7.2) ###
class TestMinCutSegmentation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.segment = pcl.MinCutSegmentation()


### ProgressiveMorphologicalFilter ###
class TestProgressiveMorphologicalFilter(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.segment = pcl.ProgressiveMorphologicalFilter()


### Segmentation ###
class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.segment = pcl.Segmentation()


### SegmentationNormal ###
class TestSegmentationNormal(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.segment = pcl.SegmentationNormal()


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
