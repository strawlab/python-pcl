import os.path
import pickle
import shutil
import tempfile
import unittest

import pcl
import numpy as np

# sample_consensus

### RandomSampleConsensus ###
class TestRandomSampleConsensus(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.smp_con = pcl.RandomSampleConsensus()


#   def test_computeModel
#     def computeModel(self):
#         self.me.computeModel(0)
# 
#     # base Class(SampleConsensus)
#     def set_DistanceThreshold(self, double param):
#         self.me.setDistanceThreshold(param)
# 
#     # base Class(SampleConsensus)
#     def get_Inliers(self):
#         cdef vector[int] inliers
#         self.me.getInliers(inliers)
#         return inliers


### SampleConsensusModel ###
class TestSampleConsensus(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.smp_con = pcl.RandomSampleConsensus()


### SampleConsensusModelCylinder ###
class TestSampleConsensusModelCylinder(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.smp_con = pcl.RandomSampleConsensus()


### SampleConsensusModelLine ###
class TestSampleConsensusModelLine(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.smp_con = pcl.RandomSampleConsensus()


### SampleConsensusModelPlane ###
class TestSampleConsensusModelPlane(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.smp_con = pcl.RandomSampleConsensus()


### SampleConsensusModelRegistration ###
class TestSampleConsensusModelRegistration(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.smp_con = pcl.RandomSampleConsensus()


### SampleConsensusModelSphere ###
class TestSampleConsensusModelSphere(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.smp_con = pcl.RandomSampleConsensus()


### SampleConsensusModelStick ###
class TestSampleConsensusModelStick(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.smp_con = pcl.RandomSampleConsensus()


def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestRandomSampleConsensus))
    suite.addTests(unittest.makeSuite(TestSampleConsensusModel))
    suite.addTests(unittest.makeSuite(TestSampleConsensusModelCylinder))
    suite.addTests(unittest.makeSuite(TestSampleConsensusModelLine))
    suite.addTests(unittest.makeSuite(TestSampleConsensusModelPlane))
    suite.addTests(unittest.makeSuite(TestSampleConsensusModelRegistration))
    suite.addTests(unittest.makeSuite(TestSampleConsensusModelSphere))
    suite.addTests(unittest.makeSuite(TestSampleConsensusModelStick))

    return suite


if __name__ == '__main__':
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
