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


### SampleConsensusModel ###
class TestSampleConsensus(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### SampleConsensusModelCylinder ###
class TestSampleConsensusModelCylinder(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### SampleConsensusModelLine ###
class TestSampleConsensusModelLine(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### SampleConsensusModelPlane ###
class TestSampleConsensusModelPlane(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### SampleConsensusModelRegistration ###
class TestSampleConsensusModelRegistration(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### SampleConsensusModelSphere ###
class TestSampleConsensusModelSphere(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


### SampleConsensusModelStick ###
class TestSampleConsensusModelStick(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)


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
    unittest.main()
