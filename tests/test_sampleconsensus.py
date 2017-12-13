import os.path
import pickle
import shutil
import tempfile
import unittest

import pcl
import numpy as np


# _data = [(i, 2 * i, 3 * i + 0.2) for i in range(500)]

import random
_data = [(random.random(), random.random(), random.random())
         for i in range(500)]


# sample_consensus

### RandomSampleConsensus ###
class TestRandomSampleConsensus(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)

    # def test_SampleConsensusModel(self):
    #     model = pcl.SampleConsensusModel(self.p)
    #     ransac = pcl.RandomSampleConsensus (model)
    #     ransac.set_DistanceThreshold (.01)
    #     ransac.computeModel()
    #     inliers = ransac.get_Inliers()
    #
    #     # print(str(len(inliers)))
    #     self.assertNotEqual(len(inliers), 0)

    # def test_SampleConsensusModelCylinder(self):
    #     model_cy = pcl.SampleConsensusModelCylinder(self.p)
    #     ransac = pcl.RandomSampleConsensus (model_cy)
    #     ransac.set_DistanceThreshold (.01)
    #     ransac.computeModel()
    #     inliers = ransac.get_Inliers()
    #
    #     # print(str(len(inliers)))
    #     self.assertNotEqual(len(inliers), 0)

    def test_SampleConsensusModelLine(self):
        model_line = pcl.SampleConsensusModelLine(self.p)
        ransac = pcl.RandomSampleConsensus(model_line)
        ransac.set_DistanceThreshold(.01)
        ransac.computeModel()
        inliers = ransac.get_Inliers()

        # print(str(len(inliers)))
        self.assertNotEqual(len(inliers), 0)

    def test_ModelPlane(self):
        model_p = pcl.SampleConsensusModelPlane(self.p)
        ransac = pcl.RandomSampleConsensus(model_p)
        ransac.set_DistanceThreshold(.01)
        ransac.computeModel()
        inliers = ransac.get_Inliers()

        # print(str(len(inliers)))
        self.assertNotEqual(len(inliers), 0)

        final = pcl.PointCloud()

        if len(inliers) != 0:
            finalpoints = np.zeros((len(inliers), 3), dtype=np.float32)

            for i in range(0, len(inliers)):
                finalpoints[i][0] = self.p[inliers[i]][0]
                finalpoints[i][1] = self.p[inliers[i]][1]
                finalpoints[i][2] = self.p[inliers[i]][2]

            final.from_array(finalpoints)

        self.assertNotEqual(final.size, 0)
        pass

    # def test_SampleConsensusModelRegistration(self):
    #     model_reg = pcl.SampleConsensusModelRegistration(self.p)
    #     ransac = pcl.RandomSampleConsensus (model_reg)
    #     ransac.set_DistanceThreshold (.01)
    #     ransac.computeModel()
    #     inliers = ransac.get_Inliers()
    #
    #     # print(str(len(inliers)))
    #     self.assertNotEqual(len(inliers), 0)

    def test_ModelSphere(self):
        model_s = pcl.SampleConsensusModelSphere(self.p)
        ransac = pcl.RandomSampleConsensus(model_s)
        ransac.set_DistanceThreshold(.01)
        ransac.computeModel()
        inliers = ransac.get_Inliers()

        # print(str(len(inliers)))
        self.assertNotEqual(len(inliers), 0)

        final = pcl.PointCloud()

        if len(inliers) != 0:
            finalpoints = np.zeros((len(inliers), 3), dtype=np.float32)

            for i in range(0, len(inliers)):
                finalpoints[i][0] = self.p[inliers[i]][0]
                finalpoints[i][1] = self.p[inliers[i]][1]
                finalpoints[i][2] = self.p[inliers[i]][2]

            final.from_array(finalpoints)

        self.assertNotEqual(final.size, 0)
        pass

    def test_SampleConsensusModelStick(self):
        model_st = pcl.SampleConsensusModelStick(self.p)
        ransac = pcl.RandomSampleConsensus(model_st)
        ransac.set_DistanceThreshold(.01)
        ransac.computeModel()
        inliers = ransac.get_Inliers()

        # print(str(len(inliers)))
        self.assertNotEqual(len(inliers), 0)

    # def testException(self):
    #     self.assertRaises(TypeError, pcl.RandomSampleConsensus)
    #     pass


def suite():
    suite = unittest.TestSuite()

    # Sampleconsensus
    suite.addTests(unittest.makeSuite(TestRandomSampleConsensus))

    return suite


if __name__ == '__main__':
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
