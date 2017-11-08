import os.path
import pickle
import shutil
import tempfile
import unittest

import pcl
import numpy as np

from nose.plugins.attrib import attr


class TestKdTree(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        # Define two dense sets of points of sizes 30 and 170, resp.
        a = rng.randn(100, 3).astype(np.float32)
        a[:30] -= 42

        self.pc = pcl.PointCloud(a)
        self.kd = pcl.KdTreeFLANN(self.pc)

    def testException(self):
        self.assertRaises(TypeError, pcl.KdTreeFLANN)
        self.assertRaises(TypeError, self.kd.nearest_k_search_for_cloud, None)

    def testKNN(self):
        # Small cluster
        ind, sqdist = self.kd.nearest_k_search_for_point(self.pc, 0, k=2)
        for i in ind:
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, 30)
        for d in sqdist:
            self.assertGreaterEqual(d, 0)

        # Big cluster
        for ref, k in ((80, 1), (59, 3), (60, 10)):
            ind, sqdist = self.kd.nearest_k_search_for_point(self.pc, ref, k=k)
            for i in ind:
                self.assertGreaterEqual(i, 30)
            for d in sqdist:
                self.assertGreaterEqual(d, 0)


class TestKdTreeFLANN(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        # Define two dense sets of points of sizes 30 and 170, resp.
        a = rng.randn(100, 3).astype(np.float32)
        a[:30] -= 42

        self.pc = pcl.PointCloud(a)
        self.kd = pcl.KdTreeFLANN(self.pc)

    def testException(self):
        self.assertRaises(TypeError, pcl.KdTreeFLANN)
        self.assertRaises(TypeError, self.kd.nearest_k_search_for_cloud, None)

    def testKNN(self):
        # Small cluster
        ind, sqdist = self.kd.nearest_k_search_for_point(self.pc, 0, k=2)
        for i in ind:
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, 30)
        for d in sqdist:
            self.assertGreaterEqual(d, 0)

        # Big cluster
        for ref, k in ((80, 1), (59, 3), (60, 10)):
            ind, sqdist = self.kd.nearest_k_search_for_point(self.pc, ref, k=k)
            for i in ind:
                self.assertGreaterEqual(i, 30)
            for d in sqdist:
                self.assertGreaterEqual(d, 0)


def suite():
    suite = unittest.TestSuite()
    # ketree
    suite.addTests(unittest.makeSuite(TestKdTree))
    suite.addTests(unittest.makeSuite(TestKdTreeFLANN))
    return suite


if __name__ == '__main__':
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
