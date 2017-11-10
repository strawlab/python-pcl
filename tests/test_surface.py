import os.path
import pickle
import shutil
import tempfile
import unittest

import pcl
import numpy as np


from nose.plugins.attrib import attr


# surface
### ConcaveHull ###
class TestConcaveHull(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load("tests" + os.path.sep + "flydracyl.pcd")
        self.surf = self.p.make_ConcaveHull()
        # self.surf = pcl.ConcaveHull()
        # self.surf.setInputCloud()

    def testreconstruct(self):
        alpha = 1.0
        self.surf.set_Alpha(alpha)
        clonepc = self.surf.reconstruct()
        # new instance is returned
        self.assertNotEqual(self.p, clonepc)
        # concavehull retains the same number of points?
        self.assertNotEqual(self.p.size, clonepc.size)


### MovingLeastSquares ###
class TestMovingLeastSquares(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load("tests" + os.path.sep + "flydracyl.pcd")
        self.surf = self.p.make_moving_least_squares()

    def testFilter(self):
        self.surf.set_search_radius(0.5)
        self.surf.set_polynomial_order(2)
        self.surf.set_polynomial_fit(True)
        f = self.surf.process()
        # new instance is returned
        self.assertNotEqual(self.p, f)
        # mls filter retains the same number of points
        self.assertEqual(self.p.size, f.size)


def suite():
    suite = unittest.TestSuite()

    # surface
    suite.addTests(unittest.makeSuite(TestConcaveHull))
    suite.addTests(unittest.makeSuite(TestMovingLeastSquares))

    return suite


if __name__ == '__main__':
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
