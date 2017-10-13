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
        self.p = pcl.load("tests" + os.path.sep + "flydracyl.pcd")

    def testFilter(self):
        mls = self.p.make_moving_least_squares()
        mls.set_search_radius(0.5)
        mls.set_polynomial_order(2)
        mls.set_polynomial_fit(True)
        f = mls.process()
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
    unittest.main()
