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


# features
### DifferenceOfNormalsEstimation ###
class TestDifferenceOfNormalsEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.feat = pcl.DifferenceOfNormalsEstimation()


### IntegralImageNormalEstimation ###
class TestIntegralImageNormalEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.feat = pcl.IntegralImageNormalEstimation(self.p)


    def test_set_NormalEstimation_Method_AVERAGE_3D_GRADIENT (self):
        self.feat.set_NormalEstimation_Method_AVERAGE_3D_GRADIENT()
        # f = self.feat.compute(self.p)
        
        # check
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)

    def test_set_NormalEstimation_Method_COVARIANCE_MATRIX (self):
        self.feat.set_NormalEstimation_Method_COVARIANCE_MATRIX()
        # f = self.feat.compute(self.p)
        
        # check
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)


    def test_set_NormalEstimation_Method_AVERAGE_DEPTH_CHANGE (self):
        self.feat.set_NormalEstimation_Method_AVERAGE_DEPTH_CHANGE()
        # f = self.feat.compute(self.p)

        # check
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)


    def test_set_NormalEstimation_Method_SIMPLE_3D_GRADIENT (self):
        self.feat.set_NormalEstimation_Method_SIMPLE_3D_GRADIENT()
        # f = self.feat.compute(self.p)
        
        # check
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)


    def test_set_MaxDepthChange_Factor(self):
        param = 0.0
        self.feat.set_MaxDepthChange_Factor(param)
        # f = self.feat.compute(self.p)
        
        # check
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)

    def test_set_NormalSmoothingSize(self):
        param = 5.0 # default 10.0
        self.feat.set_NormalSmoothingSize(param)
        # f = self.feat.compute(self.p)
        # result point param?
        
        # check
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)



### MomentOfInertiaEstimation ###
class TestMomentOfInertiaEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.feat = pcl.MomentOfInertiaEstimation()


    def test_get_MomentOfInertia (self):
        param = self.feat.get_MomentOfInertia()


    def test_get_Eccentricity (self):
        param = self.feat.get_Eccentricity()


    def test_get_AABB (self):
        param = self.feat.get_AABB()


    def test_get_EigenValues (self):
        param = self.feat.get_EigenValues()


### NormalEstimation ###
class TestNormalEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.feat = pcl.NormalEstimation()


    def test_set_SearchMethod(self):
        kdTree = pcl.KdTree()
        self.feat.set_SearchMethod(kdTree)
        # f = self.feat.compute()
        
        # check
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)
        pass


    def test_set_RadiusSearch(self):
        param = 0.0
        self.feat.set_RadiusSearch(param)
        # f = self.feat.compute()
        
        # check
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)
        pass


    def test_set_KSearch (self):
        param = 0
        self.feat.set_KSearch (param)
        # self.feat.compute()
        
        # check
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)
        pass


    def test_compute (self):
        # f = self.feat.compute()
        
        # check
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)
        pass


### RangeImageBorderExtractor ###
class TestRangeImageBorderExtractor(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        # self.feat = pcl.RangeImageBorderExtractor()


    def test_set_RangeImage(self):
        # rangeImage = pcl.RangeImage()
        # self.feat.set_RangeImage(rangeImage)
        pass


    def test_ClearData (self):
        # self.feat.clearData ()
        pass


### VFHEstimation ###
class TestVFHEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        # self.feat = pcl.VFHEstimation()
    
    def test_set_SearchMethod(self):
        # kdTree = pcl.KdTree()
        # self.feat.set_SearchMethod(kdTree)
        # f = self.feat.compute()
        
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)
        pass
    
    def test_set_KSearch (self):
        param = 0.0
        # self.me.set_KSearch (param)
        # self.feat.compute()
        
        # check
        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)
        pass


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

