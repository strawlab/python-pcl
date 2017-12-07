import os.path
import pickle
import shutil
import tempfile
import unittest

import pcl
import numpy as np

from nose.plugins.attrib import attr


_data = [(i, 2 * i, 3 * i + 0.2) for i in range(5)]
_DATA = """0.0, 0.0, 0.2;
           1.0, 2.0, 3.2;
           2.0, 4.0, 6.2;
           3.0, 6.0, 9.2;
           4.0, 8.0, 12.2"""


# features
### DifferenceOfNormalsEstimation ###
@attr('pcl_ver_0_4')
class TestDifferenceOfNormalsEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        # self.feat = pcl.DifferenceOfNormalsEstimation()

    def testException(self):
        # self.assertRaises(TypeError, pcl.DifferenceOfNormalsEstimation)
        pass


### IntegralImageNormalEstimation ###
@attr('pcl_ver_0_4')
class TestIntegralImageNormalEstimation(unittest.TestCase):
    def setUp(self):
        # self.p = pcl.PointCloud(_data)
        self.p = pcl.load(
            "tests" +
            os.path.sep +
            "tutorials" +
            os.path.sep +
            "table_scene_mug_stereo_textured.pcd")
        # self.feat = pcl.IntegralImageNormalEstimation(self.p)
        self.feat = self.p.make_IntegralImageNormalEstimation()

    # base : normal_estimation_using_integral_images.cpp
    # @unittest.skip
    def test_Tutorial(self):
        # before chack
        self.assertEqual(self.p.size, 307200)
        self.assertEqual(self.p.width, 640)
        self.assertEqual(self.p.height, 480)

        self.feat.set_NormalEstimation_Method_AVERAGE_3D_GRADIENT()
        self.feat.set_MaxDepthChange_Factor(0.02)
        self.feat.set_NormalSmoothingSize(10.0)
        # height = 1 pointdata set ng
        normals = self.feat.compute()
        # print(normals)

        # check - normals data
        # 1. return type
        # self.assertRaises(normals, pcl.PointCloud_Normal)
        # 2. point size
        # self.assertEqual(self.p.size, normals.size)

        # 3. same Tutorial data
        # size ->
        # self.assertEqual(self.p.size, normals.size)
        # for i in range(0, normals.size):
        #   print ('normal_x: '  + str(normals[i][0]) + ', normal_y : ' + str(normals[i][1])  + ', normal_z : ' + str(normals[i][2]))
        # print('end')


#     def test_set_NormalEstimation_Method_AVERAGE_3D_GRADIENT(self):
#         self.feat.set_NormalEstimation_Method_AVERAGE_3D_GRADIENT()
#         self.feat.setMaxDepthChangeFactor(0.02f)
#         self.feat.setNormalSmoothingSize(10.0)
#         f = self.feat.compute(self.p)
#
#         # check
#         # new instance is returned
#         # self.assertNotEqual(self.p, f)
#         # filter retains the same number of points
#         # self.assertEqual(self.p.size, f.size)
#
#
#     def test_set_NormalEstimation_Method_COVARIANCE_MATRIX(self):
#         self.feat.set_NormalEstimation_Method_COVARIANCE_MATRIX()
#         # f = self.feat.compute(self.p)
#
#         # check
#         # new instance is returned
#         # self.assertNotEqual(self.p, f)
#         # filter retains the same number of points
#         # self.assertEqual(self.p.size, f.size)
#
#     def test_set_NormalEstimation_Method_AVERAGE_DEPTH_CHANGE(self):
#         self.feat.set_NormalEstimation_Method_AVERAGE_DEPTH_CHANGE()
#         # f = self.feat.compute(self.p)
#
#         # check
#         # new instance is returned
#         # self.assertNotEqual(self.p, f)
#         # filter retains the same number of points
#         # self.assertEqual(self.p.size, f.size)
#
#     def test_set_NormalEstimation_Method_SIMPLE_3D_GRADIENT(self):
#         self.feat.set_NormalEstimation_Method_SIMPLE_3D_GRADIENT()
#         # f = self.feat.compute(self.p)
#
#         # check
#         # new instance is returned
#         # self.assertNotEqual(self.p, f)
#         # filter retains the same number of points
#         # self.assertEqual(self.p.size, f.size)
#
#     #
#     def test_set_MaxDepthChange_Factor(self):
#         param = 0.0
#         self.feat.set_MaxDepthChange_Factor(param)
#         # f = self.feat.compute(self.p)
#
#         # check
#         # new instance is returned
#         # self.assertNotEqual(self.p, f)
#         # filter retains the same number of points
#         # self.assertEqual(self.p.size, f.size)
#
#     def test_set_NormalSmoothingSize(self):
#         param = 5.0  # default 10.0
#         self.feat.set_NormalSmoothingSize(param)
#         # f = self.feat.compute(self.p)
#         # result point param?
#
#         # check
#         # new instance is returned
#         # self.assertNotEqual(self.p, f)
#         # filter retains the same number of points
#         # self.assertEqual(self.p.size, f.size)


### MomentOfInertiaEstimation ###
@attr('pcl_over_18')
class TestMomentOfInertiaEstimation(unittest.TestCase):
    def setUp(self):
        # self.p = pcl.PointCloud(_data)
        self.p = pcl.load(
            "tests" +
            os.path.sep +
            "tutorials" +
            os.path.sep +
            "lamppost.pcd")
        # 1.8.0
        # self.feat = pcl.MomentOfInertiaEstimation()
        self.feat = self.p.make_MomentOfInertiaEstimation()

    def test_Tutorials(self):
        self.feat.compute()

        # Get Parameters
        moment_of_inertia = self.feat.get_MomentOfInertia()
        eccentricity = self.feat.get_Eccentricity()
        [min_point_AABB, max_point_AABB] = self.feat.get_AABB()
        # [min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB] = self.feat.get_OBB ()
        [major_value, middle_value, minor_value] = self.feat.get_EigenValues()
        [major_vector, middle_vector, minor_vector] = self.feat.get_EigenVectors()
        mass_center = self.feat.get_MassCenter()

        # check parameter
        # printf("%f %f %f.\n", mass_center (0), mass_center (1), mass_center (2));
        # -10.104160 0.074005 -2.144748.
        # printf("%f %f %f.\n", major_vector (0), major_vector (1), major_vector (2));
        # 0.164287 -0.044990 -0.985386.
        # printf("%f %f %f.\n", middle_vector (0), middle_vector (1), middle_vector (2));
        # 0.920083 -0.353143 0.169523.
        # printf("%f %f %f.\n", minor_vector (0), minor_vector (1), minor_vector (2));
        # -0.355608 -0.934488 -0.016622.

        # expected = [-10.104160, 0.074005, -2.144748]
        expected = np.array([-10.104160, 0.074005, -2.144748])
        # print(str(mass_center[0][0].dtype))
        datas = np.around(mass_center[0].tolist(), decimals=6)
        # print("test : " + str(datas))
        self.assertEqual(datas.tolist(), expected.tolist())
        # self.assertEqual(datas, expected)

        expected2 = np.array([0.164287, -0.044990, -0.985386])
        datas = np.around(major_vector[0].tolist(), decimals=6)
        self.assertEqual(datas.tolist(), expected2.tolist())

        expected3 = np.array([0.920083, -0.353143, 0.169523])
        datas = np.around(middle_vector[0].tolist(), decimals=6)
        self.assertEqual(datas.tolist(), expected3.tolist())

        expected4 = np.array([-0.355608, -0.934488, -0.016622])
        datas = np.around(minor_vector[0].tolist(), decimals=6)
        self.assertEqual(datas.tolist(), expected4.tolist())


#     def test_get_MomentOfInertia(self):
#         param = self.feat.get_MomentOfInertia()
#
#     def test_get_Eccentricity(self):
#         param = self.feat.get_Eccentricity()
#
#     def test_get_AABB(self):
#         param = self.feat.get_AABB()
#
#     def test_get_EigenValues(self):
#         param = self.feat.get_EigenValues()


### NormalEstimation ###
class TestNormalEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        # self.feat = pcl.NormalEstimation()
        # self.feat.setInputCloud(selp.p)
        self.feat = self.p.make_NormalEstimation()

    def test_Tutorials_Radius(self):
        self.feat.set_RadiusSearch(0.03)
        normals = self.feat.compute()

        # check - normals data
        # 1. return type
        # self.assertEqual(type(normals), type(pcl.PointCloud_Normal))
        # 2. point size
        self.assertEqual(self.p.size, normals.size)

        # 3. same Tutorial data
        # size ->
        # self.assertEqual(self.p.size, normals.size)
        # for i in range(0, normals.size):
        #   print ('normal_x: '  + str(normals[i][0]) + ', normal_y : ' + str(normals[i][1])  + ', normal_z : ' + str(normals[i][2]))
        # print('end')

    def test_Tutorials_KSearch(self):
        tree = self.p.make_kdtree()
        self.feat.set_SearchMethod(tree)
        self.feat.set_KSearch(10)
        normals = self.feat.compute()
        # check - normals data
        # 1. return type
        # self.assertEqual(type(normals), type(pcl.PointCloud_Normal))
        # 2. point size is same
        self.assertEqual(self.p.size, normals.size)

        # 3. same Tutorial data
        # size ->
        # self.assertEqual(self.p.size, normals.size)
        # for i in range(0, normals.size):
        #   print ('normal_x: '  + str(normals[i][0]) + ', normal_y : ' + str(normals[i][1])  + ', normal_z : ' + str(normals[i][2]))
        # print('end')


### RangeImageBorderExtractor ###
@attr('pcl_ver_0_4')
class TestRangeImageBorderExtractor(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        # self.feat = pcl.RangeImageBorderExtractor()

    def test_set_RangeImage(self):
        # rangeImage = pcl.RangeImage()
        # self.feat.set_RangeImage(rangeImage)
        pass

    def test_ClearData(self):
        # self.feat.clearData ()
        pass


### VFHEstimation ###
class TestVFHEstimation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        # self.feat = pcl.VFHEstimation()
        # self.feat.setInputCloud(self.p)
        self.feat = self.p.make_VFHEstimation()

    def test_set_SearchMethod(self):
        # kdTree = pcl.KdTree()
        # self.feat.set_SearchMethod(kdTree)
        # f = self.feat.compute()

        # new instance is returned
        # self.assertNotEqual(self.p, f)
        # filter retains the same number of points
        # self.assertEqual(self.p.size, f.size)
        pass

    def test_set_KSearch(self):
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
    # compute - exception
    # suite.addTests(unittest.makeSuite(TestIntegralImageNormalEstimation))
    suite.addTests(unittest.makeSuite(TestMomentOfInertiaEstimation))
    suite.addTests(unittest.makeSuite(TestNormalEstimation))
    suite.addTests(unittest.makeSuite(TestVFHEstimation))
    # no add pxiInclude
    # suite.addTests(unittest.makeSuite(TestDifferenceOfNormalsEstimation))
    # suite.addTests(unittest.makeSuite(TestRangeImageBorderExtractor))
    return suite


if __name__ == '__main__':
    # unittest.main()
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
