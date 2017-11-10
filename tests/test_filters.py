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


_data2 = np.array(
    [[0.0080142, 0.694695, -0.26015],
     [-0.342265, -0.446349, 0.214207],
     [0.173687, -0.84253, -0.400481],
     [-0.874475, 0.706127, -0.117635],
     [0.908514, -0.598159, 0.744714]], dtype=np.float32)


# Filter
### ApproximateVoxelGrid ###


class TestApproximateVoxelGrid(unittest.TestCase):
    def setUp(self):
        self.p = pcl.load(
            "tests" +
            os.path.sep +
            "tutorials" +
            os.path.sep +
            "table_scene_lms400.pcd")
        self.fil = self.p.make_ApproximateVoxelGrid()
        # self.fil = pcl.ApproximateVoxelGrid()
        # self.fil.set_InputCloud(self.p)

    def test_VoxelGrid(self):
        x = 0.01
        y = 0.01
        z = 0.01
        self.fil.set_leaf_size(x, y, z)
        result = self.fil.filter()
        # check
        self.assertTrue(result.size < self.p.size)
        # self.assertEqual(result.size, 719)


### ConditionalRemoval ###
# Appveyor NG
@attr('pcl_ver_0_4')
@attr('pcl_over_17')
class TestConditionalRemoval(unittest.TestCase):
    def setUp(self):
        # self.p = pcl.load("tests" + os.path.sep + "flydracyl.pcd")
        # self.p = pcl.PointCloud(_data)
        self.p = pcl.PointCloud(_data2)
        self.fil = self.p.make_ConditionalRemoval(pcl.ConditionAnd())

    # result
    # nan nan nan
    # -0.342265 -0.446349 0.214207
    # nan nan nan
    # nan nan nan
    # 0.908514 -0.598159 0.744714
    def test_Condition(self):
        range_cond = self.p.make_ConditionAnd()
        range_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.GT, 0.0)
        range_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.LT, 0.8)
        # build the filter
        self.fil.set_KeepOrganized(True)
        # apply filter
        cloud_filtered = self.fil.filter()

        # check
        expected = np.array([-0.342265, -0.446349, 0.214207])
        datas = np.around(cloud_filtered.to_array()[1].tolist(), decimals=6)
        self.assertEqual(datas.tolist(), expected.tolist())

        expected2 = np.array([0.908514, -0.598159, 0.744714])
        datas = np.around(cloud_filtered.to_array()[4].tolist(), decimals=6)
        self.assertEqual(datas.tolist(), expected2.tolist())


#     def set_KeepOrganized(self, flag):
#         self.fil.setKeepOrganized(flag)
#
#     def filter(self):
#         """
#         Apply the filter according to the previously set parameters and return
#         a new pointcloud
#         """
#         cdef PointCloud pc = PointCloud()
#         self.fil.filter(pc.thisptr()[0])
#         return pc


### ConditionAnd ###
class TestConditionAnd(unittest.TestCase):
    def setUp(self):
        self.p = pcl.load("tests" + os.path.sep + "flydracyl.pcd")

    def test_Tutorial(self):
        pass


### CropBox ###
# base : pcl/tests cpp source code[TEST (CropBox, Filters)]
class TestCropBox(unittest.TestCase):

    def setUp(self):
        input = pcl.PointCloud()
        points = np.zeros((9, 3), dtype=np.float32)
        points[0] = 0.0, 0.0, 0.0
        points[1] = 0.9, 0.9, 0.9
        points[2] = 0.9, 0.9, -0.9
        points[3] = 0.9, -0.9, 0.9
        points[4] = -0.9, 0.9, 0.9
        points[5] = 0.9, -0.9, -0.9
        points[6] = -0.9, -0.9, 0.9
        points[7] = -0.9, 0.9, -0.9
        points[8] = -0.9, -0.9, -0.9
        input.from_array(points)
        self.p = input

    def testException(self):
        self.assertRaises(TypeError, pcl.CropHull)

    def testCrop(self):
        cropBoxFilter = self.p.make_cropbox()
        # Cropbox slighlty bigger then bounding box of points
        cropBoxFilter.set_Min(-1.0, -1.0, -1.0, 1.0)
        cropBoxFilter.set_Max(1.0, 1.0, 1.0, 1.0)

        # Indices
        # vector<int> indices;
        # cropBoxFilter.filter(indices)

        # Cloud
        cloud_out = cropBoxFilter.filter()

        # Should contain all
        # self.assertEqual(indices.size, 9)
        self.assertEqual(cloud_out.size, 9)
        self.assertEqual(cloud_out.width, 9)
        self.assertEqual(cloud_out.height, 1)

        # IndicesConstPtr removed_indices;
        # removed_indices = cropBoxFilter.get_RemovedIndices ()
        cropBoxFilter.get_RemovedIndices()
        # self.assertEqual(removed_indices.size, 0)
        # self.assertEqual(lemn(removed_indices), 0)

        # Test setNegative
        cropBoxFilter.set_Negative(True)
        cloud_out_negative = cropBoxFilter.filter()
        # self.assertEqual(cloud_out_negative.size, 0)

        # cropBoxFilter.filter(indices)
        # self.assertEqual(indices.size, 0)

        cropBoxFilter.set_Negative(False)
        cloud_out = cropBoxFilter.filter()

        # Translate crop box up by 1
        tx = 0
        ty = 1
        tz = 0
        cropBoxFilter.set_Translation(tx, ty, tz)
        # indices = cropBoxFilter.filter()
        cloud_out = cropBoxFilter.filter()

        # self.assertEqual(indices.size, 5)
        self.assertEqual(cloud_out.size, 5)

        # removed_indices = cropBoxFilter.get_RemovedIndices ()
        cropBoxFilter.get_RemovedIndices()
        # self.assertEqual(removed_indices.size, 4)

        # Test setNegative
        cropBoxFilter.set_Negative(True)
        cloud_out_negative = cropBoxFilter.filter()
        # self.assertEqual(cloud_out_negative.size, 4)

        # indices = cropBoxFilter.filter()
        # self.assertEqual(indices.size, 4)

        cropBoxFilter.set_Negative(False)
        cloud_out = cropBoxFilter.filter()

        #  Rotate crop box up by 45
        # cropBoxFilter.setRotation (Eigen::Vector3f (0.0, 45.0f * float (M_PI) / 180.0, 0.0f))
        # cropBoxFilter.filter(indices)
        # cropBoxFilter.filter(cloud_out)
        rx = 0.0
        ry = 45.0 * (3.141592 / 180.0)
        rz = 0.0
        cropBoxFilter.set_Rotation(rx, ry, rz)
        # indices = cropBoxFilter.filter()
        cloud_out = cropBoxFilter.filter()

        # self.assertEqual(indices.size, 1)
        self.assertEqual(cloud_out.size, 1)
        self.assertEqual(cloud_out.width, 1)
        self.assertEqual(cloud_out.height, 1)

        # removed_indices = cropBoxFilter.get_RemovedIndices ()
        # self.assertEqual(removed_indices.size, 8)
        cropBoxFilter.get_RemovedIndices()

        #  Test setNegative
        cropBoxFilter.set_Negative(True)
        cloud_out_negative = cropBoxFilter.filter()
        # self.assertEqual(cloud_out_negative.size, 8)

        # indices = cropBoxFilter.filter()
        # self.assertEqual(indices.size, 8)

        cropBoxFilter.set_Negative(False)
        cloud_out = cropBoxFilter.filter()

        # // Rotate point cloud by -45
        # cropBoxFilter.set_Transform (getTransformation (0.0, 0.0, 0.0, 0.0, 0.0, -45.0f * float (M_PI) / 180.0f))
        # indices = cropBoxFilter.filter()
        # cloud_out = cropBoxFilter.filter()
        #
        # # self.assertEqual(indices.size, 3)
        # self.assertEqual(cloud_out.size, 3)
        # self.assertEqual(cloud_out.width, 3)
        # self.assertEqual(cloud_out.height, 1)
        ##

        # removed_indices = cropBoxFilter.get_RemovedIndices ()
        # self.assertEqual(removed_indices.size, 6)
        cropBoxFilter.get_RemovedIndices()

        # // Test setNegative
        cropBoxFilter.set_Negative(True)
        cloud_out_negative = cropBoxFilter.filter()
        # self.assertEqual(cloud_out_negative.size, 6)

        # indices = cropBoxFilter.filter()
        # self.assertEqual(indices.size, 6)

        cropBoxFilter.set_Negative(False)
        cloud_out = cropBoxFilter.filter()

        # Translate point cloud down by -1
        # # cropBoxFilter.setTransform (getTransformation(0, -1, 0, 0, 0, -45.0 * float (M_PI) / 180.0))
        # # cropBoxFilter.filter(indices)
        # cropBoxFilter.filter(cloud_out)
        #
        # # self.assertEqual(indices.size, 2)
        # self.assertEqual(cloud_out.size, 2)
        # self.assertEqual(cloud_out.width, 2)
        # self.assertEqual(cloud_out.height, 1)
        ##

        # removed_indices = cropBoxFilter.get_RemovedIndices ()
        # self.assertEqual(removed_indices.size, 7)

        # Test setNegative
        cropBoxFilter.set_Negative(True)
        cloud_out_negative = cropBoxFilter.filter()
        # self.assertEqual(cloud_out_negative.size, 7)

        # indices = cropBoxFilter.filter()
        # self.assertEqual(indices.size, 7)

        cropBoxFilter.set_Negative(False)
        cloud_out = cropBoxFilter.filter()

        # // Remove point cloud rotation
        # cropBoxFilter.set_Transform (getTransformation(0, -1, 0, 0, 0, 0))
        # indices = cropBoxFilter.filter()
        # cloud_out = cropBoxFilter.filter()

        # self.assertEqual(indices.size, 0)
        # self.assertEqual(cloud_out.size, 0)
        # self.assertEqual(cloud_out.width, 0)
        # self.assertEqual(cloud_out.height, 1)

        # removed_indices = cropBoxFilter.get_RemovedIndices ()
        # self.assertEqual(removed_indices.size, 9)

        # Test setNegative
        cropBoxFilter.set_Negative(True)
        cloud_out_negative = cropBoxFilter.filter()
        # self.assertEqual(cloud_out_negative.size, 9)

        # indices = cropBoxFilter.filter()
        # self.assertEqual(indices.size, 9)

        # PCLPointCloud2
        # // ------------------------------------------------------------------
        # Create cloud with center point and corner points
        # PCLPointCloud2::Ptr input2 (new PCLPointCloud2)
        # pcl::toPCLPointCloud2 (*input, *input2)
        #
        # Test the PointCloud<PointT> method
        # CropBox<PCLPointCloud2> cropBoxFilter2(true)
        # cropBoxFilter2.setInputCloud (input2)
        #
        # Cropbox slighlty bigger then bounding box of points
        # cropBoxFilter2.setMin (min_pt)
        # cropBoxFilter2.setMax (max_pt)
        #
        # Indices
        # vector<int> indices2;
        # cropBoxFilter2.filter (indices2)
        #
        # Cloud
        # PCLPointCloud2 cloud_out2;
        # cropBoxFilter2.filter (cloud_out2)
        #
        # // Should contain all
        # self.assertEqual(indices2.size, 9)
        # self.assertEqual(indices2.size, int (cloud_out2.width * cloud_out2.height))
        #
        # IndicesConstPtr removed_indices2;
        # removed_indices2 = cropBoxFilter2.get_RemovedIndices ()
        # self.assertEqual(removed_indices2.size, 0)
        #
        # // Test setNegative
        # PCLPointCloud2 cloud_out2_negative;
        # cropBoxFilter2.setNegative (true)
        # cropBoxFilter2.filter (cloud_out2_negative)
        # self.assertEqual(cloud_out2_negative.width), 0)
        #
        # cropBoxFilter2.filter (indices2)
        # self.assertEqual(indices2.size, 0)
        #
        # cropBoxFilter2.setNegative (false)
        # cropBoxFilter2.filter (cloud_out2)
        #
        # // Translate crop box up by 1
        # cropBoxFilter2.setTranslation (Eigen::Vector3f(0, 1, 0))
        # cropBoxFilter2.filter (indices2)
        # cropBoxFilter2.filter (cloud_out2)
        #
        # self.assertEqual(indices2.size, 5)
        # self.assertEqual(indices2.size, int (cloud_out2.width * cloud_out2.height))
        #
        # removed_indices2 = cropBoxFilter2.get_RemovedIndices ()
        # self.assertEqual(removed_indices2.size, 4)
        #
        # // Test setNegative
        # cropBoxFilter2.setNegative (true)
        # cropBoxFilter2.filter (cloud_out2_negative)
        # self.assertEqual(cloud_out2_negative.width), 4)
        #
        # cropBoxFilter2.filter (indices2)
        # self.assertEqual(indices2.size, 4)
        #
        # cropBoxFilter2.setNegative (false)
        # cropBoxFilter2.filter (cloud_out2)
        #
        # // Rotate crop box up by 45
        # cropBoxFilter2.setRotation (Eigen::Vector3f (0.0, 45.0f * float (M_PI) / 180.0, 0.0f))
        # cropBoxFilter2.filter (indices2)
        # cropBoxFilter2.filter (cloud_out2)
        #
        # self.assertEqual(indices2.size, 1)
        # self.assertEqual(indices2.size, int (cloud_out2.width * cloud_out2.height))
        #
        # // Rotate point cloud by -45
        # cropBoxFilter2.setTransform (getTransformation (0.0, 0.0, 0.0, 0.0, 0.0, -45.0f * float (M_PI) / 180.0f))
        # cropBoxFilter2.filter (indices2)
        # cropBoxFilter2.filter (cloud_out2)
        #
        # self.assertEqual(indices2.size, 3)
        # self.assertEqual(cloud_out2.width * cloud_out2.height), 3)
        #
        # removed_indices2 = cropBoxFilter2.get_RemovedIndices ()
        # self.assertEqual(removed_indices2.size, 6)
        #
        # // Test setNegative
        # cropBoxFilter2.setNegative (true)
        # cropBoxFilter2.filter (cloud_out2_negative)
        # self.assertEqual(cloud_out2_negative.width), 6)
        #
        # cropBoxFilter2.filter (indices2)
        # self.assertEqual(indices2.size, 6)
        #
        # cropBoxFilter2.setNegative (false)
        # cropBoxFilter2.filter (cloud_out2)
        #
        # // Translate point cloud down by -1
        # cropBoxFilter2.setTransform (getTransformation (0.0, -1.0, 0.0, 0.0, 0.0, -45.0f * float (M_PI) / 180.0f))
        # cropBoxFilter2.filter (indices2)
        # cropBoxFilter2.filter (cloud_out2)
        #
        # self.assertEqual(indices2.size, 2)
        # self.assertEqual(cloud_out2.width * cloud_out2.height), 2)
        #
        # removed_indices2 = cropBoxFilter2.get_RemovedIndices ()
        # self.assertEqual(removed_indices2.size, 7)
        #
        # // Test setNegative
        # cropBoxFilter2.setNegative (true)
        # cropBoxFilter2.filter (cloud_out2_negative)
        # self.assertEqual(cloud_out2_negative.width), 7)
        #
        # cropBoxFilter2.filter (indices2)
        # self.assertEqual(indices2.size, 7)
        #
        # cropBoxFilter2.setNegative (false)
        # cropBoxFilter2.filter (cloud_out2)
        #
        # // Remove point cloud rotation
        # cropBoxFilter2.setTransform (getTransformation(0, -1, 0, 0, 0, 0))
        # cropBoxFilter2.filter (indices2)
        # cropBoxFilter2.filter (cloud_out2)
        #
        # self.assertEqual(indices2.size, 0)
        # self.assertEqual(cloud_out2.width * cloud_out2.height), 0)
        #
        # removed_indices2 = cropBoxFilter2.get_RemovedIndices ()
        # self.assertEqual(removed_indices2.size, 9)
        #
        # // Test setNegative
        # cropBoxFilter2.setNegative (true)
        # cropBoxFilter2.filter (cloud_out2_negative)
        # self.assertEqual(cloud_out2_negative.width), 9)
        #
        # cropBoxFilter2.filter (indices2)
        # self.assertEqual(indices2.size, 9)
        #
        # cropBoxFilter2.setNegative (false)
        # cropBoxFilter2.filter (cloud_out2)


### CropHull ###
class TestCropHull(unittest.TestCase):

    def setUp(self):
        self.pc = pcl.load(
            "tests" +
            os.path.sep +
            "tutorials" +
            os.path.sep +
            "table_scene_mug_stereo_textured.pcd")

    def testException(self):
        self.assertRaises(TypeError, pcl.CropHull)

    def testCropHull(self):
        filterCloud = pcl.PointCloud()
        vt = pcl.Vertices()
#         # // inside point
#         # cloud->push_back(pcl::PointXYZ(M_PI * 0.3, M_PI * 0.3, 0))
#         # // hull points
#         # cloud->push_back(pcl::PointXYZ(0,0,0))
#         # cloud->push_back(pcl::PointXYZ(M_PI,0,0))
#         # cloud->push_back(pcl::PointXYZ(M_PI,M_PI*0.5,0))
#         # cloud->push_back(pcl::PointXYZ(0,M_PI*0.5,0))
#         # cloud->push_back(pcl::PointXYZ(0,0,0))
#         # // outside point
#         # cloud->push_back(pcl::PointXYZ(-M_PI * 0.3, -M_PI * 0.3, 0))
#         points_2 = np.array([
#                         [1 * 0.3, 1 * 0.3, 0],
#                         [0, 0, 0],
#                         [1, 0, 0],
#                         [1, 1 * 0.5, 0],
#                         [0, 1 * 0.5, 0],
#                         [0, 0, 0],
#                         # [-1 * 0.3 , -1 * 0.3, 0]
#                     ], dtype=np.float32)
#         filterCloud.from_array(points_2)
#         # print(filterCloud)
#
#         vertices_point_1 = np.array([1, 2, 3, 4, 5], dtype=np.int)
#         vt.from_array(vertices_point_1)
#         # print(vt)
#         # vt.vertices.push_back(1)
#         # vt.vertices.push_back(2)
#         # vt.vertices.push_back(3)
#         # vt.vertices.push_back(4)
#         # vt.vertices.push_back(5)
#         # vertices = vector[pcl.Vertices]
#         # vertices.push_back(vt)
#
#         outputCloud = pcl.PointCloud()
#         # crophull = pcl.CropHull()
#         # crophull.setInputCloud(self.pc)
#         crophull = self.pc.make_crophull()
#         # crophull.setHullIndices(vertices)
#         # crophull.setHullIndices(vt)
#         # crophull.setHullCloud(filterCloud)
#         # crophull.setDim(2)
#         # crophull.setCropOutside(false)
#         crophull.SetParameter(filterCloud, vt)
#         # indices = vector[int]
#         # cropHull.filter(indices)
#         # outputCloud = cropHull.filter()
#         # print("before: " + outputCloud)
#         crophull.filter(outputCloud)
#         # print(outputCloud)


### FieldComparison ###
class TestFieldComparison(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load("tests/table_scene_mug_stereo_textured_noplane.pcd")
        compare = CompareOp2
        thresh = 1.0
        self.fil = pcl.FieldComparison(compare, thresh)


### PassThroughFilter ###
class TestPassthroughFilter(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load("tests/table_scene_mug_stereo_textured_noplane.pcd")

    def testFilter(self):
        fil = self.p.make_passthrough_filter()
        fil.set_filter_field_name("z")
        fil.set_filter_limits(0, 0.75)
        c = fil.filter()
        self.assertTrue(c.size < self.p.size)
        self.assertEqual(c.size, 7751)

    def testFilterBoth(self):
        total = self.p.size
        fil = self.p.make_passthrough_filter()
        fil.set_filter_field_name("z")
        fil.set_filter_limits(0, 0.75)
        front = fil.filter().size
        fil.set_filter_limits(0.75, 100)
        back = fil.filter().size
        self.assertEqual(total, front + back)


### ProjectInliers ###
class TestProjectInliers(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load("tests/table_scene_mug_stereo_textured_noplane.pcd")
        self.fil = self.p.make_ProjectInliers()
        # self.fil = pcl.ProjectInliers()
        # self.fil.set_InputCloud(self.p)

    def test_model_type(self):
        # param1
        m = pcl.SACMODEL_PLANE
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_PLANE)

        # param2
        m = pcl.SACMODEL_LINE
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_LINE)

        # param3
        m = pcl.SACMODEL_CIRCLE2D
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_CIRCLE2D)

        # param4
        m = pcl.SACMODEL_CIRCLE3D
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_CIRCLE3D)

        # param5
        m = pcl.SACMODEL_SPHERE
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_SPHERE)

        # param6
        m = pcl.SACMODEL_CYLINDER
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_CYLINDER)

        # param7
        m = pcl.SACMODEL_CONE
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_CONE)

        # param8
        m = pcl.SACMODEL_TORUS
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_TORUS)

        # param9
        m = pcl.SACMODEL_PARALLEL_LINE
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_PARALLEL_LINE)

        # param10
        m = pcl.SACMODEL_PERPENDICULAR_PLANE
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_PERPENDICULAR_PLANE)

        # param11
        m = pcl.SACMODEL_PARALLEL_LINES
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_PARALLEL_LINES)

        # param12
        m = pcl.SACMODEL_NORMAL_PLANE
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_NORMAL_PLANE)

        # param13
        m = pcl.SACMODEL_NORMAL_SPHERE
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_NORMAL_SPHERE)

        # param14
        m = pcl.SACMODEL_REGISTRATION
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_REGISTRATION)

        # param15
        m = pcl.SACMODEL_PARALLEL_PLANE
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_PARALLEL_PLANE)

        # param16
        m = pcl.SACMODEL_NORMAL_PARALLEL_PLANE
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_NORMAL_PARALLEL_PLANE)

        # param17
        m = pcl.SACMODEL_STICK
        self.fil.set_model_type(m)
        result_param = self.fil.get_model_type()
        self.assertEqual(result_param, pcl.SACMODEL_STICK)
        pass

    def test_copy_all_data(self):
        self.fil.set_copy_all_data(True)
        datas = self.fil.get_copy_all_data()
        # result
        self.assertEqual(datas, True)

        self.fil.set_copy_all_data(False)
        datas = self.fil.get_copy_all_data()
        # result2
        self.assertEqual(datas, False)


### RadiusOutlierRemoval ###
class TestRadiusOutlierRemoval(unittest.TestCase):

    def setUp(self):
        # self.p = pcl.load("tests/table_scene_mug_stereo_textured_noplane.pcd")
        self.p = pcl.PointCloud(_data2)
        print(self.p.size)
        self.fil = self.p.make_RadiusOutlierRemoval()
        # self.fil = pcl.RadiusOutlierRemoval()
        # self.fil.set_InputCloud(self.p)

    def test_radius_seach(self):
        radius = 15.8
        self.fil.set_radius_search(radius)
        result = self.fil.get_radius_search()
        self.assertEqual(radius, result)

        min_pts = 2
        self.fil.set_MinNeighborsInRadius(2)
        result = self.fil.get_MinNeighborsInRadius()
        self.assertEqual(min_pts, result)

        result_point = self.fil.filter()

        # check
        # new instance is returned
        # self.assertNotEqual(self.p, result)
        # filter retains the same number of points
        # self.assertNotEqual(result_point.size, 0)
        # self.assertNotEqual(self.p.size, result_point.size)


### StatisticalOutlierRemovalFilter ###
class TestStatisticalOutlierRemovalFilter(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load(
            "tests" +
            os.path.sep +
            "table_scene_mug_stereo_textured_noplane.pcd")
        self.fil = self.p.make_statistical_outlier_filter()
        # self.fil = pcl.StatisticalOutlierRemovalFilter()
        # self.fil.set_InputCloud(self.p)

    def _tpos(self, c):
        self.assertEqual(c.size, 22745)
        self.assertEqual(c.width, 22745)
        self.assertEqual(c.height, 1)
        self.assertTrue(c.is_dense)

    def _tneg(self, c):
        self.assertEqual(c.size, 1015)
        self.assertEqual(c.width, 1015)
        self.assertEqual(c.height, 1)
        self.assertTrue(c.is_dense)

    def testFilterPos(self):
        fil = self.p.make_statistical_outlier_filter()
        fil.set_mean_k(50)
        self.assertEqual(fil.mean_k, 50)
        fil.set_std_dev_mul_thresh(1.0)
        self.assertEqual(fil.stddev_mul_thresh, 1.0)
        c = fil.filter()
        self._tpos(c)

    def testFilterNeg(self):
        fil = self.p.make_statistical_outlier_filter()
        fil.set_mean_k(50)
        fil.set_std_dev_mul_thresh(1.0)
        self.assertEqual(fil.negative, False)
        fil.set_negative(True)
        self.assertEqual(fil.negative, True)
        c = fil.filter()
        self._tneg(c)

    def testFilterPosNeg(self):
        fil = self.p.make_statistical_outlier_filter()
        fil.set_mean_k(50)
        fil.set_std_dev_mul_thresh(1.0)
        c = fil.filter()
        self._tpos(c)
        fil.set_negative(True)
        c = fil.filter()
        self._tneg(c)


### VoxelGridFilter ###
class TestVoxelGridFilter(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load(
            "tests" +
            os.path.sep +
            "table_scene_mug_stereo_textured_noplane.pcd")
        self.fil = self.p.make_voxel_grid_filter()
        # self.fil = pcl.VoxelGridFilter()
        # self.fil.set_InputCloud(self.p)

    def testFilter(self):
        self.fil.set_leaf_size(0.01, 0.01, 0.01)
        c = self.fil.filter()
        self.assertTrue(c.size < self.p.size)
        self.assertEqual(c.size, 719)


### Official Test Base ###
p_65558 = np.array([-0.058448, -0.189095, 0.723415], dtype=np.float32)
p_84737 = np.array([-0.088929, -0.152957, 0.746095], dtype=np.float32)
p_57966 = np.array([0.123646, -0.397528, 1.393187], dtype=np.float32)
p_39543 = np.array([0.560287, -0.545020, 1.602833], dtype=np.float32)
p_17766 = np.array([0.557854, -0.711976, 1.762013], dtype=np.float32)
p_70202 = np.array([0.150500, -0.160329, 0.646596], dtype=np.float32)
p_102219 = np.array([0.175637, -0.101353, 0.661631], dtype=np.float32)
p_81765 = np.array([0.223189, -0.151714, 0.708332], dtype=np.float32)

# class TESTFastBilateralFilter(unittest.TestCase):
#     def setUp(self):
#         self.p = pcl.load("tests" + os.path.sep + "milk_cartoon_all_small_clorox.pcd")
#
#     def testFastBilateralFilter(self):
#         fbf = pcl.FastBilateralFilter()
#         fbf.setInputCloud(cloud)
#         fbf.setSigmaS (5)
#         fbf.setSigmaR (0.03f)
#         cloud_filtered = fbf.filter()
#         # for (size_t dim = 0; dim < 3; ++dim):
#         for dim range(0:3):
#             EXPECT_NEAR (p_84737[dim],  cloud_filtered[84737][dim], 1e-3)
#             EXPECT_NEAR (p_57966[dim],  cloud_filtered[57966][dim], 1e-3)
#             EXPECT_NEAR (p_39543[dim],  cloud_filtered[39543][dim], 1e-3)
#             EXPECT_NEAR (p_17766[dim],  cloud_filtered[17766][dim], 1e-3)
#             EXPECT_NEAR (p_70202[dim],  cloud_filtered[70202][dim], 1e-3)
#             EXPECT_NEAR (p_102219[dim], cloud_filtered[102219][dim], 1e-3)
#             EXPECT_NEAR (p_81765[dim],  cloud_filtered[81765][dim], 1e-3)
#         pass


# class TESTFastBilateralFilterOMP(unittest.TestCase):
#
#     def setUp(self):
#         self.p = pcl.load("tests" + os.path.sep + "milk_cartoon_all_small_clorox.pcd")
#
#         sigma_s = [2.341,  5.2342, 10.29380]
#         sigma_r = [0.0123, 0.023,  0.0345]
#         for (size_t i = 0; i < 3; i++)
#             FastBilateralFilter<PointXYZ> fbf;
#             fbf.setInputCloud (cloud);
#             fbf.setSigmaS (sigma_s[i]);
#             fbf.setSigmaR (sigma_r[i]);
#             PointCloud<PointXYZ>::Ptr cloud_filtered (new PointCloud<PointXYZ> ());
#             fbf.filter (*cloud_filtered);
#
#             FastBilateralFilterOMP<PointXYZ> fbf_omp (0);
#             fbf_omp.setInputCloud (cloud);
#             fbf_omp.setSigmaS (sigma_s[i]);
#             fbf_omp.setSigmaR (sigma_r[i]);
#     PointCloud<PointXYZ>::Ptr cloud_filtered_omp (new PointCloud<PointXYZ> ());
#     fbf_omp.filter (*cloud_filtered_omp);
#     PCL_INFO ("[FastBilateralFilterOMP] filtering took %f ms\n", tt.toc ());
#
#
#     EXPECT_EQ (cloud_filtered_omp->points.size (), cloud_filtered->points.size ());
#     for (size_t j = 0; j < cloud_filtered_omp->size (); ++j)
#     {
#       if (pcl_isnan (cloud_filtered_omp->at (j).x))
#         EXPECT_TRUE (pcl_isnan (cloud_filtered->at (j).x));
#       else
#       {
#         EXPECT_NEAR (cloud_filtered_omp->at (j).x, cloud_filtered->at (j).x, 1e-3);
#         EXPECT_NEAR (cloud_filtered_omp->at (j).y, cloud_filtered->at (j).y, 1e-3);
#         EXPECT_NEAR (cloud_filtered_omp->at (j).z, cloud_filtered->at (j).z, 1e-3);
#       }
#     }


def suite():
    suite = unittest.TestSuite()

    # Filter
    suite.addTests(unittest.makeSuite(TestApproximateVoxelGrid))
    suite.addTests(unittest.makeSuite(TestConditionalRemoval))
    # suite.addTests(unittest.makeSuite(TestConditionAnd))
    suite.addTests(unittest.makeSuite(TestCropBox))
    suite.addTests(unittest.makeSuite(TestCropHull))
    suite.addTests(unittest.makeSuite(TestFieldComparison))
    suite.addTests(unittest.makeSuite(TestPassthroughFilter))
    suite.addTests(unittest.makeSuite(TestProjectInliers))
    # suite.addTests(unittest.makeSuite(TestRadiusOutlierRemoval))
    suite.addTests(unittest.makeSuite(TestStatisticalOutlierRemovalFilter))
    suite.addTests(unittest.makeSuite(TestVoxelGridFilter))

    # PointCloudLibrary Official Base Test?
    # suite.addTests(unittest.makeSuite(TestFastBilateralFilter))

    return suite


if __name__ == '__main__':
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
