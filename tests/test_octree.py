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


from nose.plugins.attrib import attr


# class TestOctreePointCloud(unittest.TestCase):
#
#     def setUp(self):
#         self.p = pcl.load("tests" + os.path.sep + "table_scene_mug_stereo_textured_noplane.pcd")
#         self.octree = pcl.OctreePointCloud(0.1)
#
#     def testLoad(self):
#         self.octree.set_input_cloud(self.p)
#         self.octree.define_bounding_box()
#         self.octree.add_points_from_input_cloud()
#         good_point = [0.035296999, -0.074322999, 1.2074]
#         rs = self.octree.is_voxel_occupied_at_point(good_point)
#         self.assertTrue(rs)
#         bad_point = [0.5, 0.5, 0.5]
#         rs = self.octree.is_voxel_occupied_at_point(bad_point)
#         self.assertFalse(rs)
#         voxels_len = 44
#         self.assertEqual(len(self.octree.get_occupied_voxel_centers()), voxels_len)
#         self.octree.delete_voxel_at_point(good_point)
#         self.assertEqual(len(self.octree.get_occupied_voxel_centers()), voxels_len - 1)


class TestOctreePointCloudSearch(unittest.TestCase):

    def setUp(self):
        self.octree = pcl.OctreePointCloudSearch(0.1)
        self.p = pcl.load(
            "tests" +
            os.path.sep +
            "tutorials" +
            os.path.sep +
            "table_scene_mug_stereo_textured_noplane.pcd")
        self.octree.set_input_cloud(self.p)
        self.octree.define_bounding_box()
        self.octree.add_points_from_input_cloud()

    def testConstructor(self):
        self.assertRaises(ValueError, pcl.OctreePointCloudSearch, 0.)

    def testRadiusSearch(self):
        good_point = (0.035296999, -0.074322999, 1.2074)
        rs = self.octree.radius_search(good_point, 0.5, 1)
        self.assertEqual(len(rs[0]), 1)
        self.assertEqual(len(rs[1]), 1)
        rs = self.octree.radius_search(good_point, 0.5)
        self.assertEqual(len(rs[0]), 19730)
        self.assertEqual(len(rs[1]), 19730)


class TestOctreePointCloudChangeDetector(unittest.TestCase):

    def setUp(self):
        self.octree = pcl.OctreePointCloudSearch(0.1)
        self.p = pcl.load(
            "tests" +
            os.path.sep +
            "tutorials" +
            os.path.sep +
            "table_scene_mug_stereo_textured_noplane.pcd")
        self.octree.set_input_cloud(self.p)
        self.octree.define_bounding_box()
        self.octree.add_points_from_input_cloud()

    def testConstructor(self):
        self.assertRaises(ValueError, pcl.OctreePointCloudChangeDetector, 0.)

    def testRadiusSearch(self):
        good_point = (0.035296999, -0.074322999, 1.2074)
        rs = self.octree.radius_search(good_point, 0.5, 1)
        self.assertEqual(len(rs[0]), 1)
        self.assertEqual(len(rs[1]), 1)
        rs = self.octree.radius_search(good_point, 0.5)
        self.assertEqual(len(rs[0]), 19730)
        self.assertEqual(len(rs[1]), 19730)


def suite():
    suite = unittest.TestSuite()
    # octree
    # base class
    # suite.addTests(unittest.makeSuite(TestOctreePointCloud))
    suite.addTests(unittest.makeSuite(TestOctreePointCloudSearch))
    suite.addTests(unittest.makeSuite(TestOctreePointCloudChangeDetector))
    return suite


if __name__ == '__main__':
    # unittest.main()
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
