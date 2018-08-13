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

# segmentation

### ConditionalEuclideanClustering(1.7.2/1.8.0) ###


@attr('pcl_over_17')
@attr('pcl_ver_0_4')
class TestConditionalEuclideanClustering(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.segment = pcl.ConditionalEuclideanClustering()

    def testInstance(self):
        assertIsInstance(type(self.segment), "ConditionalEuclideanClustering")


### EuclideanClusterExtraction ###
class TestEuclideanClusterExtraction(unittest.TestCase):
    def setUp(self):
        # self.p = pcl.PointCloud(_data)
        self.p = pcl.load(
            './examples/pcldata/tutorials/table_scene_lms400.pcd')
        # self.segment = self.p

    def testTutorial(self):
        vg = self.p.make_voxel_grid_filter()
        vg.set_leaf_size(0.01, 0.01, 0.01)
        cloud_filtered = vg.filter()
        tree = cloud_filtered.make_kdtree()

        self.segment = cloud_filtered.make_EuclideanClusterExtraction()
        self.segment.set_ClusterTolerance(0.02)
        self.segment.set_MinClusterSize(100)
        self.segment.set_MaxClusterSize(25000)
        self.segment.set_SearchMethod(tree)
        cluster_indices = self.segment.Extract()

        cloud_cluster = pcl.PointCloud()

        # print('cluster_indices : ' + str(cluster_indices.count) + " count.")
        cloud_cluster = pcl.PointCloud()
        for j, indices in enumerate(cluster_indices):
            # print('indices = ' + str(len(indices)))
            points = np.zeros((len(indices), 3), dtype=np.float32)

            for i, indice in enumerate(indices):
                points[i][0] = cloud_filtered[indice][0]
                points[i][1] = cloud_filtered[indice][1]
                points[i][2] = cloud_filtered[indice][2]

            cloud_cluster.from_array(points)

### RegionGrowing (1.7.2/1.8.0)###
@attr('pcl_over_17')
@attr('pcl_ver_0_4')
class TestRegionGrowing(unittest.TestCase):
    def setUp(self):
        # self.p = pcl.PointCloud(_data)
        self.p = pcl.load(
            './examples/pcldata/tutorials/table_scene_lms400.pcd')

    def testTutorial(self):
        vg = self.p.make_voxel_grid_filter()
        vg.set_leaf_size(0.01, 0.01, 0.01)
        cloud_filtered = vg.filter()
        tree = cloud_filtered.make_kdtree()

        self.segment = cloud_filtered.make_RegionGrowing(ksearch=50)
        self.segment.set_MinClusterSize(100)
        self.segment.set_MaxClusterSize(25000)
        self.segment.set_NumberOfNeighbours(5)
        self.segment.set_SmoothnessThreshold(0.2)
        self.segment.set_CurvatureThreshold(0.05)
        self.segment.set_SearchMethod(tree)
        cluster_indices = self.segment.Extract()

        cloud_cluster = pcl.PointCloud()

        # print('cluster_indices : ' + str(cluster_indices.count) + " count.")
        cloud_cluster = pcl.PointCloud()
        for j, indices in enumerate(cluster_indices):
            # print('indices = ' + str(len(indices)))
            points = np.zeros((len(indices), 3), dtype=np.float32)

            for i, indice in enumerate(indices):
                points[i][0] = cloud_filtered[indice][0]
                points[i][1] = cloud_filtered[indice][1]
                points[i][2] = cloud_filtered[indice][2]

            cloud_cluster.from_array(points)

### MinCutSegmentation(1.7.2) ###
@attr('pcl_over_17')
@attr('pcl_ver_0_4')
class TestMinCutSegmentation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.segment = pcl.MinCutSegmentation()

    def testTutorial(self):
        pass


### ProgressiveMorphologicalFilter ###
@attr('pcl_over_17')
@attr('pcl_ver_0_4')
class TestProgressiveMorphologicalFilter(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.segment = pcl.ProgressiveMorphologicalFilter()

    def testTutorial(self):
        pass


# copy the output of seg
SEGDATA = """ 0.352222 -0.151883  2;
             -0.106395 -0.397406  1;
             -0.473106  0.292602  1;
             -0.731898  0.667105 -2;
              0.441304 -0.734766  1;
              0.854581 -0.0361733 1;
             -0.4607   -0.277468  4;
             -0.916762  0.183749  1;
              0.968809  0.512055  1;
             -0.998983 -0.463871  1;
              0.691785  0.716053  1;
              0.525135 -0.523004  1;
              0.439387  0.56706   1;
              0.905417 -0.579787  1;
              0.898706 -0.504929  1"""

SEGINLIERS = """-0.106395 -0.397406  1;
                -0.473106  0.292602  1;
                 0.441304 -0.734766  1;
                 0.854581 -0.0361733 1;
                -0.916762  0.183749  1;
                 0.968809  0.512055  1;
                -0.998983 -0.463871  1;
                 0.691785  0.716053  1;
                 0.525135 -0.523004  1;
                 0.439387  0.56706   1;
                 0.905417 -0.579787  1;
                 0.898706 -0.504929  1"""
SEGINLIERSIDX = [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]

SEGCOEFF = [0.0, 0.0, 1.0, -1.0]

### Segmentation ###
# class TestSegmentation(unittest.TestCase):


class TestSegmentPlane(unittest.TestCase):

    def setUp(self):
        self.a = np.array(np.mat(SEGDATA, dtype=np.float32))
        self.p = pcl.PointCloud()
        self.p.from_array(self.a)
        self.segment = self.p.make_segmenter()

    def testLoad(self):
        npts = self.a.shape[0]
        self.assertEqual(npts, self.p.size)
        self.assertEqual(npts, self.p.width)
        self.assertEqual(1, self.p.height)

    def testSegmentPlaneObject(self):
        seg = self.p.make_segmenter()
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.01)

        indices, coefficients = seg.segment()
        self.assertListEqual(indices, SEGINLIERSIDX)
        self.assertListEqual(coefficients, SEGCOEFF)
        pass


### SegmentationNormal ###
class TestSegmentationNormal(unittest.TestCase):
    def setUp(self):
        # self.a = np.array(np.mat(SEGDATA, dtype=np.float32))
        # self.p = pcl.PointCloud()
        # self.p.from_array(self.a)
        cloud = pcl.load('tests'  + os.path.sep + 'tutorials'  + os.path.sep + 'table_scene_mug_stereo_textured.pcd')
        
        fil = cloud.make_passthrough_filter()
        fil.set_filter_field_name("z")
        fil.set_filter_limits(0, 1.5)
        cloud_filtered = fil.filter()

        seg = cloud_filtered.make_segmenter_normals(ksearch=50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_normal_distance_weight(0.1)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(100)
        seg.set_distance_threshold(0.03)
        indices, model = seg.segment()
        
        self.p = cloud_filtered.extract(indices, negative=True)
        
        self.segment = self.p.make_segmenter_normals(ksearch=50)
        # self.segment = pcl.SegmentationNormal()
        # self.segment.setInputCloud(self.p)

    # def testLoad(self):
    #     npts = self.a.shape[0]
    #     self.assertEqual(npts, self.p.size)
    #     self.assertEqual(npts, self.p.width)
    #     self.assertEqual(1, self.p.height)


    def testSegmentNormalCylinderObject(self):
        self.segment.set_optimize_coefficients(True)
        self.segment.set_model_type(pcl.SACMODEL_CYLINDER)
        self.segment.set_normal_distance_weight(0.1)
        self.segment.set_method_type(pcl.SAC_RANSAC)
        self.segment.set_max_iterations(10000)
        self.segment.set_distance_threshold(0.05)
        self.segment.set_radius_limits(0, 0.1)

        self.segment.set_axis(1.0, 0.0, 0.0)
        expected = np.array([1.0, 0.0, 0.0])
        param = self.segment.get_axis()
        self.assertEqual(param.tolist(), expected.tolist())
        epsAngle = 35.0
        expected = epsAngle / 180.0 * 3.14
        self.segment.set_eps_angle(epsAngle / 180.0 * 3.14)
        param = self.segment.get_eps_angle()
        self.assertEqual(param, expected)

        indices, coefficients = self.segment.segment()
        # self.assertListEqual(indices, SEGINLIERSIDX)
        # self.assertListEqual(coefficients, SEGCOEFF)

        epsAngle = 50.0
        expected2 = epsAngle / 180.0 * 3.14
        self.segment.set_eps_angle(epsAngle / 180.0 * 3.14)
        param2 = self.segment.get_eps_angle()
        self.assertEqual(param2, expected2)
        self.assertNotEqual(param, param2)
        
        indices2, coefficients2 = self.segment.segment()
        # self.assertListEqual(indices2, SEGINLIERSIDX)
        # self.assertListEqual(coefficients2, SEGCOEFF)
        
        # print(len(indices))
        # print(coefficients)
        
        # print(len(indices2))
        # print(coefficients2)
        
        self.assertNotEqual(len(indices), len(indices2))
        # self.assertListNotEqual(coefficients, coefficients2)
        pass


def suite():
    suite = unittest.TestSuite()

    # segmentation
    suite.addTests(unittest.makeSuite(TestEuclideanClusterExtraction))
    # suite.addTests(unittest.makeSuite(TestSegmentation))
    suite.addTests(unittest.makeSuite(TestSegmentPlane))
    suite.addTests(unittest.makeSuite(TestSegmentationNormal))
    # 1.7.2/1.8.0
    # suite.addTests(unittest.makeSuite(TestConditionalEuclideanClustering))
    # suite.addTests(unittest.makeSuite(TestRegionGrowing))
    # suite.addTests(unittest.makeSuite(TestMinCutSegmentation))
    # suite.addTests(unittest.makeSuite(TestProgressiveMorphologicalFilter))

    return suite


if __name__ == '__main__':
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
