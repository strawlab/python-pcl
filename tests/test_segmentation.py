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
        self.p = pcl.load('./examples/pcldata/tutorials/table_scene_lms400.pcd')
        # self.segment = self.p


    def testTutorial(self):
        vg = self.p.make_voxel_grid_filter()
        vg.set_leaf_size (0.01, 0.01, 0.01)
        cloud_filtered = vg.filter ()
        tree = cloud_filtered.make_kdtree()

        self.segment = cloud_filtered.make_EuclideanClusterExtraction()
        self.segment.set_ClusterTolerance (0.02)
        self.segment.set_MinClusterSize (100)
        self.segment.set_MaxClusterSize (25000)
        self.segment.set_SearchMethod (tree)
        cluster_indices = self.segment.Extract()

        cloud_cluster = pcl.PointCloud()

        print('cluster_indices : ' + str(cluster_indices.count) + " count.")
        cloud_cluster = pcl.PointCloud()
        for j, indices in enumerate(cluster_indices):
            print('indices = ' + str(len(indices)))
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


### Segmentation ###
class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.segment = pcl.Segmentation()


    def testTutorial(self):
        pass


### SegmentationNormal ###
class TestSegmentationNormal(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        self.segment = pcl.SegmentationNormal()


    def testTutorial(self):
        pass


def suite():
    suite = unittest.TestSuite()

    # segmentation
    suite.addTests(unittest.makeSuite(TestEuclideanClusterExtraction))
    suite.addTests(unittest.makeSuite(TestSegmentation))
    suite.addTests(unittest.makeSuite(TestSegmentationNormal))
    # 1.7.2/1.8.0
    # suite.addTests(unittest.makeSuite(TestConditionalEuclideanClustering))
    # suite.addTests(unittest.makeSuite(TestMinCutSegmentation))
    # suite.addTests(unittest.makeSuite(TestProgressiveMorphologicalFilter))

    return suite


if __name__ == '__main__':
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
