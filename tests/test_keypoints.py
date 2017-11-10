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


# keyPoints
### HarrisKeypoint3D ###


class TestHarrisKeypoint3D(unittest.TestCase):
    def setUp(self):
        self.p = pcl.load(
            "tests" +
            os.path.sep +
            "tutorials" +
            os.path.sep +
            "bunny.pcd")
        self.kp = self.p.make_HarrisKeypoint3D()

    def test_HarrisKeyPoint3D(self):
        # 397
        base_point_count = 397
        self.assertEqual(self.p.size, base_point_count)

        self.kp.set_NonMaxSupression(True)
        self.kp.set_Radius(0.01)
        # self.kp.set_RadiusSearch (0.01)
        keypoints = self.kp.compute()

        # pcl - 1.8, 51
        # pcl - 1.7, 48
        # keypoint_count = 51
        # self.assertEqual(keypoints.size, keypoint_count)
        self.assertNotEqual(keypoints.size, 0)

        count = 0
        minIts = 999.00
        maxIts = -999.00
        points = np.zeros((keypoints.size, 3), dtype=np.float32)
        # Generate the data
        for i in range(0, keypoints.size):
            # set Point Plane
            points[i][0] = keypoints[i][0]
            points[i][1] = keypoints[i][1]
            points[i][2] = keypoints[i][2]
            intensity = keypoints[i][3]
            if intensity > maxIts:
                print("coords: " +
                      str(keypoints[i][0]) +
                      ";" +
                      str(keypoints[i][1]) +
                      ";" +
                      str(keypoints[i][2]))
                maxIts = intensity

            if intensity < minIts:
                minIts = intensity

            count = count + 1

        # points.resize(count, 3)
        # print(points)
        # keypoints3D.from_array(points)
        # print("maximal responce: " + str(maxIts) + " min responce:  " +  str(minIts) )
        ##
        # coords: 0.008801460266113281;0.12533344328403473;0.03247201442718506
        # coords: 0.02295708656311035;0.12180554866790771;0.029724061489105225
        # coords: -0.06679701805114746;0.15040874481201172;0.03854072093963623
        # coords: -0.0672549456357956;0.11913366615772247;0.05214547738432884
        # coords: -0.05888630822300911;0.1165248453617096;0.03698881343007088
        # coords: 0.04757949709892273;0.07463110238313675;0.018482372164726257
        # maximal responce: 0.0162825807929039 min responce:  0.0

        # pcl 1.7 : 0.01632295921444893
        # self.assertEqual(maxIts, 0.0162825807929039)
        self.assertGreaterEqual(maxIts, 0.0)
        self.assertEqual(minIts, 0.0)


### NarfKeypoint ###
@attr('pcl_ver_0_4')
class TestNarfKeypoint(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        # self.kp = pcl.NarfKeypoint()
        # self.kp.setInputCloud(self.p)

    def test_NarfKeypoint(self):
        pass


### UniformSampling ###
@attr('pcl_ver_0_4')
class TestUniformSampling(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)
        # self.kp = pcl.UniformSampling()

    def test_UniformSampling(self):
        pass


def suite():
    suite = unittest.TestSuite()

    # keypoints
    suite.addTests(unittest.makeSuite(TestHarrisKeypoint3D))
    # RangeImage no set
    # suite.addTests(unittest.makeSuite(TestNarfKeypoint))
    # no add pxiInclude
    # suite.addTests(unittest.makeSuite(TestUniformSampling))

    return suite


if __name__ == '__main__':
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
