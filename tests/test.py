import os.path
import pickle
import shutil
import tempfile
import unittest

import pcl
import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_equal


_data = [(i, 2 * i, 3 * i + 0.2) for i in range(5)]
_DATA = """0.0, 0.0, 0.2;
           1.0, 2.0, 3.2;
           2.0, 4.0, 6.2;
           3.0, 6.0, 9.2;
           4.0, 8.0, 12.2"""

# io
class TestListIO(unittest.TestCase):
    def setUp(self):
        self.p = pcl.PointCloud(_data)

    def testFromList(self):
        for i, d in enumerate(_data):
            pt = self.p[i]
            assert np.allclose(pt, _data[i])

    def testToList(self):
        l = self.p.to_list()
        assert np.allclose(l, _data)


class TestNumpyIO(unittest.TestCase):
    def setUp(self):
        self.a = np.array(np.mat(_DATA, dtype=np.float32))
        self.p = pcl.PointCloud(self.a)

    def testFromNumpy(self):
        for i, d in enumerate(_data):
            pt = self.p[i]
            assert np.allclose(pt, _data[i])

    def testToNumpy(self):
        a = self.p.to_array()
        self.assertTrue(np.alltrue(a == self.a))

    def test_asarray(self):
        p = pcl.PointCloud(self.p)      # copy
        # old0 = p[0]
        a = np.asarray(p)               # view
        a[:] += 6
        assert_array_almost_equal(p[0], a[0])

        # Regression test: deleting a second view would previously
        # reset the view count to zero.
        b = np.asarray(p)
        del b

        self.assertRaises(ValueError, p.resize, 2 * p.size)

    def test_pickle(self):
        """Test pickle support."""
        # In this testcase because picking reduces to pickling NumPy arrays.
        s = pickle.dumps(self.p)
        p = pickle.loads(s)
        self.assertTrue(np.all(self.a == p.to_array()))

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


class TestSegmentPlane(unittest.TestCase):
    def setUp(self):
        self.a = np.array(np.mat(SEGDATA, dtype=np.float32))
        self.p = pcl.PointCloud()
        self.p.from_array(self.a)

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

        indices, model = seg.segment()
        self.assertListEqual(indices, SEGINLIERSIDX)
        self.assertListEqual(model, SEGCOEFF)


def test_pcd_read():
    TMPL = """# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH %(npts)d
HEIGHT 1
VIEWPOINT 0.1 0 0.5 0 1 0 0
POINTS %(npts)d
DATA ascii
%(data)s"""

    a = np.array(np.mat(SEGDATA, dtype=np.float32))
    npts = a.shape[0]
    tmp_file =  tempfile.mkstemp(suffix='.pcd')[1]
    with open(tmp_file, "w") as f:
        f.write(TMPL % {"npts": npts, "data": SEGDATA.replace(";", "")})

    p = pcl.load(tmp_file)

    assert p.width == npts
    assert p.height == 1

    for i, row in enumerate(a):
        pt = np.array(p[i])
        ssd = sum((row - pt) ** 2)
        assert ssd < 1e-6

    assert_array_equal(p.sensor_orientation,
                       np.array([0, 1, 0, 0], dtype=np.float32))
    assert_array_equal(p.sensor_origin,
                       np.array([.1, 0, .5, 0], dtype=np.float32))


def test_copy():
    a = np.random.randn(100, 3).astype(np.float32)
    p1 = pcl.PointCloud(a)
    p2 = pcl.PointCloud(p1)
    assert_array_equal(p2.to_array(), a)


SEGCYLMOD = [0.0552167, 0.0547035, 0.757707,
             -0.0270852, -4.41026, -2.88995, 0.0387603]
# 1.6 - (Only Mac), other
SEGCYLIN = 11461
# 1.7.2 - (Only Mac)
# SEGCYLIN = 11462
# 1.8 - (Only Mac)
# SEGCYLIN = 11450

class TestSegmentCylinder(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load("tests" + os.path.sep + "table_scene_mug_stereo_textured_noplane.pcd")

    def testSegment(self):
        seg = self.p.make_segmenter_normals(50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_CYLINDER)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_normal_distance_weight(0.1)
        seg.set_max_iterations(10000)
        seg.set_distance_threshold(0.05)
        seg.set_radius_limits(0, 0.1)

        indices, model = seg.segment()

		# MAC NG
        # self.assertEqual(len(indices), SEGCYLIN)

        # npexp = np.array(SEGCYLMOD)
        # npmod = np.array(model)
        # ssd = sum((npexp - npmod) ** 2)
        # self.assertLess(ssd, 1e-6)


class TestSave(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load("tests" + os.path.sep + "table_scene_mug_stereo_textured_noplane.pcd")
        self.tmpdir = tempfile.mkdtemp(suffix='pcl-test')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def testSave(self):
        for ext in ["pcd", "ply"]:
            d = os.path.join(self.tmpdir, "foo." + ext)
            pcl.save(self.p, d)
            p = pcl.load(d)
            self.assertEqual(self.p.size, p.size)


class TestFilter(unittest.TestCase):

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


class TestExtract(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load("tests" + os.path.sep + "flydracyl.pcd")

    def testExtractPos(self):
        p2 = self.p.extract([1, 2, 3], False)
        # new instance is returned
        self.assertNotEqual(self.p, p2)
        self.assertEqual(p2.size, 3)

    def testExtractNeg(self):
        p2 = self.p.extract([1, 2, 3], True)
        self.assertNotEqual(self.p, p2)
        self.assertEqual(p2.size, self.p.size - 3)


class TestExceptions(unittest.TestCase):

    def setUp(self):
        self.p = pcl.PointCloud(np.arange(9, dtype=np.float32).reshape(3, 3))

    def testIndex(self):
        self.assertRaises(IndexError, self.p.__getitem__, self.p.size)
        self.assertRaises(Exception, self.p.get_point, self.p.size, 1)

    def testResize(self):
        # XXX MemoryError isn't actually the prettiest exception for a
        # negative argument. Don't hesitate to change this test to reflect
        # better exceptions.
        self.assertRaises(MemoryError, self.p.resize, -1)


class TestSegmenterNormal(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load("tests" + os.path.sep + "table_scene_mug_stereo_textured_noplane.pcd")

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


class TestVoxelGridFilter(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load("tests" + os.path.sep + "table_scene_mug_stereo_textured_noplane.pcd")

    def testFilter(self):
        fil = self.p.make_voxel_grid_filter()
        fil.set_leaf_size(0.01, 0.01, 0.01)
        c = fil.filter()
        self.assertTrue(c.size < self.p.size)
        self.assertEqual(c.size, 719)


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


# class TestOctreePointCloud(unittest.TestCase):
# 
#     def setUp(self):
#         self.t = pcl.OctreePointCloud(0.1)
# 
#     def testLoad(self):
#         pc = pcl.load("tests" + os.path.sep + "table_scene_mug_stereo_textured_noplane.pcd")
#         self.t.set_input_cloud(pc)
#         self.t.define_bounding_box()
#         self.t.add_points_from_input_cloud()
#         good_point = (0.035296999, -0.074322999, 1.2074)
#         rs = self.t.is_voxel_occupied_at_point(good_point)
#         self.assertTrue(rs)
#         bad_point = (0.5, 0.5, 0.5)
#         rs = self.t.is_voxel_occupied_at_point(bad_point)
#         self.assertFalse(rs)
#         voxels_len = 44
#         self.assertEqual(len(self.t.get_occupied_voxel_centers()), voxels_len)
#         self.t.delete_voxel_at_point(good_point)
#         self.assertEqual(
#             len(self.t.get_occupied_voxel_centers()), voxels_len - 1)


class TestOctreePointCloudSearch(unittest.TestCase):

    def setUp(self):
        self.t = pcl.OctreePointCloudSearch(0.1)
        pc = pcl.load("tests" + os.path.sep + "table_scene_mug_stereo_textured_noplane.pcd")
        self.t.set_input_cloud(pc)
        self.t.define_bounding_box()
        self.t.add_points_from_input_cloud()

    def testConstructor(self):
        self.assertRaises(ValueError, pcl.OctreePointCloudSearch, 0.)

    def testRadiusSearch(self):
        good_point = (0.035296999, -0.074322999, 1.2074)
        rs = self.t.radius_search(good_point, 0.5, 1)
        self.assertEqual(len(rs[0]), 1)
        self.assertEqual(len(rs[1]), 1)
        rs = self.t.radius_search(good_point, 0.5)
        self.assertEqual(len(rs[0]), 19730)
        self.assertEqual(len(rs[1]), 19730)

# class TestOctreePointCloudChangeDetector(unittest.TestCase):
# 
#     def setUp(self):
#         self.t = pcl.OctreePointCloudSearch(0.1)
#         pc = pcl.load("tests" + os.path.sep + "table_scene_mug_stereo_textured_noplane.pcd")
#         self.t.set_input_cloud(pc)
#         self.t.define_bounding_box()
#         self.t.add_points_from_input_cloud()
# 
#     def testConstructor(self):
#         self.assertRaises(ValueError, pcl.OctreePointCloudChangeDetector, 0.)
# 
#     def testRadiusSearch(self):
#         good_point = (0.035296999, -0.074322999, 1.2074)
#         rs = self.t.radius_search(good_point, 0.5, 1)
#         self.assertEqual(len(rs[0]), 1)
#         self.assertEqual(len(rs[1]), 1)
#         rs = self.t.radius_search(good_point, 0.5)
#         self.assertEqual(len(rs[0]), 19730)
#         self.assertEqual(len(rs[1]), 19730)

# class TestCropHull(unittest.TestCase):
# 
#     def setUp(self):
#         self.pc = pcl.load("tests" + os.path.sep + "tutorials" + os.path.sep + "table_scene_mug_stereo_textured.pcd")
# 
#     def testException(self):
#         self.assertRaises(TypeError, pcl.CropHull)
# 
#     def testCropHull(self):
#         filterCloud = pcl.PointCloud()
#         vt = pcl.Vertices()
#         # // inside point
#         # cloud->push_back(pcl::PointXYZ(M_PI * 0.3, M_PI * 0.3, 0));
#         # // hull points
#         # cloud->push_back(pcl::PointXYZ(0,0,0));
#         # cloud->push_back(pcl::PointXYZ(M_PI,0,0));
#         # cloud->push_back(pcl::PointXYZ(M_PI,M_PI*0.5,0));
#         # cloud->push_back(pcl::PointXYZ(0,M_PI*0.5,0));
#         # cloud->push_back(pcl::PointXYZ(0,0,0));
#         # // outside point
#         # cloud->push_back(pcl::PointXYZ(-M_PI * 0.3, -M_PI * 0.3, 0));
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
#         # cropHull.filter(indices);
#         # outputCloud = cropHull.filter();
#         # print("before: " + outputCloud)
#         crophull.Filtering(outputCloud)
#         # print(outputCloud)


# Viewer
# // pcl::visualization::CloudViewer viewer ("Cluster viewer");
# // viewer.showCloud(colored_cloud);

# Write Point
# pcl::PCDWriter writer;
# std::stringstream ss;
# ss << "min_cut_seg" << ".pcd";
# // writer.write<pcl::PointXYZRGB> (ss.str (), *cloud, false);
# pcl::io::savePCDFile(ss.str(), *outputCloud, false);


# class TestCropBox(unittest.TestCase):
# 
#     def setUp(self):
#         # self.pc = pcl.load("tests" + os.path.sep + "table_scene_mug_stereo_textured_noplane.pcd")
#         self.pc = pcl.load("tests" + os.path.sep + "tutorials" + os.path.sep + "table_scene_mug_stereo_textured.pcd")
# 
#     def testException(self):
#         self.assertRaises(TypeError, pcl.CropHull)
# 
#     def testCrop(self):
#         pc = pcl.load("tests" + os.path.sep + "tutorials" + os.path.sep + "table_scene_mug_stereo_textured.pcd")
#         clipper = self.pc.make_cropbox()
#         outcloud = pcl.PointCloud()
#         self.assertEqual(outcloud.size, 0)
#         
#         # clipper.setTranslation(Eigen::Vector3f(pose->tx, pose->ty, pose->tz));
#         # clipper.setRotation(Eigen::Vector3f(pose->rx, pose->ry, pose->rz));
#         # clipper.setMin(-Eigen::Vector4f(tracklet->l/2, tracklet->w/2, 0, 0));
#         # clipper.setMax(Eigen::Vector4f(tracklet->l/2, tracklet->w/2, tracklet->h, 0));
#         # clipper.filter(*outcloud);
#         tx = 0
#         ty = 0
#         tz = 0
#         clipper.set_Translation(tx, ty, tz)
#         rx = 0
#         ry = 0
#         rz = 0
#         clipper.set_Rotation(rx, ry, rz)
#         minx = -1.5
#         miny = -1.5
#         minz = 0
#         mins = 0
#         maxx = 3.5
#         maxy = 3.5
#         maxz = 2
#         maxs = 0
#         clipper.set_MinMax(minx, miny, minz, mins, maxx, maxy, maxz, maxs)
#         clipper.Filtering(outcloud)
#         # self.assertNotEqual(outcloud.size, 0)
#         self.assertNotEqual(outcloud.size, self.pc.size)

# Add ProjectInlier
# class TestProjectInlier(unittest.TestCase):
# 
#     def setUp(self):
#         # TestData
#         self.pc = pcl.PointCloud(a)
#         self.kd = pcl.CropBox(self.pc)
# 
#     def testException(self):
#         self.assertRaises(TypeError, pcl.CropHull)
#         self.assertRaises(TypeError, self.kd.nearest_k_search_for_cloud, None)
# 
#     def testCrop(self):
#         # Big cluster
#         for ref, k in ((80, 1), (59, 3), (60, 10)):
#             ind, sqdist = self.kd.nearest_k_search_for_point(self.pc, ref, k=k)
#             for i in ind:
#                 self.assertGreaterEqual(i, 30)
#             for d in sqdist:
#                 self.assertGreaterEqual(d, 0)
# 

# Add RadiusOutlierRemoval

# Add ConditionAnd

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestListIO))
    suite.addTests(unittest.makeSuite(TestNumpyIO))
    suite.addTests(unittest.makeSuite(TestSegmentPlane))
    suite.addTests(unittest.makeSuite(TestSegmentCylinder))
    suite.addTests(unittest.makeSuite(TestSave))
    suite.addTests(unittest.makeSuite(TestFilter))
    suite.addTests(unittest.makeSuite(TestExtract))
    suite.addTests(unittest.makeSuite(TestExceptions))
    suite.addTests(unittest.makeSuite(TestSegmenterNormal))
    suite.addTests(unittest.makeSuite(TestVoxelGridFilter))
    suite.addTests(unittest.makeSuite(TestPassthroughFilter))
    suite.addTests(unittest.makeSuite(TestKdTree))
    suite.addTests(unittest.makeSuite(TestOctreePointCloudSearch))
    return suite

if __name__ == '__main__':
    unittest.main()