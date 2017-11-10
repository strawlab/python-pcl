import os.path
import pickle
import shutil
import tempfile
import unittest

import pcl
import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_equal

from nose.plugins.attrib import attr


_data = [(i, 2 * i, 3 * i + 0.2) for i in range(5)]
_DATA = """0.0, 0.0, 0.2;
           1.0, 2.0, 3.2;
           2.0, 4.0, 6.2;
           3.0, 6.0, 9.2;
           4.0, 8.0, 12.2"""


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


### local function ###
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
    tmp_file = tempfile.mkstemp(suffix='.pcd')[1]
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


###


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


class TestSave(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load(
            "tests" +
            os.path.sep +
            "table_scene_mug_stereo_textured_noplane.pcd")
        self.tmpdir = tempfile.mkdtemp(suffix='pcl-test')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def testSave(self):
        for ext in ["pcd", "ply"]:
            d = os.path.join(self.tmpdir, "foo." + ext)
            pcl.save(self.p, d)
            p = pcl.load(d)
            self.assertEqual(self.p.size, p.size)


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


def suite():
    suite = unittest.TestSuite()
    # io
    suite.addTests(unittest.makeSuite(TestListIO))
    suite.addTests(unittest.makeSuite(TestNumpyIO))
    suite.addTests(unittest.makeSuite(TestSave))
    suite.addTests(unittest.makeSuite(TestExtract))
    suite.addTests(unittest.makeSuite(TestExceptions))
    return suite


if __name__ == '__main__':
    testSuite = suite()
    unittest.TextTestRunner().run(testSuite)
