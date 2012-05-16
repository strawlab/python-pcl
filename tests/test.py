import unittest

import pcl
import numpy as np

_data = [(i,2*i,3*i) for i in range(5)]
_DATA = \
"""0.0, 0.0, 0.0;
1.0, 2.0, 3.0;
2.0, 4.0, 6.0;
3.0, 6.0, 9.0;
4.0, 8.0, 12.0"""

def test_fromlist():
    p = pcl.PointCloud()
    p.from_list(_data)
    for i,d in enumerate(_data):
        pt = p[i]
        assert pt == _data[i]

def test_fromnp():
    p = pcl.PointCloud()
    a = np.array(np.mat(_DATA, dtype=np.float32))
    p.from_array(a)
    for i,d in enumerate(_data):
        pt = p[i]
        assert pt == _data[i]

#copy the output of seg
SEGDATA = \
"""0.352222 -0.151883 2;
-0.106395 -0.397406 1;
-0.473106 0.292602 1;
-0.731898 0.667105 -2;
0.441304 -0.734766 1;
0.854581 -0.0361733 1;
-0.4607 -0.277468 4;
-0.916762 0.183749 1;
0.968809 0.512055 1;
-0.998983 -0.463871 1;
0.691785 0.716053 1;
0.525135 -0.523004 1;
0.439387 0.56706 1;
0.905417 -0.579787 1;
0.898706 -0.504929 1"""

SEGINLIERS = \
"""-0.106395 -0.397406 1;
-0.473106 0.292602 1;
0.441304 -0.734766 1;
0.854581 -0.0361733 1;
-0.916762 0.183749 1;
0.968809 0.512055 1;
-0.998983 -0.463871 1;
0.691785 0.716053 1;
0.525135 -0.523004 1;
0.439387 0.56706 1;
0.905417 -0.579787 1;
0.898706 -0.504929 1"""
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

    def testSegmentPlaneFunc(self):
        a = np.array(np.mat(SEGINLIERS, dtype=np.float32))
        
        indices, model = self.p._simple_segment_plane()
        self.assertListEqual(indices, SEGINLIERSIDX)
        self.assertListEqual(model, SEGCOEFF)

    def testSegmentPlaneObject(self):
        seg = self.p.make_segmenter()
        seg.set_optimize_coefficients (True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold (0.01)

        indices, model = seg.segment()
        self.assertListEqual(indices, SEGINLIERSIDX)
        self.assertListEqual(model, SEGCOEFF)

def test_pcd_read():
    TMPL = """
# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH %(npts)d
HEIGHT 1
VIEWPOINT 0 0 0 0 1 0 0
POINTS %(npts)d
DATA ascii
%(data)s"""

    a = np.array(np.mat(SEGDATA, dtype=np.float32))
    npts = a.shape[0]
    with open("/tmp/test.pcd","w") as f:
        f.write(TMPL % {"npts":npts,"data":SEGDATA.replace(";","")})

    p = pcl.PointCloud()
    p.from_file("/tmp/test.pcd")

    assert p.width == npts
    assert p.height == 1

    for i,row in enumerate(a):
        pt = np.array(p[i])
        ssd = sum((row - pt) ** 2)
        assert ssd < 1e-6

SEGCYLMOD = [0.0552167, 0.0547035, 0.757707, -0.0270852, -4.41026, -2.88995, 0.0387603]
SEGCYLIN = 11461

class TestSegmentCylinder(unittest.TestCase):

    def setUp(self):
        self.p = pcl.PointCloud()
        self.p.from_file("tests/table_scene_mug_stereo_textured_noplane.pcd")

    def testSegment(self):
        seg = self.p.make_segmenter_normals(50)
        seg.set_optimize_coefficients (True);
        seg.set_model_type (pcl.SACMODEL_CYLINDER)
        seg.set_method_type (pcl.SAC_RANSAC)
        seg.set_normal_distance_weight (0.1)
        seg.set_max_iterations (10000)
        seg.set_distance_threshold (0.05)
        seg.set_radius_limits (0, 0.1)

        indices, model = seg.segment()

        self.assertEqual(len(indices), SEGCYLIN)

        npexp = np.array(SEGCYLMOD)
        npmod = np.array(model)
        ssd = sum((npexp - npmod) ** 2)
        self.assertLess(ssd, 1e-6)        
        
