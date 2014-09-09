import numpy as np
from numpy import cos, sin
import unittest

import pcl
from pcl.registration import icp, gicp, icp_nl

transform = 0

class TestICP(unittest.TestCase):
    def setUp(self):
        # Check if ICP can find a mild rotation.
        theta = [-.031, .4, .59]
        rot_x = [[ 1,              0,             0             ],
                 [ 0,              cos(theta[0]), -sin(theta[0])],
                 [ 0,              sin(theta[0]),  cos(theta[0])]]
        rot_y = [[ cos(theta[1]),  0,              sin(theta[1])],
                 [ 0,              1,              0            ],
                 [-sin(theta[1]),  0,              cos(theta[1])]]
        rot_z = [[ cos(theta[2]), -sin(theta[1]),  0            ],
                 [ sin(theta[2]),  cos(theta[1]),  0            ],
                 [ 0,              0,              1            ]]
        transform = np.dot(rot_x, np.dot(rot_y, rot_z))

        source = np.random.RandomState(42).randn(900, 3)
        self.source = pcl.PointCloud(source.astype(np.float32))

        target = np.dot(source, transform)
        self.target = pcl.PointCloud(target.astype(np.float32))

    def testICP(self):
        converged, transf, estimate, fitness = icp(self.source, self.target,
                                                   max_iter=1000)
        self.assertTrue(converged is True)
        self.assertLess(fitness, .1)

        # XXX I think I misunderstand fitness, it's not equal to the following
        # MSS.
        # mss = (np.linalg.norm(estimate.to_array()
        #                       - self.source.to_array(), axis=1) ** 2).mean()
        # self.assertLess(mss, 1)

        # TODO check the actual transformation matrix.
        print("------ICP")
        print "Converged: ", converged, "Estimate: ", estimate, "Fitness ", fitness
        print "Rotation: "
        print transf[0:3,0:3]
        print "Translation: ", transf[3, 0:3]
        print("---------")
   
    def testGICP(self):
        converged, transf, estimate, fitness = gicp(self.source, self.target,
                                                    max_iter=1000)
        self.assertTrue(converged is True)
        self.assertLess(fitness, .1)

        # XXX I think I misunderstand fitness, it's not equal to the following
        # MSS.
        # mss = (np.linalg.norm(estimate.to_array()
        #                       - self.source.to_array(), axis=1) ** 2).mean()
        # self.assertLess(mss, 1)

        # TODO check the actual transformation matrix.
        print("-----GICP")
        print "Converged: ", converged, "Estimate: ", estimate, "Fitness ", fitness
        print "Rotation: "
        print transf[0:3,0:3]
        print "Translation: ", transf[3, 0:3]
        print("---------")

    def testICP_NL(self):
        converged, transf, estimate, fitness = icp_nl(self.source, self.target,
                                                      max_iter=1000)
        self.assertTrue(converged is True)
        self.assertLess(fitness, .1)

        # XXX I think I misunderstand fitness, it's not equal to the following
        # MSS.
        # mss = (np.linalg.norm(estimate.to_array()
        #                       - self.source.to_array(), axis=1) ** 2).mean()
        # self.assertLess(mss, 1)

        # TODO check the actual transformation matrix.
        print("---ICP_NL")
        print "Converged: ", converged, "Estimate: ", estimate, "Fitness ", fitness
        print "Rotation: "
        print transf[0:3,0:3]
        print "Translation: ", transf[3, 0:3]
        print("---------")
