from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp
cimport pcl_keypoints as keypt

# boost
from boost_shared_ptr cimport shared_ptr

###############################################################################
# Types
###############################################################################

### base class ###

cdef class HarrisKeypoint3D:
    """
    HarrisKeypoint3D class for 
    """
    cdef keypt.HarrisKeypoint3DPtr_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new keypt.HarrisKeypoint3DPtr_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    def set_NonMaxSupression(self, double param):
        self.me.setNonMaxSupression (true);

    def set_Radius(self, double param):
        self.me.setRadius (0.01);

    def set_RadiusSearch(self, double param):
        self.me.setRadiusSearch (0.01);

    # def compute(self):
    #    self.me.compute ();


