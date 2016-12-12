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
        # self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    # def set_NonMaxSupression(self, bool param):
    #     self.me.setNonMaxSupression (param)

    # 
    #     self.me.setRadius (param)

    # def set_RadiusSearch(self, double param):
    #     self.me.setRadiusSearch (param)

    # def compute(self):
    #    self.me.compute ();


