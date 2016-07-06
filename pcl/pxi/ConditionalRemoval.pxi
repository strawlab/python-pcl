from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_filters as pclfil

cimport eigen as eigen3

from boost_shared_ptr cimport shared_ptr

cdef class ConditionalRemoval:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in ConditionalRemoval(pc).
    """
    cdef pclfil.ConditionalRemoval_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pclfil.ConditionalRemoval_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

	def set_KeepOrganized(flag):
		self.me.setKeepOrganized(flag)

    def set_ConditionAdd(condAdd):
        # Convert Eigen::Vector3f
        cdef eigen3.Vector3f origin
        cdef float *data = origin.data()
        data[0] = tx
        data[1] = ty
        data[2] = tz
        self.me.setTranslation(origin)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc
