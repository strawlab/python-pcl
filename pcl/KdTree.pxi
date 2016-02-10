
cimport pcl_defs as cpp

cdef class KdTree:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTree(pc).
    """
    cdef cpp.KdTree_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new cpp.KdTree_t()

        self.me.setInputCloud(pc.thisptr_shared)
