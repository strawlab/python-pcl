# Header for _pcl.pyx functionality that needs sharing with other
# modules.

cimport pcl_defs as cpp


cdef class PointCloud:
    cdef cpp.PointCloud[cpp.PointXYZ] *thisptr
