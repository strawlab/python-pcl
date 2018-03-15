# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_keypoints_172 as pcl_kp

cdef class UniformSampling:
    """
    """
    cdef pcl_kp.UniformSampling_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pcl_kp.UniformSampling_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me


# cdef class UniformSampling_PointXYZI:
#     """
#     """
#     cdef pcl_kp.UniformSampling_PointXYZI_t *me
# 
#     def __cinit__(self, PointCloud_PointXYZI pc not None):
#         self.me = new pcl_kp.UniformSampling_PointXYZI_t()
#         # self.me.setInputCloud(<shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> pc.thisptr_shared)
#         self.me.setInputCloud(pc.thisptr_shared)
# 
#     def __dealloc__(self):
#         del self.me
###

# cdef class UniformSampling_PointXYZRGB:
#     """
#     """
#     cdef pcl_kp.UniformSampling_PointXYZRGB_t *me
# 
#     def __cinit__(self, PointCloud_PointXYZRGB pc not None):
#         self.me = new pcl_kp.UniformSampling_PointXYZRGB_t()
#         # self.me.setInputCloud(<shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> pc.thisptr_shared)
#         self.me.setInputCloud(pc.thisptr_shared)
# 
#     def __dealloc__(self):
#         del self.me
###

# cdef class UniformSampling_PointXYZRGBA:
#     """
#     """
#     cdef pcl_kp.UniformSampling_PointXYZRGBA_t *me
# 
#     def __cinit__(self, PointCloud_PointXYZRGBA pc not None):
#         self.me = new pcl_kp.UniformSampling_PointXYZRGBA_t()
#         # self.me.setInputCloud(<shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> pc.thisptr_shared)
#         self.me.setInputCloud(pc.thisptr_shared)
# 
#     def __dealloc__(self):
#         del self.me
###
