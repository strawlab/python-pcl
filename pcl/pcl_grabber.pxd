# -*- coding: utf-8 -*-
# Header for pcl_grabber.pyx functionality that needs sharing with other modules.

cimport pcl_grabber as cpp

# # class override(PointCloud)
# cdef class PointCloud:
#     cdef cpp.PointCloudPtr_t thisptr_shared     # XYZ
#     
#     # Buffer protocol support.
#     cdef Py_ssize_t _shape[2]
#     cdef Py_ssize_t _view_count
#     
#     cdef inline cpp.PointCloud[cpp.PointXYZ] *thisptr(self) nogil:
#         # Shortcut to get raw pointer to underlying PointCloud<PointXYZ>.
#         return self.thisptr_shared.get()
# 
# 
# # class override(PointCloud_PointXYZI)
# cdef class PointCloud_PointXYZI:
#     cdef cpp.PointCloud_PointXYZI_Ptr_t thisptr_shared     # XYZI
#     
#     # Buffer protocol support.
#     cdef Py_ssize_t _shape[2]
#     cdef Py_ssize_t _view_count
#     
#     cdef inline cpp.PointCloud[cpp.PointXYZI] *thisptr(self) nogil:
#         # Shortcut to get raw pointer to underlying PointCloud<PointXYZ>.
#         return self.thisptr_shared.get()
# 
# 
# # class override(PointCloud_PointXYZRGB)
# cdef class PointCloud_PointXYZRGB:
#     cdef cpp.PointCloud_PointXYZRGB_Ptr_t thisptr_shared
#     
#     # Buffer protocol support.
#     cdef Py_ssize_t _shape[2]
#     cdef Py_ssize_t _view_count
#     
#     cdef inline cpp.PointCloud[cpp.PointXYZRGB] *thisptr(self) nogil:
#         # Shortcut to get raw pointer to underlying PointCloud<PointXYZRGB>.
#         return self.thisptr_shared.get()
# 
# 
# # class override(PointCloud_PointXYZRGBA)
# cdef class PointCloud_PointXYZRGBA:
#     cdef cpp.PointCloud_PointXYZRGBA_Ptr_t thisptr_shared   # XYZRGBA
#     
#     # Buffer protocol support.
#     cdef Py_ssize_t _shape[2]
#     cdef Py_ssize_t _view_count
#     
#     cdef inline cpp.PointCloud[cpp.PointXYZRGBA] *thisptr(self) nogil:
#         # Shortcut to get raw pointer to underlying PointCloud<PointXYZRGBA>.
#         return self.thisptr_shared.get()
# 
