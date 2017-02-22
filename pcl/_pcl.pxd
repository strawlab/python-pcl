# -*- coding: utf-8 -*-
# Header for _pcl.pyx functionality that needs sharing with other modules.

cimport pcl_defs as cpp
# KdTree
cimport pcl_kdtree as pclkdt
# RangeImage
cimport pcl_range_image as pcl_rngimg
# Features
cimport pcl_features as pcl_ftr

# class override(PointCloud)
cdef class PointCloud:
    cdef cpp.PointCloudPtr_t thisptr_shared     # XYZ
    
    # Buffer protocol support.
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _view_count
    
    cdef inline cpp.PointCloud[cpp.PointXYZ] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloud<PointXYZ>.
        return self.thisptr_shared.get()


# class override(PointCloud_PointXYZI)
cdef class PointCloud_PointXYZI:
    cdef cpp.PointCloud_PointXYZI_Ptr_t thisptr_shared     # XYZI
    
    # Buffer protocol support.
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _view_count
    
    cdef inline cpp.PointCloud[cpp.PointXYZI] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloud<PointXYZ>.
        return self.thisptr_shared.get()


# class override(PointCloud_PointXYZRGB)
cdef class PointCloud_PointXYZRGB:
    cdef cpp.PointCloud_PointXYZRGB_Ptr_t thisptr_shared
    
    # Buffer protocol support.
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _view_count
    
    cdef inline cpp.PointCloud[cpp.PointXYZRGB] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloud<PointXYZRGB>.
        return self.thisptr_shared.get()


# class override(PointCloud_PointXYZRGBA)
cdef class PointCloud_PointXYZRGBA:
    cdef cpp.PointCloud_PointXYZRGBA_Ptr_t thisptr_shared   # XYZRGBA
    
    # Buffer protocol support.
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _view_count
    
    cdef inline cpp.PointCloud[cpp.PointXYZRGBA] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloud<PointXYZRGBA>.
        return self.thisptr_shared.get()

# class override
cdef class Vertices:
    cdef cpp.VerticesPtr_t thisptr_shared     # Vertices
    
    # Buffer protocol support.
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _view_count
    
    cdef inline cpp.Vertices *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying Vertices
        return self.thisptr_shared.get()

# class override(PointCloud_PointWithViewpoint)
cdef class PointCloud_PointWithViewpoint:
    cdef cpp.PointCloud_PointWithViewpoint_Ptr_t thisptr_shared   # PointWithViewpoint
    
    # Buffer protocol support.
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _view_count
    
    cdef inline cpp.PointCloud[cpp.PointWithViewpoint] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloud<PointWithViewpoint>.
        return self.thisptr_shared.get()


# class override(PointCloud_Normal)
cdef class PointCloud_Normal:
    cdef cpp.PointCloud_Normal_Ptr_t thisptr_shared   # Normal
    
    # Buffer protocol support.
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _view_count
    
    cdef inline cpp.PointCloud[cpp.Normal] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloud<Normal>.
        return self.thisptr_shared.get()


# class override(PointCloud_PointNormal)
cdef class PointCloud_PointNormal:
    cdef cpp.PointCloud_PointNormal_Ptr_t thisptr_shared   # PointNormal
    
    # Buffer protocol support.
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _view_count
    
    cdef inline cpp.PointCloud[cpp.PointNormal] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloud<PointNormal>.
        return self.thisptr_shared.get()


## KdTree
# class override
cdef class KdTree:
    cdef pclkdt.KdTreePtr_t thisptr_shared   # KdTree
    
    cdef inline pclkdt.KdTree[cpp.PointXYZ] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying KdTree<PointXYZ>.
        return self.thisptr_shared.get()

# cdef class KdTreeFLANN:
#     cdef pclkdt.KdTreeFLANNPtr_t thisptr_shared   # KdTreeFLANN
#     
#     cdef inline pclkdt.KdTreeFLANN[cpp.PointXYZ] *thisptr(self) nogil:
#         # Shortcut to get raw pointer to underlying KdTreeFLANN<PointXYZ>.
#         return self.thisptr_shared.get()


## RangeImages
# class override
cdef class RangeImages:
    cdef pcl_rngimg.RangeImagePtr_t thisptr_shared   # RangeImages
    
    cdef inline pcl_rngimg.RangeImage *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying RangeImage.
        return self.thisptr_shared.get()


### Features
# class override
cdef class IntegralImageNormalEstimation:
    cdef pcl_ftr.IntegralImageNormalEstimationPtr_t thisptr_shared     # IntegralImageNormalEstimation
    
    cdef inline pcl_ftr.IntegralImageNormalEstimation[cpp.PointXYZ, cpp.Normal] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal>.
        return self.thisptr_shared.get()


