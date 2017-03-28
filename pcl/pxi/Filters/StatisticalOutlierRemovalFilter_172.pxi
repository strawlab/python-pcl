# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_filters_172 as pclfil

cdef class StatisticalOutlierRemovalFilter:
    """
    Filter class uses point neighborhood statistics to filter outlier data.
    """
    cdef pclfil.StatisticalOutlierRemoval_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pclfil.StatisticalOutlierRemoval_t()
        (<cpp.PCLBase_t*>self.me).setInputCloud (pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    property mean_k:
        def __get__(self):
            return self.me.getMeanK()
        def __set__(self, int k):
            self.me.setMeanK(k)

    property negative:
        def __get__(self):
            return (<pclfil.FilterIndices[cpp.PointXYZ]*>self.me).getNegative()
        def __set__(self, bool neg):
            (<pclfil.FilterIndices[cpp.PointXYZ]*>self.me).setNegative(neg)

    property stddev_mul_thresh:
        def __get__(self):
            return self.me.getStddevMulThresh()
        def __set__(self, double thresh):
            self.me.setStddevMulThresh(thresh)

    def set_InputCloud(self, PointCloud pc not None):
        (<cpp.PCLBase_t*>self.me).setInputCloud (pc.thisptr_shared)

    def set_mean_k(self, int k):
        """
        Set the number of points (k) to use for mean distance estimation. 
        """
        self.me.setMeanK(k)

    def set_std_dev_mul_thresh(self, double std_mul):
        """
        Set the standard deviation multiplier threshold.
        """
        self.me.setStddevMulThresh(std_mul)

    def set_negative(self, bool negative):
        """
        Set whether the indices should be returned, or all points except the indices. 
        """
        (<pclfil.FilterIndices[cpp.PointXYZ]*>self.me).setNegative(negative)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc


cdef class StatisticalOutlierRemovalFilter_PointXYZI:
    """
    Filter class uses point neighborhood statistics to filter outlier data.
    """
    cdef pclfil.StatisticalOutlierRemoval_PointXYZI_t *me

    def __cinit__(self, PointCloud_PointXYZI pc not None):
        self.me = new pclfil.StatisticalOutlierRemoval_PointXYZI_t()
        (<cpp.PCLBase_PointXYZI_t*>self.me).setInputCloud (pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    property mean_k:
        def __get__(self):
            return self.me.getMeanK()
        def __set__(self, int k):
            self.me.setMeanK(k)

    property negative:
        def __get__(self):
            return (<pclfil.FilterIndices[cpp.PointXYZI]*>self.me).getNegative()
        def __set__(self, bool neg):
            (<pclfil.FilterIndices[cpp.PointXYZI]*>self.me).setNegative(neg)

    property stddev_mul_thresh:
        def __get__(self):
            return self.me.getStddevMulThresh()
        def __set__(self, double thresh):
            self.me.setStddevMulThresh(thresh)

    def set_InputCloud(self, PointCloud_PointXYZI pc not None):
        (<cpp.PCLBase_PointXYZI_t*>self.me).setInputCloud (pc.thisptr_shared)

    def set_mean_k(self, int k):
        """
        Set the number of points (k) to use for mean distance estimation. 
        """
        self.me.setMeanK(k)

    def set_std_dev_mul_thresh(self, double std_mul):
        """
        Set the standard deviation multiplier threshold.
        """
        self.me.setStddevMulThresh(std_mul)

    def set_negative(self, bool negative):
        """
        Set whether the indices should be returned, or all points except the indices. 
        """
        (<pclfil.FilterIndices[cpp.PointXYZ]*>self.me).setNegative(negative)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZI pc = PointCloud_PointXYZI()
        self.me.filter(pc.thisptr()[0])
        return pc


cdef class StatisticalOutlierRemovalFilter_PointXYZRGB:
    """
    Filter class uses point neighborhood statistics to filter outlier data.
    """
    cdef pclfil.StatisticalOutlierRemoval_PointXYZRGB_t *me

    def __cinit__(self, PointCloud_PointXYZRGB pc not None):
        self.me = new pclfil.StatisticalOutlierRemoval_PointXYZRGB_t()
        (<cpp.PCLBase_PointXYZRGB_t*>self.me).setInputCloud (pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    property mean_k:
        def __get__(self):
            return self.me.getMeanK()
        def __set__(self, int k):
            self.me.setMeanK(k)

    property negative:
        def __get__(self):
            return (<pclfil.FilterIndices[cpp.PointXYZRGB]*>self.me).getNegative()
        def __set__(self, bool neg):
            (<pclfil.FilterIndices[cpp.PointXYZRGB]*>self.me).setNegative(neg)

    property stddev_mul_thresh:
        def __get__(self):
            return self.me.getStddevMulThresh()
        def __set__(self, double thresh):
            self.me.setStddevMulThresh(thresh)

    def set_InputCloud(self, PointCloud_PointXYZRGB pc not None):
        (<cpp.PCLBase_PointXYZRGB_t*>self.me).setInputCloud (pc.thisptr_shared)

    def set_mean_k(self, int k):
        """
        Set the number of points (k) to use for mean distance estimation. 
        """
        self.me.setMeanK(k)

    def set_std_dev_mul_thresh(self, double std_mul):
        """
        Set the standard deviation multiplier threshold.
        """
        self.me.setStddevMulThresh(std_mul)

    def set_negative(self, bool negative):
        """
        Set whether the indices should be returned, or all points except the indices. 
        """
        (<pclfil.FilterIndices[cpp.PointXYZRGB]*>self.me).setNegative(negative)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZRGB pc = PointCloud_PointXYZRGB()
        self.me.filter(pc.thisptr()[0])
        return pc


cdef class StatisticalOutlierRemovalFilter_PointXYZRGBA:
    """
    Filter class uses point neighborhood statistics to filter outlier data.
    """
    cdef pclfil.StatisticalOutlierRemoval_PointXYZRGBA_t *me

    def __cinit__(self, PointCloud_PointXYZRGBA pc not None):
        self.me = new pclfil.StatisticalOutlierRemoval_PointXYZRGBA_t()
        (<cpp.PCLBase_PointXYZRGBA_t*>self.me).setInputCloud (pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    property mean_k:
        def __get__(self):
            return self.me.getMeanK()
        def __set__(self, int k):
            self.me.setMeanK(k)

    property negative:
        def __get__(self):
            return (<pclfil.FilterIndices[cpp.PointXYZRGBA]*>self.me).getNegative()
        def __set__(self, bool neg):
            (<pclfil.FilterIndices[cpp.PointXYZRGBA]*>self.me).setNegative(neg)

    property stddev_mul_thresh:
        def __get__(self):
            return self.me.getStddevMulThresh()
        def __set__(self, double thresh):
            self.me.setStddevMulThresh(thresh)

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        (<cpp.PCLBase_PointXYZRGBA_t*>self.me).setInputCloud (pc.thisptr_shared)

    def set_mean_k(self, int k):
        """
        Set the number of points (k) to use for mean distance estimation. 
        """
        self.me.setMeanK(k)

    def set_std_dev_mul_thresh(self, double std_mul):
        """
        Set the standard deviation multiplier threshold.
        """
        self.me.setStddevMulThresh(std_mul)

    def set_negative(self, bool negative):
        """
        Set whether the indices should be returned, or all points except the indices. 
        """
        self.me.setNegative(negative)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZRGBA pc = PointCloud_PointXYZRGBA()
        self.me.filter(pc.thisptr()[0])
        return pc

