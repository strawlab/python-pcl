
cimport pcl_defs as cpp

cdef class StatisticalOutlierRemovalFilter:
    """
    Filter class uses point neighborhood statistics to filter outlier data.
    """
    cdef cpp.StatisticalOutlierRemoval_t *me
    def __cinit__(self):
        self.me = new cpp.StatisticalOutlierRemoval_t()
    def __dealloc__(self):
        del self.me

    property mean_k:
        def __get__(self):
            return self.me.getMeanK()
        def __set__(self, int k):
            self.me.setMeanK(k)

    property negative:
        def __get__(self):
            return self.me.getNegative()
        def __set__(self, bool neg):
            self.me.setNegative(neg)

    property stddev_mul_thresh:
        def __get__(self):
            return self.me.getStddevMulThresh()
        def __set__(self, double thresh):
            self.me.setStddevMulThresh(thresh)

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
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc