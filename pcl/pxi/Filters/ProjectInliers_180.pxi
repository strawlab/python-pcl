# -*- coding: utf-8 -*-
cimport pcl_filters_180 as pclfil
cimport pcl_segmentation_180 as pclseg
cimport pcl_defs as cpp

cdef class ProjectInliers:
    """
    ProjectInliers class for ...
    """
    cdef pclfil.ProjectInliers_t *me
    def __cinit__(self):
        self.me = new pclfil.ProjectInliers_t()
    def __dealloc__(self):
        del self.me

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc

    # def set_Model_Coefficients(self):
    #     cdef cpp.ModelCoefficients *coeffs
    #     coeffs.values.resize(4)
    #     coeffs.values[0] = 0
    #     coeffs.values[1] = 0
    #     coeffs.values[2] = 1.0
    #     coeffs.values[3] = 0
    #     self.me.setModelCoefficients(coeffs)
    #     
    # def get_Model_Coefficients(self):
    #     self.me.getModelCoefficients()
    def set_model_type(self, pclseg.SacModel m):
        self.me.setModelType(m)
    def get_model_type(self):
        return self.me.getModelType()
    def set_copy_all_data(self, bool m):
        self.me.setCopyAllData (m)
    def get_copy_all_data(self):
        return self.me.getCopyAllData ()

