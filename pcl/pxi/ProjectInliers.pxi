
cimport pcl_filters as pclfil
cimport pcl_segmentation as pclseg
cimport pcl_defs as cpp

cdef class ProjectInliers:
    """
    ProjectInliers class for Sample Consensus methods and models
    """
    cdef pclfil.ProjectInliers_t *me
    def __cinit__(self):
        self.me = new pclfil.ProjectInliers_t()
    def __dealloc__(self):
        del self.me

    def filter(self):
        # cdef cpp.PointCloud_t pc
        cdef PointCloud pc = PointCloud()
        self.me.filter (pc.thisptr()[0])
        # return [ind.indices[i] for i in range(ind.indices.size())], \
        #        [coeffs.values[i] for i in range(coeffs.values.size())]
        return pc

    # def set_Model_Coefficients(self, cpp.ModelCoefficients coeffs):
    #     self.me.setModelCoefficients(coeffs)
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

