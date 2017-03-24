
from libcpp cimport bool

cimport numpy as np
import numpy as np

cimport _pcl
cimport pcl_defs as cpp
cimport pcl_registration_172 as pcl_reg
from boost_shared_ptr cimport shared_ptr

from eigen cimport Matrix4f

np.import_array()

cdef class IterativeClosestPoint:
    """
    Registration class for IterativeClosestPoint
    """
    cdef pcl_reg.IterativeClosestPoint_t *me

    def __cinit__(self):
        self.me = new pcl_reg.IterativeClosestPoint_t()

    def __dealloc__(self):
        del self.me

    # def set_InputTarget(self, pcl_reg.Registration[cpp.PointXYZ, cpp.PointXYZ] &reg):
    def set_InputTarget(self, _pcl.PointCloud cloud):
        self.me.setInputTarget (cloud.thisptr_shared)
        pass

    # def get_Resolution(self):
    #     return self.me.getResolution()

    # def get_StepSize(self):
    #     return self.me.getStepSize()

    # def set_StepSize(self, double step_size):
    #     self.me.setStepSize(step_size)

    # def get_OulierRatio(self):
    #     return self.me.getOulierRatio()

    # def set_OulierRatio(self, double outlier_ratio):
    #     self.me.setOulierRatio(outlier_ratio)

    # def get_TransformationProbability(self):
    #     return self.me.getTransformationProbability()

    # def get_FinalNumIteration(self):
    #     return self.me.getFinalNumIteration()

    cdef object run(self, pcl_reg.Registration[cpp.PointXYZ, cpp.PointXYZ, float] &reg, _pcl.PointCloud source, _pcl.PointCloud target, max_iter=None):
        reg.setInputTarget(target.thisptr_shared)
        
        if max_iter is not None:
            reg.setMaximumIterations(max_iter)
        
        cdef _pcl.PointCloud result = _pcl.PointCloud()
        
        with nogil:
            reg.align(result.thisptr()[0])
        
        # Get transformation matrix and convert from Eigen to NumPy format.
        # cdef pcl_reg.Registration[cpp.PointXYZ, cpp.PointXYZ].Matrix4f mat
        cdef Matrix4f mat
        mat = reg.getFinalTransformation()
        cdef np.ndarray[dtype=np.float32_t, ndim=2, mode='fortran'] transf
        cdef np.float32_t *transf_data
        
        transf = np.empty((4, 4), dtype=np.float32, order='fortran')
        transf_data = <np.float32_t *>np.PyArray_DATA(transf)
        
        for i in range(16):
            transf_data[i] = mat.data()[i]
        
        return reg.hasConverged(), transf, result, reg.getFitnessScore()

    def icp(self, _pcl.PointCloud source, _pcl.PointCloud target, max_iter=None):
        """
        Align source to target using iterative closest point (ICP).
        Parameters
        ----------
        source : PointCloud
            Source point cloud.
        target : PointCloud
            Target point cloud.
        max_iter : integer, optional
            Maximum number of iterations. If not given, uses the default number
            hardwired into PCL.
        Returns
        -------
        converged : bool
            Whether the ICP algorithm converged in at most max_iter steps.
        transf : np.ndarray, shape = [4, 4]
            Transformation matrix.
        estimate : PointCloud
            Transformed version of source.
        fitness : float
            Sum of squares error in the estimated transformation.
        """
        cdef pcl_reg.IterativeClosestPoint_t icp
        icp.setInputCloud(source.thisptr_shared)
        return self.run(icp, source, target, max_iter)
