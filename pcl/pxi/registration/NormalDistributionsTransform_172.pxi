# -*- coding: utf-8 -*-
# pcl version 1.7.2
cimport pcl_defs as cpp
cimport pcl_registration_172 as pclreg

cdef class NormalDistributionsTransform:
    """
    Registration class for NormalDistributionsTransform
    """
    cdef pcl_reg.NormalDistributionsTransform_t *me

    def __cinit__(self):
        self.me = new pcl_reg.NormalDistributionsTransform_t()

    def __dealloc__(self):
        del self.me

    # # def set_InputTarget(self, pcl_reg.RegistrationPtr_t cloud):
    # def set_InputTarget(self):
    #     # self.me.setInputTarget (cloud.this_ptr())
    #     pass

    # def set_Resolution(self, float resolution):
    #     self.me.setResolution(resolution)
    #     pass

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

    cdef object run(self, pcl_reg.NormalDistributionsTransform_t &reg, _pcl.PointCloud source, _pcl.PointCloud target):
        reg.setInputTarget(target.thisptr_shared)
                
        cdef _pcl.PointCloud result = _pcl.PointCloud()
        
        reg.align(result.thisptr()[0])
        
        # Get transformation matrix and convert from Eigen to NumPy format.
        # cdef pcl_reg.Registration[cpp.PointXYZ, cpp.PointXYZ].Matrix4f mat
        cdef Matrix4f mat
        mat = reg.getFinalTransformation()
        cdef np.ndarray[dtype=np.float32_t, ndim=2, mode='fortran'] transf
        cdef np.float32_t *transf_data
        
        transf = np.empty((4, 4), dtype=np.float32, order='F')
        transf_data = <np.float32_t *>np.PyArray_DATA(transf)
        
        for i in range(16):
            transf_data[i] = mat.data()[i]
        
        return reg.hasConverged(), transf, result, reg.getTransformationProbability()

    def ndt(self, _pcl.PointCloud source, _pcl.PointCloud target, max_iter=None, resolution=None, step_size=None, outlier_ratio=None):
        """
        Align source to target using normal distributions transform (NDT).
        Parameters
        ----------
        source : PointCloud
            Source point cloud.
        target : PointCloud
            Target point cloud.
        max_iter : integer, optional
            Maximum number of iterations. If not given, uses the default number
            hardwired into PCL.
        resolution : float, optional
            Voxel grid resolution. If not given, uses the default number
            hardwired into PCL.
        step_size : float, optional
            Newton line search maximum step length. If not given, uses the default number
            hardwired into PCL.
        outlier_ratio : float, optional
            Point cloud outlier ratio. If not given, uses the default number
            hardwired into PCL.
        Returns
        -------
        converged : bool
            Whether the NDT algorithm converged in at most max_iter steps.
        transf : np.ndarray, shape = [4, 4]
            Transformation matrix.
        estimate : PointCloud
            Transformed version of source.
        transf_prob : float
            The registration alignment probability.
        """
        cdef pcl_reg.NormalDistributionsTransform_t ndt

        if max_iter is not None:
            ndt.setMaximumIterations(max_iter)
        if resolution is not None:
            ndt.setResolution(max_iter)
        if step_size is not None:
            ndt.setStepSize(max_iter)
        if outlier_ratio is not None:
            ndt.setOulierRatio(max_iter)

        ndt.setInputCloud(source.thisptr_shared)
        return self.run(ndt, source, target)
