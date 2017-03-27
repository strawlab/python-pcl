# -*- coding: utf-8 -*-
# pcl version 1.8.0
cimport pcl_defs as cpp
cimport pcl_registration_180 as pclreg

cdef class NormalDistributionsTransform:
    """
    Registration class for NormalDistributionsTransform
    """
    cdef pclreg.NormalDistributionsTransform_t *me

    def __cinit__(self):
        self.me = new pclreg.NormalDistributionsTransform_t()

    def __dealloc__(self):
        del self.me

    # def set_InputTarget(self, pclreg.RegistrationPtr_t cloud):
    def set_InputTarget(self):
        # self.me.setInputTarget (cloud.this_ptr())
        pass

    def set_Resolution(self, float resolution):
        self.me.setResolution(resolution)
        pass

    def get_Resolution(self):
        return self.me.getResolution()

    def get_StepSize(self):
        return self.me.getStepSize()

    def set_StepSize(self, double step_size):
        self.me.setStepSize(step_size)

    def get_OulierRatio(self):
        return self.me.getOulierRatio()

    def set_OulierRatio(self, double outlier_ratio):
        self.me.setOulierRatio(outlier_ratio)
    
    def get_TransformationProbability(self):
        return self.me.getTransformationProbability()
    
    def get_FinalNumIteration(self):
        return self.me.getFinalNumIteration()

