# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_sample_consensus as pcl_sac

cdef class SampleConsensusModelRegistration:
    """
    defines a model for Point-To-Point registration outlier rejection class.
    """

    def __cinit__(self, PointCloud pc not None):
        # shared_ptr
        sp_assign(self.thisptr_shared, new pcl_sac.SampleConsensusModelRegistration[cpp.PointXYZ](pc.thisptr_shared))
        pass

