# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_sample_consensus as pcl_sac

from boost_shared_ptr cimport sp_assign

cdef class SampleConsensusModel:
    """
    represents the base model class.
    """
    # cdef pcl_sac.SampleConsensusModel_t *me

    def __cinit__(self, PointCloud pc not None):
        # NG
        # self.me = new pcl_sac.SampleConsensusModel_t()
        # self.me = new pcl_sac.SampleConsensusModel_t(pc.thisptr_shared)
        # shared_ptr
        # sp_assign(self.thisptr_shared, pc.thisptr_shared)
        sp_assign(self.thisptr_shared, new pcl_sac.SampleConsensusModel_t(pc.thisptr_shared))
        pass

    # def __dealloc__(self):
    #     del self.me

