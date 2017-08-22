# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_sample_consensus as pcl_sac

from boost_shared_ptr cimport sp_assign

cdef class SampleConsensusModelSphere:
    """
    """
    # cdef pcl_sac.SampleConsensusModelSphere_t *me

    def __cinit__(self, PointCloud pc not None):
        # NG
        # self.me = new pcl_sac.SampleConsensusModelSphere_t()
        # self.me = new pcl_sac.SampleConsensusModelSphere_t(pc.thisptr_shared)
        # shared_ptr
        # NG
        # sp_assign(self.thisptr_shared, new pcl_sac.SampleConsensusModelSphere_t(pc.thisptr_shared))
        sp_assign(self.thisptr_shared, new pcl_sac.SampleConsensusModelSphere[cpp.PointXYZ](pc.thisptr_shared))
        pass

    # def __dealloc__(self):
    #     del self.me

