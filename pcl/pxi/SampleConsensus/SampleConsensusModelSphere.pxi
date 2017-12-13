# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_sample_consensus as pcl_sac

from boost_shared_ptr cimport sp_assign

cdef class SampleConsensusModelSphere:
    """
    define a model for 3D sphere segmentation class.
    """
    # cdef pcl_sac.SampleConsensusModelSphere_t *me

    def __cinit__(self, PointCloud pc not None):
        # NG
        # sp_assign(self.thisptr_shared, new pcl_sac.SampleConsensusModelSphere_t(pc.thisptr_shared))
        sp_assign(self.thisptr_shared, new pcl_sac.SampleConsensusModelSphere[cpp.PointXYZ](pc.thisptr_shared))
        pass

    # def __dealloc__(self):
    #     del self.me

