# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_sample_consensus as pcl_sac

cdef class SampleConsensusModelPlane:
    """
    defines a model for 3D plane segmentation class.
    """
    # cdef pcl_sac.SampleConsensusModelPlane_t *me

    def __cinit__(self, PointCloud pc not None):
        # NG
        # self.me = new pcl_sac.SampleConsensusModelPlane_t()
        # self.me = new pcl_sac.SampleConsensusModelPlane_t(pc.thisptr_shared)
        # shared_ptr
        sp_assign(self.thisptr_shared, new pcl_sac.SampleConsensusModelPlane[cpp.PointXYZ](pc.thisptr_shared))
        pass

    # def __dealloc__(self):
    #     del self.me

