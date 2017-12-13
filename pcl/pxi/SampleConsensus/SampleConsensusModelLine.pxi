# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_sample_consensus as pcl_sac

cdef class SampleConsensusModelLine:
    """
    defines a model for 3D line segmentation class.
    """

    def __cinit__(self, PointCloud pc not None):
        # shared_ptr
        sp_assign(self.thisptr_shared, new pcl_sac.SampleConsensusModelLine[cpp.PointXYZ](pc.thisptr_shared))
        pass

