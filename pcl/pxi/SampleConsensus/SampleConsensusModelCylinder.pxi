# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_sample_consensus as pcl_sac

cdef class SampleConsensusModelCylinder:
    """
    defines a model for 3D cylinder segmentation class.
    """

    def __cinit__(self, PointCloud pc not None):
        # shared_ptr
        sp_assign(self.thisptr_shared, new pcl_sac.SampleConsensusModelCylinder[cpp.PointXYZ, cpp.Normal](pc.thisptr_shared))
        pass

