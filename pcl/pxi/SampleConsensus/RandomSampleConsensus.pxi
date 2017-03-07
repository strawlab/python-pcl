# -*- coding: utf-8 -*-
from libcpp.vector cimport vector

cimport pcl_defs as cpp
cimport pcl_sample_consensus as pcl_sac


cdef class RandomSampleConsensus:
    """
    """
    cdef pcl_sac.RandomSampleConsensus_t *me

    def __cinit__(self, SampleConsensusModel model):
        # NG
        # self.me = new pcl_sac.RandomSampleConsensus_t()
        self.me = new pcl_sac.RandomSampleConsensus_t(<pcl_sac.SampleConsensusModelPtr_t> model.thisptr_shared)
        # shared_ptr[SampleConsensusModel[T]]
        pass

    def __cinit__(self, SampleConsensusModelPlane model):
        # NG
        # self.me = new pcl_sac.RandomSampleConsensus_t()
        self.me = new pcl_sac.RandomSampleConsensus_t(<pcl_sac.SampleConsensusModelPtr_t> model.thisptr_shared)
        pass

    def __cinit__(self, SampleConsensusModelSphere model):
        # NG
        # self.me = new pcl_sac.RandomSampleConsensus_t()
        self.me = new pcl_sac.RandomSampleConsensus_t(<pcl_sac.SampleConsensusModelPtr_t> model.thisptr_shared)
        pass

    def __dealloc__(self):
        del self.me

    def computeModel(self):
        self.me.computeModel(0)

    # base Class(SampleConsensus)
    def set_DistanceThreshold(self, double param):
        self.me.setDistanceThreshold(param)

    # base Class(SampleConsensus)
    def get_Inliers(self):
        cdef vector[int] inliers
        self.me.getInliers(inliers)
        return inliers


