# -*- coding: utf-8 -*-
from libcpp.vector cimport vector

cimport pcl_defs as cpp
cimport pcl_sample_consensus as pcl_sac


cdef class RandomSampleConsensus:
    """
    represents an implementation of the RANSAC (RAndom SAmple Consensus) algorithm.
    """
    cdef pcl_sac.RandomSampleConsensus_t *me


    # build error
    def __cinit__(self, model=None):
        cdef SampleConsensusModel tmpModel
        cdef SampleConsensusModelCylinder tmpModelCylinder
        cdef SampleConsensusModelSphere tmpModelSphere
        cdef SampleConsensusModelLine tmpModelLine
        cdef SampleConsensusModelPlane tmpModelPlane
        cdef SampleConsensusModelRegistration tmpModelRegistration
        cdef SampleConsensusModelStick tmpModelStick

        if model is None:
            return
        elif isinstance(model, SampleConsensusModel):
            tmpModel = model
            # tmpModel.thisptr()[0] = model.thisptr()[0]
            self.me = new pcl_sac.RandomSampleConsensus_t(<pcl_sac.SampleConsensusModelPtr_t> tmpModel.thisptr_shared)
        elif isinstance(model, SampleConsensusModelCylinder):
            tmpModelCylinder = model
            # tmpModelCylinder.thisptr()[0] = model.thisptr()[0]
            self.me = new pcl_sac.RandomSampleConsensus_t(<pcl_sac.SampleConsensusModelPtr_t> tmpModelCylinder.thisptr_shared)
        elif isinstance(model, SampleConsensusModelLine):
            tmpModelLine = model
            self.me = new pcl_sac.RandomSampleConsensus_t(<pcl_sac.SampleConsensusModelPtr_t> tmpModelLine.thisptr_shared)
        elif isinstance(model, SampleConsensusModelPlane):
            tmpModelPlane = model
            self.me = new pcl_sac.RandomSampleConsensus_t(<pcl_sac.SampleConsensusModelPtr_t> tmpModelPlane.thisptr_shared)
        elif isinstance(model, SampleConsensusModelRegistration):
            tmpModelRegistration = model
            self.me = new pcl_sac.RandomSampleConsensus_t(<pcl_sac.SampleConsensusModelPtr_t> tmpModelRegistration.thisptr_shared)
        elif isinstance(model, SampleConsensusModelSphere):
            tmpModelSphere = model
            self.me = new pcl_sac.RandomSampleConsensus_t(<pcl_sac.SampleConsensusModelPtr_t> tmpModelSphere.thisptr_shared)
        elif isinstance(model, SampleConsensusModelStick):
            tmpModelStick = model
            self.me = new pcl_sac.RandomSampleConsensus_t(<pcl_sac.SampleConsensusModelPtr_t> tmpModelStick.thisptr_shared)
        else:
            raise TypeError("Can't initialize a RandomSampleConsensus from a %s"
                            % type(model))
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


