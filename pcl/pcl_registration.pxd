from libcpp cimport bool

# main
cimport pcl_defs as cpp
from boost_shared_ptr cimport shared_ptr

cdef extern from "pcl/registration/registration.h" namespace "pcl" nogil:
    cdef cppclass Registration[Source, Target]:
        cppclass Matrix4:
            float *data()
        void align(cpp.PointCloud[Source] &) except +
        Matrix4 getFinalTransformation() except +
        double getFitnessScore() except +
        bool hasConverged() except +
        void setInputSource(cpp.PointCloudPtr_t) except +
        void setInputTarget(cpp.PointCloudPtr_t) except +
        void setMaximumIterations(int) except +

cdef extern from "pcl/registration/icp.h" namespace "pcl" nogil:
    cdef cppclass IterativeClosestPoint[Source, Target](Registration[Source, Target]):
        IterativeClosestPoint() except +

cdef extern from "pcl/registration/gicp.h" namespace "pcl" nogil:
    cdef cppclass GeneralizedIterativeClosestPoint[Source, Target](Registration[Source, Target]):
        GeneralizedIterativeClosestPoint() except +

cdef extern from "pcl/registration/icp_nl.h" namespace "pcl" nogil:
    cdef cppclass IterativeClosestPointNonLinear[Source, Target](Registration[Source, Target]):
        IterativeClosestPointNonLinear() except +

