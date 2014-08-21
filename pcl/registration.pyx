#cython: embedsignature=True
#
# Copyright 2014 Netherlands eScience Center

from libcpp cimport bool

cimport _pcl
cimport pcl_defs as cpp

cdef extern from "pcl/registration/icp.h" namespace "pcl":
    cdef cppclass IterativeClosestPoint[Source, Target]:
        IterativeClosestPoint() except +
        void align(cpp.PointCloud[Source] &) except +
        double getFitnessScore() except +
        bool hasConverged() except +
        void setInputSource(cpp.PointCloudPtr_t) except +
        void setInputTarget(cpp.PointCloudPtr_t) except +
        void setMaximumIterations(int) except +


def icp(_pcl.PointCloud source, _pcl.PointCloud target,
        unsigned max_iter):
    """Align source to target using iterative closests point (ICP).

    Parameters
    ----------
    source : PointCloud
        Source point cloud.
    target : PointCloud
        Target point cloud.
    max_iter : integer
        Maximum number of iterations.

    Returns
    -------
    converged : bool
        Whether the ICP algorithm converged in at most max_iter steps.
    transf : np.ndarray, shape = [4, 4]
        Transformation matrix.
    estimate : PointCloud
        Transformed version of source.
    fitness : float
        Sum of squares error in the estimated transformation.
    """

    cdef IterativeClosestPoint[cpp.PointXYZ,
                               cpp.PointXYZ] icp

    # XXX the following deep-copies both point clouds. ICP desperately wants
    # shared pointers; it doesn't work on bare pointers or references.
    # Must check whether http://stackoverflow.com/a/10644487/166749 does what
    # it promises, else change _pcl.PointCloud to use a smart pointer.
    icp.setInputSource(source.thisptr.makeShared())
    icp.setInputTarget(target.thisptr.makeShared())

    icp.setMaximumIterations(max_iter)

    cdef _pcl.PointCloud result = _pcl.PointCloud()

    icp.align(result.thisptr[0])

    # TODO return transformation as promised.
    return icp.hasConverged(), result, icp.getFitnessScore()
