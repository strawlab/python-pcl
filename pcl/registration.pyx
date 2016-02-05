#cython: embedsignature=True
#
# Copyright 2014 Netherlands eScience Center

from libcpp cimport bool

cimport numpy as np
import numpy as np

cimport _pcl
cimport pcl_defs as cpp

np.import_array()


cdef extern from "pcl/registration/icp.h" namespace "pcl":
    cdef cppclass IterativeClosestPoint[Source, Target]:
        cppclass Matrix4:
            float *data()

        IterativeClosestPoint() except +
        void align(cpp.PointCloud[Source] &) except +
        Matrix4 getFinalTransformation() except +
        double getFitnessScore() except +
        bool hasConverged() except +
        void setInputSource(cpp.PointCloudPtr_t) except +
        void setInputTarget(cpp.PointCloudPtr_t) except +
        void setMaximumIterations(int) except +


def icp(_pcl.PointCloud source, _pcl.PointCloud target,
        max_iter=None):
    """Align source to target using iterative closests point (ICP).

    Parameters
    ----------
    source : PointCloud
        Source point cloud.
    target : PointCloud
        Target point cloud.
    max_iter : integer, optional
        Maximum number of iterations. If not given, uses the default number
        hardwired into PCL.

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
    # icp.setInputSource(source.thisptr.makeShared())
    # icp.setInputTarget(target.thisptr.makeShared())

    if max_iter is not None:
        icp.setMaximumIterations(max_iter)

    cdef _pcl.PointCloud result = _pcl.PointCloud()

    # icp.align(result.thisptr[0])

    # Convert transformation from Eigen to NumPy format.
    cdef const float *transf_data = icp.getFinalTransformation().data()
    cdef np.ndarray[dtype=np.float32_t, ndim=1, mode='c'] transf_flat
    transf = np.zeros((4, 4), dtype=np.float32)
    transf_flat = transf.ravel()

    for i in range(16):
        transf_flat[i] = transf_data[i]

    return icp.hasConverged(), transf, result, icp.getFitnessScore()

