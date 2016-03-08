#cython: embedsignature=True
#
# Copyright 2014 Netherlands eScience Center

from libcpp cimport bool

cimport numpy as np
import numpy as np

cimport _pcl
cimport pcl_defs as cpp
cimport pcl_grabber as pcl_grb
from boost_shared_ptr cimport shared_ptr

# cdef object run(pcl_reg.Registration[cpp.PointXYZ, cpp.PointXYZ] &reg,
#                 _pcl.PointCloud source, _pcl.PointCloud target, max_iter=None):
#     # 1.6.0 NG(No descrription)
#     # reg.setInputSource(source.thisptr_shared)
#     # PCLBase
#     cdef cpp.PCLBase[cpp.PointXYZ] pclbase
#     # NG(Convert)
#     # pclbase = reg
#     # pclbase = <cpp.PCLBase> reg
#     pclbase = <cpp.PCLBase[cpp.PointXYZ]> reg
#     # pclbase.setInputCloud(source.thisptr_shared)
#     pclbase.setInputCloud(<cpp.PointCloudPtr_t> source.thisptr_shared)
#     # set PointCloud?
#     # getInputCloud
#     # reg.setInputCloud(<cpp.PointCloudPtr_t> pclbase.getInputCloud())
#     reg.setInputTarget(target.thisptr_shared)
# 
#     if max_iter is not None:
#         reg.setMaximumIterations(max_iter)
# 
#     cdef _pcl.PointCloud result = _pcl.PointCloud()
# 
#     with nogil:
#         reg.align(result.thisptr()[0])
# 
#     # Get transformation matrix and convert from Eigen to NumPy format.
#     # cdef pcl_reg.Registration[cpp.PointXYZ, cpp.PointXYZ].Matrix4f mat
#     cdef Matrix4f mat
#     mat = reg.getFinalTransformation()
#     cdef np.ndarray[dtype=np.float32_t, ndim=2, mode='fortran'] transf
#     cdef np.float32_t *transf_data
# 
#     transf = np.empty((4, 4), dtype=np.float32, order='fortran')
#     transf_data = <np.float32_t *>np.PyArray_DATA(transf)
# 
#     for i in range(16):
#         transf_data[i] = mat.data()[i]
# 
#     print (reg.hasConverged())
#     print (transf)
#     print (result)
#     # NG
#     print (reg.getFitnessScore())
# 
#     return reg.hasConverged(), transf, result, reg.getFitnessScore()
# 
# 
# def icp(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None):
#     """Align source to target using iterative closest point (ICP).
# 
#     Parameters
#     ----------
#     source : PointCloud
#         Source point cloud.
#     target : PointCloud
#         Target point cloud.
#     max_iter : integer, optional
#         Maximum number of iterations. If not given, uses the default number
#         hardwired into PCL.
# 
#     Returns
#     -------
#     converged : bool
#         Whether the ICP algorithm converged in at most max_iter steps.
#     transf : np.ndarray, shape = [4, 4]
#         Transformation matrix.
#     estimate : PointCloud
#         Transformed version of source.
#     fitness : float
#         Sum of squares error in the estimated transformation.
#     """
#     cdef pcl_reg.IterativeClosestPoint[cpp.PointXYZ, cpp.PointXYZ] icp
#     # icp.setInputCloud(source);
#     return run(icp, source, target, max_iter)
# 
# 
# def gicp(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None):
#     """Align source to target using generalized iterative closest point (GICP).
# 
#     Parameters
#     ----------
#     source : PointCloud
#         Source point cloud.
#     target : PointCloud
#         Target point cloud.
#     max_iter : integer, optional
#         Maximum number of iterations. If not given, uses the default number
#         hardwired into PCL.
# 
#     Returns
#     -------
#     converged : bool
#         Whether the ICP algorithm converged in at most max_iter steps.
#     transf : np.ndarray, shape = [4, 4]
#         Transformation matrix.
#     estimate : PointCloud
#         Transformed version of source.
#     fitness : float
#         Sum of squares error in the estimated transformation.
#     """
#     cdef pcl_reg.GeneralizedIterativeClosestPoint[cpp.PointXYZ, cpp.PointXYZ] gicp
#     # gicp.setInputCloud(<shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> source);
#     return run(gicp, source, target, max_iter)
# 
# 
# def icp_nl(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None):
#     """Align source to target using generalized non-linear ICP (ICP-NL).
# 
#     Parameters
#     ----------
#     source : PointCloud
#         Source point cloud.
#     target : PointCloud
#         Target point cloud.
# 
#     max_iter : integer, optional
#         Maximum number of iterations. If not given, uses the default number
#         hardwired into PCL.
# 
#     Returns
#     -------
#     converged : bool
#         Whether the ICP algorithm converged in at most max_iter steps.
#     transf : np.ndarray, shape = [4, 4]
#         Transformation matrix.
#     estimate : PointCloud
#         Transformed version of source.
#     fitness : float
#         Sum of squares error in the estimated transformation.
#     """
#     cdef pcl_reg.IterativeClosestPointNonLinear[cpp.PointXYZ, cpp.PointXYZ] icp_nl
#     # icp_nl.setInputCloud(source);
#     return run(icp_nl, source, target, max_iter)
###
 