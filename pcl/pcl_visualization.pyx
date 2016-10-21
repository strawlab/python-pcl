# cython: embedsignature=True
# Copyright 2014 Netherlands eScience Center

from libcpp cimport bool

cimport numpy as np
import numpy as np

cimport pcl_defs as cpp
cimport pcl_visualization as pcl_vis
from boost_shared_ptr cimport shared_ptr

include "pxi/Visualization/Visualization.pxi"
include "pxi/Visualization/PCLHistogramViewing.pxi"
include "pxi/Visualization/PCLVisualizering.pxi"


# def SetColorACloud(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None):
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
# def SetColorCloud(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None):
#     """
#     Align source to target using generalized iterative closest point (GICP).
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
# def GrayCloud(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None):
#     """
#     Align source to target using generalized iterative closest point (GICP).
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
# def MonochromeCloud(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None):
#     """
#     Align source to target using generalized iterative closest point (GICP).
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
# def WasStopped (int millis_to_wait = 1)
###


