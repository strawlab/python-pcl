# -*- coding: utf-8 -*-
# cython: embedsignature=True

from collections import Sequence
import numbers
import numpy as np
cimport numpy as cnp

cimport pcl_common as pcl_cmn
cimport pcl_defs as cpp

### DEFINE ###
PCL_MAJOR_VERSION    = cpp.PCL_MAJOR_VERSION
PCL_MINOR_VERSION    = cpp.PCL_MINOR_VERSION
# PCL_REVISION_VERSION = cpp.PCL_REVISION_VERSION

include "pxi/pyx_cimport_180.pxi"

cimport cython
# from cython.operator import dereference as deref
from cython.operator cimport dereference as deref, preincrement as inc
from cython cimport address

from cpython cimport Py_buffer

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

from boost_shared_ptr cimport sp_assign

cnp.import_array()

### Enum ###

## Enum Setting
SAC_RANSAC = pcl_sac.SAC_RANSAC
SAC_LMEDS = pcl_sac.SAC_LMEDS
SAC_MSAC = pcl_sac.SAC_MSAC
SAC_RRANSAC = pcl_sac.SAC_RRANSAC
SAC_RMSAC = pcl_sac.SAC_RMSAC
SAC_MLESAC = pcl_sac.SAC_MLESAC
SAC_PROSAC = pcl_sac.SAC_PROSAC

SACMODEL_PLANE = pcl_sac.SACMODEL_PLANE
SACMODEL_LINE = pcl_sac.SACMODEL_LINE
SACMODEL_CIRCLE2D = pcl_sac.SACMODEL_CIRCLE2D
SACMODEL_CIRCLE3D = pcl_sac.SACMODEL_CIRCLE3D
SACMODEL_SPHERE = pcl_sac.SACMODEL_SPHERE
SACMODEL_CYLINDER = pcl_sac.SACMODEL_CYLINDER
SACMODEL_CONE = pcl_sac.SACMODEL_CONE
SACMODEL_TORUS = pcl_sac.SACMODEL_TORUS
SACMODEL_PARALLEL_LINE = pcl_sac.SACMODEL_PARALLEL_LINE
SACMODEL_PERPENDICULAR_PLANE = pcl_sac.SACMODEL_PERPENDICULAR_PLANE
SACMODEL_PARALLEL_LINES = pcl_sac.SACMODEL_PARALLEL_LINES
SACMODEL_NORMAL_PLANE = pcl_sac.SACMODEL_NORMAL_PLANE 
SACMODEL_NORMAL_SPHERE = pcl_sac.SACMODEL_NORMAL_SPHERE
SACMODEL_REGISTRATION = pcl_sac.SACMODEL_REGISTRATION
SACMODEL_PARALLEL_PLANE = pcl_sac.SACMODEL_PARALLEL_PLANE
SACMODEL_NORMAL_PARALLEL_PLANE = pcl_sac.SACMODEL_NORMAL_PARALLEL_PLANE
SACMODEL_STICK = pcl_sac.SACMODEL_STICK

### Enum Setting(define Class InternalType) ###

# CythonCompareOp
@cython.internal
cdef class _CythonCompareOp_Type:
    cdef:
        readonly int GT
        readonly int GE
        readonly int LT
        readonly int LE
        readonly int EQ

    def __cinit__(self):
        self.GT = pcl_fil.COMPAREOP_GT
        self.GE = pcl_fil.COMPAREOP_GE
        self.LT = pcl_fil.COMPAREOP_LT
        self.LE = pcl_fil.COMPAREOP_LE
        self.EQ = pcl_fil.COMPAREOP_EQ

CythonCompareOp_Type = _CythonCompareOp_Type()

# RangeImage 
# CythonCoordinateFrame
@cython.internal
cdef class _CythonCoordinateFrame_Type:
    cdef:
        readonly int CAMERA_FRAME
        readonly int LASER_FRAME

    def __cinit__(self):
        self.CAMERA_FRAME = pcl_rim.COORDINATEFRAME_CAMERA
        self.LASER_FRAME = pcl_rim.COORDINATEFRAME_LASER

CythonCoordinateFrame_Type = _CythonCoordinateFrame_Type()

# # features
# # CythonBorderPolicy
# @cython.internal
# cdef class _CythonBorderPolicy_Type:
#     cdef:
#         readonly int BORDER_POLICY_IGNORE
#         readonly int BORDER_POLICY_MIRROR
# 
#     def __cinit__(self):
#         self.BORDER_POLICY_IGNORE = pcl_ftr.BORDERPOLICY2_IGNORE
#         self.BORDER_POLICY_MIRROR = pcl_ftr.BORDERPOLICY2_MIRROR
# 
# CythonBorderPolicy_Type = _CythonBorderPolicy_Type()
###


# CythonNormalEstimationMethod
# @cython.internal
# cdef class _CythonNormalEstimationMethod_Type:
#     cdef:
#         readonly int COVARIANCE_MATRIX
#         readonly int AVERAGE_3D_GRADIENT
#         readonly int AVERAGE_DEPTH_CHANGE
#         readonly int SIMPLE_3D_GRADIENT
# 
#     def __cinit__(self):
#         self.COVARIANCE_MATRIX = pcl_ftr.ESTIMATIONMETHOD2_COVARIANCE_MATRIX
#         self.AVERAGE_3D_GRADIENT = pcl_ftr.ESTIMATIONMETHOD2_AVERAGE_3D_GRADIENT
#         self.AVERAGE_DEPTH_CHANGE = pcl_ftr.ESTIMATIONMETHOD2_AVERAGE_DEPTH_CHANGE
#         self.SIMPLE_3D_GRADIENT = pcl_ftr.ESTIMATIONMETHOD2_SIMPLE_3D_GRADIENT
# 
# CythonNormalEstimationMethod_Type = _CythonNormalEstimationMethod_Type()
###

include "pxi/pxiInclude_180.pxi"

include "pxi/PointCloud_PointXYZ_180.pxi"
include "pxi/PointCloud_PointXYZI_180.pxi"
include "pxi/PointCloud_PointXYZRGB_180.pxi"
include "pxi/PointCloud_PointXYZRGBA_180.pxi"
include "pxi/PointCloud_PointWithViewpoint.pxi"
include "pxi/PointCloud_Normal.pxi"
include "pxi/PointCloud_PointNormal.pxi"
include "pxi/PolygonMesh.pxi"

### common ###
def deg2rad(float alpha):
    return pcl_cmn.deg2rad(alpha)

def rad2deg(float alpha):
    return pcl_cmn.rad2deg(alpha)

# cdef double deg2rad(double alpha):
#     return pcl_cmn.rad2deg(alpha)
# 
# cdef double rad2deg(double alpha):
#     return pcl_cmn.rad2deg(alpha)
# 
# cdef float normAngle (float alpha):
#     return pcl_cmn.normAngle(alpha)

