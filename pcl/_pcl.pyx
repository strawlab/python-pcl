# -*- coding: utf-8 -*-
# cython: embedsignature=True

from collections import Sequence
import numbers
import numpy as np

cimport numpy as cnp

cimport pcl_common as pcl_cmn
cimport pcl_defs as cpp
cimport pcl_sample_consensus as pcl_sc
cimport pcl_features as pcl_ftr
cimport pcl_filters as pcl_fil
cimport pcl_range_image as pcl_r_img
# cimport pcl_segmentation as segmentation

cimport cython
# from cython.operator import dereference as deref
from cython.operator cimport dereference as deref, preincrement as inc

from cpython cimport Py_buffer

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

# cimport pcl_segmentation as pclseg

from boost_shared_ptr cimport sp_assign

# Enum Setting
SAC_RANSAC = pcl_sc.SAC_RANSAC
SAC_LMEDS = pcl_sc.SAC_LMEDS
SAC_MSAC = pcl_sc.SAC_MSAC
SAC_RRANSAC = pcl_sc.SAC_RRANSAC
SAC_RMSAC = pcl_sc.SAC_RMSAC
SAC_MLESAC = pcl_sc.SAC_MLESAC
SAC_PROSAC = pcl_sc.SAC_PROSAC

SACMODEL_PLANE = pcl_sc.SACMODEL_PLANE
SACMODEL_LINE = pcl_sc.SACMODEL_LINE
SACMODEL_CIRCLE2D = pcl_sc.SACMODEL_CIRCLE2D
SACMODEL_CIRCLE3D = pcl_sc.SACMODEL_CIRCLE3D
SACMODEL_SPHERE = pcl_sc.SACMODEL_SPHERE
SACMODEL_CYLINDER = pcl_sc.SACMODEL_CYLINDER
SACMODEL_CONE = pcl_sc.SACMODEL_CONE
SACMODEL_TORUS = pcl_sc.SACMODEL_TORUS
SACMODEL_PARALLEL_LINE = pcl_sc.SACMODEL_PARALLEL_LINE
SACMODEL_PERPENDICULAR_PLANE = pcl_sc.SACMODEL_PERPENDICULAR_PLANE
SACMODEL_PARALLEL_LINES = pcl_sc.SACMODEL_PARALLEL_LINES
SACMODEL_NORMAL_PLANE = pcl_sc.SACMODEL_NORMAL_PLANE 
SACMODEL_NORMAL_SPHERE = pcl_sc.SACMODEL_NORMAL_SPHERE
SACMODEL_REGISTRATION = pcl_sc.SACMODEL_REGISTRATION
SACMODEL_PARALLEL_PLANE = pcl_sc.SACMODEL_PARALLEL_PLANE
SACMODEL_NORMAL_PARALLEL_PLANE = pcl_sc.SACMODEL_NORMAL_PARALLEL_PLANE
SACMODEL_STICK = pcl_sc.SACMODEL_STICK

# BORDER_POLICY_IGNORE = pcl_ftr.BORDER_POLICY_IGNORE
# BORDER_POLICY_MIRROR = pcl_ftr.BORDER_POLICY_MIRROR
# COVARIANCE_MATRIX = pcl_ftr.COVARIANCE_MATRIX
# AVERAGE_3D_GRADIENT = pcl_ftr.AVERAGE_3D_GRADIENT
# AVERAGE_DEPTH_CHANGE = pcl_ftr.AVERAGE_DEPTH_CHANGE
# SIMPLE_3D_GRADIENT = pcl_ftr.SIMPLE_3D_GRADIENT
cnp.import_array()

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
# CythonCompareOp
@cython.internal
cdef class _CythonCoordinateFrame_Type:
    cdef:
        readonly int CAMERA_FRAME
        readonly int LASER_FRAME

    def __cinit__(self):
        self.CAMERA_FRAME = pcl_r_img.COORDINATEFRAME_CAMERA
        self.LASER_FRAME = pcl_r_img.COORDINATEFRAME_LASER

CythonCoordinateFrame_Type = _CythonCompareOp_Type()


include "pxi/PointCloudWrapper_PointXYZ.pxi"
include "pxi/PointCloudWrapper_PointXYZI.pxi"
include "pxi/PointCloudWrapper_PointXYZRGB.pxi"
include "pxi/PointCloudWrapper_PointXYZRGBA.pxi"

# include "pxi/OctreePointCloudSearch.pxi"
# include "pxi/OctreePointCloudChangeDetector.pxi"
