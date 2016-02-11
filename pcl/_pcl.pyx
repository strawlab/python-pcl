#cython: embedsignature=True

from collections import Sequence
import numbers
import numpy as np

cimport numpy as cnp

cimport pcl_defs as cpp
cimport pcl_features as features
# cimport pcl_segmentation as segmentation

cimport cython
# from cython.operator import dereference as deref
from cython.operator cimport dereference as deref, preincrement as inc

from cpython cimport Py_buffer

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

cimport pcl_segmentation as pclseg

from boost_shared_ptr cimport sp_assign

SAC_RANSAC = pclseg.SAC_RANSAC
SAC_LMEDS = pclseg.SAC_LMEDS
SAC_MSAC = pclseg.SAC_MSAC
SAC_RRANSAC = pclseg.SAC_RRANSAC
SAC_RMSAC = pclseg.SAC_RMSAC
SAC_MLESAC = pclseg.SAC_MLESAC
SAC_PROSAC = pclseg.SAC_PROSAC

SACMODEL_PLANE = pclseg.SACMODEL_PLANE
SACMODEL_LINE = pclseg.SACMODEL_LINE
SACMODEL_CIRCLE2D = pclseg.SACMODEL_CIRCLE2D
SACMODEL_CIRCLE3D = pclseg.SACMODEL_CIRCLE3D
SACMODEL_SPHERE = pclseg.SACMODEL_SPHERE
SACMODEL_CYLINDER = pclseg.SACMODEL_CYLINDER
SACMODEL_CONE = pclseg.SACMODEL_CONE
SACMODEL_TORUS = pclseg.SACMODEL_TORUS
SACMODEL_PARALLEL_LINE = pclseg.SACMODEL_PARALLEL_LINE
SACMODEL_PERPENDICULAR_PLANE = pclseg.SACMODEL_PERPENDICULAR_PLANE
SACMODEL_PARALLEL_LINES = pclseg.SACMODEL_PARALLEL_LINES
SACMODEL_NORMAL_PLANE = pclseg.SACMODEL_NORMAL_PLANE 
SACMODEL_NORMAL_SPHERE = pclseg.SACMODEL_NORMAL_SPHERE
SACMODEL_REGISTRATION = pclseg.SACMODEL_REGISTRATION
SACMODEL_PARALLEL_PLANE = pclseg.SACMODEL_PARALLEL_PLANE
SACMODEL_NORMAL_PARALLEL_PLANE = pclseg.SACMODEL_NORMAL_PARALLEL_PLANE
SACMODEL_STICK = pclseg.SACMODEL_STICK

cnp.import_array()

include "pxi/PointCloudWrapper_PointXYZ.pxi"
include "pxi/PointCloudWrapper_PointXYZRGBA.pxi"

include "pxi/OctreePointCloudSearch.pxi"

