# -*- coding: utf-8 -*-
# cython: embedsignature=True
from libcpp cimport bool

from collections import Sequence
import numbers
import numpy as np
cimport numpy as cnp

cimport pcl_defs as cpp
cimport pcl_visualization as pcl_vis

cimport cython
# from cython.operator import dereference as deref
from cython.operator cimport dereference as deref, preincrement as inc

from cpython cimport Py_buffer

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

from boost_shared_ptr cimport sp_assign

cnp.import_array()

### Enum ###

### Enum Setting ###
# pcl_visualization.pxd
# cdef enum RenderingProperties:
# Re: [Cython] resolving name conflict -- does not work for enums !? 
# https://www.mail-archive.com/cython-dev@codespeak.net/msg02494.html
PCLVISUALIZER_POINT_SIZE = pcl_vis.PCL_VISUALIZER_POINT_SIZE
PCLVISUALIZER_OPACITY = pcl_vis.PCL_VISUALIZER_OPACITY
PCLVISUALIZER_LINE_WIDTH = pcl_vis.PCL_VISUALIZER_LINE_WIDTH
PCLVISUALIZER_FONT_SIZE = pcl_vis.PCL_VISUALIZER_FONT_SIZE
PCLVISUALIZER_COLOR = pcl_vis.PCL_VISUALIZER_COLOR
PCLVISUALIZER_REPRESENTATION = pcl_vis.PCL_VISUALIZER_REPRESENTATION
PCLVISUALIZER_IMMEDIATE_RENDERING = pcl_vis.PCL_VISUALIZER_IMMEDIATE_RENDERING

# cdef enum RenderingRepresentationProperties:
PCLVISUALIZER_REPRESENTATION_POINTS = pcl_vis.PCL_VISUALIZER_REPRESENTATION_POINTS
PCLVISUALIZER_REPRESENTATION_WIREFRAME = pcl_vis.PCL_VISUALIZER_REPRESENTATION_WIREFRAME
PCLVISUALIZER_REPRESENTATION_SURFACE = pcl_vis.PCL_VISUALIZER_REPRESENTATION_SURFACE

### Enum Setting(define Class InternalType) ###

###

# PointCloud/Common
# NG
# include "pxi/PointCloudWrapper_PointXYZ.pxi"
# include "pxi/PointCloudWrapper_PointXYZI.pxi"
# include "pxi/Common/RangeImage/RangeImages.pxi"

# VTK
include "pxi/Visualization/CloudViewing.pxi"
include "pxi/Visualization/PCLVisualizering.pxi"
include "pxi/Visualization/PCLHistogramViewing.pxi"

# VTK - Handler
include "pxi/Visualization/Handler/PointCloudColorHandleringCustom.pxi"
include "pxi/Visualization/Handler/PointCloudColorHandleringGenericField.pxi"
include "pxi/Visualization/Handler/PointCloudColorHandleringHSVField.pxi"
include "pxi/Visualization/Handler/PointCloudColorHandleringRandom.pxi"
include "pxi/Visualization/Handler/PointCloudColorHandleringRGBField.pxi"
include "pxi/Visualization/Handler/PointCloudGeometryHandleringSurfaceNormal.pxi"
include "pxi/Visualization/Handler/PointCloudGeometryHandleringXYZ.pxi"

# NG(vtk Link Error)
# include "pxi/Visualization/RangeImageVisualization.pxi"
# include "pxi/Visualization/PCLHistogramViewing.pxi"
