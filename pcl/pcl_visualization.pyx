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

## Enum Setting
# common.h
# cdef extern from "pcl/visualization/common/common.h" namespace "pcl":
# cdef enum RenderingProperties:
# PCL_VISUALIZER_POINT_SIZE = pcl_vis.PCL_VISUALIZER_POINT_SIZE
# PCL_VISUALIZER_OPACITY = pcl_vis.PCL_VISUALIZER_OPACITY
# PCL_VISUALIZER_LINE_WIDTH = pcl_vis.PCL_VISUALIZER_LINE_WIDTH
# PCL_VISUALIZER_FONT_SIZE = pcl_vis.PCL_VISUALIZER_FONT_SIZE
# PCL_VISUALIZER_COLOR = pcl_vis.PCL_VISUALIZER_COLOR
# PCL_VISUALIZER_REPRESENTATION = pcl_vis.PCL_VISUALIZER_REPRESENTATION
# PCL_VISUALIZER_IMMEDIATE_RENDERING = pcl_vis.PCL_VISUALIZER_IMMEDIATE_RENDERING

# cdef enum RenderingRepresentationProperties:
# PCL_VISUALIZER_REPRESENTATION_POINTS = pcl_vis.PCL_VISUALIZER_REPRESENTATION_POINTS
# PCL_VISUALIZER_REPRESENTATION_WIREFRAME = pcl_vis.PCL_VISUALIZER_REPRESENTATION_WIREFRAME
# PCL_VISUALIZER_REPRESENTATION_SURFACE = pcl_vis.PCL_VISUALIZER_REPRESENTATION_SURFACE

### Enum Setting(define Class InternalType) ###

# NG : VTK library error
include "pxi/Visualization/PointCloudColorHandlerCustoms.pxi"
include "pxi/Visualization/Visualization.pxi"
include "pxi/Visualization/PCLHistogramViewing.pxi"
include "pxi/Visualization/PCLVisualizering.pxi"

