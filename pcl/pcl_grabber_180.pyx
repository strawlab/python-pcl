# -*- coding: utf-8 -*-
# cython: embedsignature=True
#
# Copyright 2014 Netherlands eScience Center

from libcpp cimport bool

cimport numpy as np

cimport _pcl
cimport pcl_defs as cpp
cimport pcl_grabber_defs_180 as pcl_grb
from boost_shared_ptr cimport shared_ptr


### Enum Setting ###
# pcl_visualization_defs.pxd
# cdef enum RenderingProperties:
# Re: [Cython] resolving name conflict -- does not work for enums !? 
# https://www.mail-archive.com/cython-dev@codespeak.net/msg02494.html
# PCLVISUALIZER_POINT_SIZE = pcl_grb.PCL_VISUALIZER_POINT_SIZE
# PCLVISUALIZER_OPACITY = pcl_grb.PCL_VISUALIZER_OPACITY
# PCLVISUALIZER_LINE_WIDTH = pcl_grb.PCL_VISUALIZER_LINE_WIDTH
# PCLVISUALIZER_FONT_SIZE = pcl_grb.PCL_VISUALIZER_FONT_SIZE
# PCLVISUALIZER_COLOR = pcl_grb.PCL_VISUALIZER_COLOR
# PCLVISUALIZER_REPRESENTATION = pcl_grb.PCL_VISUALIZER_REPRESENTATION
# PCLVISUALIZER_IMMEDIATE_RENDERING = pcl_grb.PCL_VISUALIZER_IMMEDIATE_RENDERING
### Enum Setting(define Class InternalType) ###

# Grabber
# include "pxi/Grabber/ONIGrabber.pxi"
# include "pxi/Grabber/OpenNIGrabber.pxi"

