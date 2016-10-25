# -*- coding: utf-8 -*-
# cython: embedsignature=True
from libcpp cimport bool

cimport numpy as np
import numpy as np

cimport pcl_defs as cpp
cimport pcl_visualization as pcl_vis
from boost_shared_ptr cimport shared_ptr

# NG : VTK library error
# include "pxi/Visualization/PointCloudColorHandlerCustoms.pxi"
include "pxi/Visualization/Visualization.pxi"
include "pxi/Visualization/PCLHistogramViewing.pxi"
include "pxi/Visualization/PCLVisualizering.pxi"


