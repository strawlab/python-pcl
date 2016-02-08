#cython: embedsignature=True
#
# Copyright 2014 Netherlands eScience Center

from libcpp cimport bool

cimport numpy as np
import numpy as np

# cimport _pcl
cimport pcl_defs as cpp

cdef extern from "pcl/registration/icp.h" namespace "pcl":
    cdef cppclass IterativeClosestPoint[Source, Target]:

        IterativeClosestPoint () except +

