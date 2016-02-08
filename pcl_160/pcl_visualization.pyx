#cython: embedsignature=True
#
# Copyright 2015 Netherlands eScience Center

from libcpp cimport bool
from libcpp cimport string

cimport numpy as np
import numpy as np

cimport _pcl
cimport pcl_defs as cpp

np.import_array()

cdef extern from "pcl/visualization/cloud_viewer.h" namespace "pcl":
    cdef cppclass CloudViewer [Target]:

        CloudViewer (string)

		void  showCloud (cpp.PointCloudPtr_t, string = "cloud") 
		bool  wasStopped (int = 1) 

        void align(cpp.PointCloud[Source] &) except +
        double getFitnessScore() except +
        bool hasConverged() except +
        void setInputSource(cpp.PointCloudPtr_t) except +
        void setInputTarget(cpp.PointCloudPtr_t) except +
        void setMaximumIterations(int) except +
