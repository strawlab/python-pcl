# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import pcl

import pcl.pcl_visualization
# from pcl.pcl_registration_160 import icp, gicp, icp_nl

# cloud = pcl.load('G:\\tmp\\PCL\\extendlibrary\\python-pcl\\examples\\pcldata\\tutorials\\table_scene_mug_stereo_textured.pcd')
# cloud = pcl.load_XYZI('G:\\tmp\\PCL\\extendlibrary\\python-pcl\\examples\\pcldata\\tutorials\\table_scene_mug_stereo_textured.pcd')
cloud = pcl.load_XYZRGB('G:\\tmp\\PCL\\extendlibrary\\python-pcl\\examples\\pcldata\\tutorials\\table_scene_mug_stereo_textured.pcd')
# cloud = pcl.load_XYZRGBA('G:\\tmp\\PCL\\extendlibrary\\python-pcl\\examples\\pcldata\\tutorials\\table_scene_mug_stereo_textured.pcd')
# view set
visual = pcl.pcl_visualization.Visualization()

# PointXYZ
# visual.ShowMonochromeCloud(cloud)
# visual.ShowGrayCloud(cloud)
visual.ShowColorCloud(cloud)
# visual.ShowColorACloud(cloud)

while True:
    visual.WasStopped()
end

