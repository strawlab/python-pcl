# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import pcl

import pcl.pcl_visualization
# from pcl.pcl_registration import icp, gicp, icp_nl


def main():
    cloud = pcl.load_XYZRGB('./examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')
    visual = pcl.pcl_visualization.CloudViewing()

    # PointXYZ
    # visual.ShowMonochromeCloud(cloud)

    # visual.ShowGrayCloud(cloud, b'cloud')
    visual.ShowColorCloud(cloud, b'cloud')
    # visual.ShowColorACloud(cloud, b'cloud')

    v = True
    while v:
        v=not(visual.WasStopped())


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()