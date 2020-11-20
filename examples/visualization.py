# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import pcl

import pcl.pcl_visualization
# from pcl.pcl_registration import icp, gicp, icp_nl


def main():
    # cloud = pcl.load_XYZRGB(
    #     './examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')
    cloud = pcl.load("Tile_173078_LD_010_017_L22.obj")
    # Centred the data
    centred = cloud - np.mean(cloud, 0)
    # print(centred)
    ptcloud_centred = pcl.PointCloud()
    ptcloud_centred.from_array(centred)
    # ptcloud_centred = pcl.load("Tile_173078_LD_010_017_L22.obj")

    visual = pcl.pcl_visualization.CloudViewing()

    # PointXYZ
    visual.ShowMonochromeCloud(ptcloud_centred, b'cloud')
    # visual.ShowGrayCloud(ptcloud_centred, b'cloud')
    # visual.ShowColorCloud(ptcloud_centred, b'cloud')
    # visual.ShowColorACloud(ptcloud_centred, b'cloud')

    v = True
    while v:
        v = not(visual.WasStopped())


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
