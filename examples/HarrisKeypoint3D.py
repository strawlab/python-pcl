# -*- coding: utf-8 -*-
# http://virtuemarket-lab.blogspot.jp/2015/03/harris.html

from __future__ import print_function

import numpy as np
import pcl

import pcl.pcl_visualization

# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
# pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *cloud);
# cloud = pcl.load_XYZRGB('./examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')
cloud = pcl.load('./examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')

# pcl::HarrisKeypoint3D<pcl::PointXYZ,pcl::PointXYZI> detector;
# detector.setNonMaxSupression (true);
# detector.setRadius (0.01);
# //detector.setRadiusSearch (100);
# detector.setInputCloud(cloud);
detector = cloud.make_HarrisKeypoint3D()

# pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());
# detector.compute(*keypoints);
keypoints = detector.compute()

# std::cout << "keypoints detected: " << keypoints->size() << std::endl;

# pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints3D(new pcl::PointCloud<pcl::PointXYZ>());
# pcl::PointXYZ tmp;
# double max = 0,min=0;
# 
# for(pcl::PointCloud<pcl::PointXYZI>::iterator i = keypoints->begin(); i!= keypoints->end(); i++){
#     tmp = pcl::PointXYZ((*i).x,(*i).y,(*i).z);
#     if ((*i).intensity>max ){
#         std::cout << (*i) << " coords: " << (*i).x << ";" << (*i).y << ";" << (*i).z << std::endl;
#         max = (*i).intensity;
#     }
#     if ((*i).intensity<min){
#         min = (*i).intensity;
#     }
#     keypoints3D->push_back(tmp);
# }
# 
# std::cout << "maximal responce: "<< max << " min responce:  "<< min<<std::endl;
# 
# //show point cloud
# pcl::visualization::PCLVisualizer viewer ("3D Viewer");
viewer = pcl.pcl_visualization.PCLVisualizering(b"3D Viewer")

# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pccolor(cloud, 255, 255, 255);
# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> kpcolor(keypoints3D, 255, 0, 0);
pccolor = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 255, 255, 255)
# kpcolor = pcl.pcl_visualization.PointCloudColorHandleringCustom(keypoints3D, 255, 0, 0)
kpcolor = pcl.pcl_visualization.PointCloudColorHandleringCustom(keypoints, 255, 0, 0)


viewer.addPointCloud(cloud,pccolor,"testimg.png");
viewer.addPointCloud(keypoints3D,kpcolor,"keypoints.png");
# viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints.png");


while True:
    # viewer.wasStopped()
    viewer.spinOnce()
    # pcl_sleep (0.01)
end
