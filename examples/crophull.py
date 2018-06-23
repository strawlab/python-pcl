# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import pcl

# http://www.pcl-users.org/CropHull-filter-question-td4030345.html
datacloud = pcl.load('./examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')

print(datacloud)

filterCloud = pcl.PointCloud()
vt = pcl.Vertices()

# // inside point
# cloud->push_back(pcl::PointXYZ(M_PI * 0.3, M_PI * 0.3, 0));
# // hull points
# cloud->push_back(pcl::PointXYZ(0,0,0));
# cloud->push_back(pcl::PointXYZ(M_PI,0,0));
# cloud->push_back(pcl::PointXYZ(M_PI,M_PI*0.5,0));
# cloud->push_back(pcl::PointXYZ(0,M_PI*0.5,0));
# cloud->push_back(pcl::PointXYZ(0,0,0));
# // outside point
# cloud->push_back(pcl::PointXYZ(-M_PI * 0.3, -M_PI * 0.3, 0));

points_2 = np.array([
                        [1 * 0.3, 1 * 0.3, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [1, 1 * 0.5, 0],
                        [0, 1 * 0.5, 0],
                        [0, 0, 0],
                        [-1 * 0.3 , -1 * 0.3, 0]
                    ], dtype=np.float32)
filterCloud.from_array(points_2)
print(filterCloud)

vertices_point_1 = np.array([1, 2, 3, 4, 5], dtype=np.int)
vt.from_array(vertices_point_1)

# print(vt)
# vt.vertices.push_back(1)
# vt.vertices.push_back(2)
# vt.vertices.push_back(3)
# vt.vertices.push_back(4)
# vt.vertices.push_back(5)
# vertices = vector[pcl.Vertices]
# vertices.push_back(vt)

outputCloud = pcl.PointCloud()
# crophull = pcl.CropHull()
# crophull.setInputCloud(datacloud)
crophull = datacloud.make_crophull()
# crophull.setHullIndices(vertices)
# crophull.setHullIndices(vt)
# crophull.setHullCloud(filterCloud)
# crophull.setDim(2)
# crophull.setCropOutside(false)
crophull.SetParameter(filterCloud, vt)

# indices = vector[int]
# cropHull.filter(indices);
# outputCloud = cropHull.filter();
# print("before: " + outputCloud)
crophull.Filtering(outputCloud)
print(outputCloud)


# Viewer
# // pcl::visualization::CloudViewer viewer ("Cluster viewer");
# // viewer.showCloud(colored_cloud);

# pcl.visualization.CloudViewer

# Write Point
# pcl::PCDWriter writer;
# std::stringstream ss;
# ss << "min_cut_seg" << ".pcd";
# // writer.write<pcl::PointXYZRGB> (ss.str (), *cloud, false);
# pcl::io::savePCDFile(ss.str(), *outputCloud, false);

