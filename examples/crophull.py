from __future__ import print_function

import numpy as np
import pcl
from numpy import array

# http://www.pcl-users.org/CropHull-filter-question-td4030345.html
# pcl::PointCloud <pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZ>);
# datacloud = pcl.load("pcldata\\tutorials\\table_scene_mug_stereo_textured.pcd")
datacloud = pcl.load('G:\\tmp\\PCL\\extendlibrary\\python-pcl\\examples\\pcldata\\tutorials\\table_scene_mug_stereo_textured.pcd')

print(datacloud)

filterCloud = pcl.PointCloud()
vt = pcl.Vertices()
# vertices = vector[pcl.Vertices]

# // inside point
# cloud->push_back(pcl::PointXYZ(M_PI * 0.3, M_PI * 0.3, 0));
# // 
# cloud->push_back(pcl::PointXYZ(0,0,0));
# cloud->push_back(pcl::PointXYZ(M_PI,0,0));
# cloud->push_back(pcl::PointXYZ(M_PI,M_PI*0.5,0));
# cloud->push_back(pcl::PointXYZ(0,M_PI*0.5,0));
# cloud->push_back(pcl::PointXYZ(0,0,0));
# // outside point
# cloud->push_back(pcl::PointXYZ(-M_PI*0.3,-M_PI*0.3,0));

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
# 
vt.vertices.push_back(1)
vt.vertices.push_back(2)
vt.vertices.push_back(3)
vt.vertices.push_back(4)
vt.vertices.push_back(5)
vertices.push_back(vt)

outputCloud = pcl.PointCloud()
crophull = pcl.CropHull()
crophull.setInputCloud(datacloud)
crophull.setHullIndices(vertices)
crophull.setHullCloud(filterCloud)
crophull.setDim(2)
crophull.setCropOutside(false)

indices = vector[int]
cropHull.filter(indices);
cropHull.filter(*outputCloud);

# Viewer
# // pcl::PointCloud <pcl::PointXYZRGB>::Ptr filtered_cloud = cloud (new pcl::PointCloud <pcl::PointXYZ>);
# // pcl::visualization::CloudViewer viewer ("Cluster viewer");
# // viewer.showCloud(colored_cloud);

# Write Point
# pcl::PCDWriter writer;
# std::stringstream ss;
# ss << "min_cut_seg" << ".pcd";
# // writer.write<pcl::PointXYZRGB> (ss.str (), *cloud, false);
# pcl::io::savePCDFile(ss.str(), *outputCloud, false);

