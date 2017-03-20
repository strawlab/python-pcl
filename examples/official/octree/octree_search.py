# -*- coding: utf-8 -*-
# Spatial Partitioning and Search Operations with Octrees
# http://pointclouds.org/documentation/tutorials/octree.php#octree-search

import pcl
import numpy as np
import random

# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
cloud = pcl.PointCloud()

##
# // Generate pointcloud data
# cloud->width = 1000;
# cloud->height = 1;
# cloud->points.resize (cloud->width * cloud->height);
# 
# for (size_t i = 0; i < cloud->points.size (); ++i)
# {
#     cloud->points[i].x = 1024.0f * rand () / (RAND_MAX + 1.0f);
#     cloud->points[i].y = 1024.0f * rand () / (RAND_MAX + 1.0f);
#     cloud->points[i].z = 1024.0f * rand () / (RAND_MAX + 1.0f);
# }
# 
points = np.zeros((1000, 3), dtype=np.float32)
RAND_MAX = 1024.0
for i in range(0, 1000):
    points[i][0] = 1024 * random.random () / RAND_MAX
    points[i][1] = 1024 * random.random () / RAND_MAX
    points[i][2] = 1024 * random.random () / RAND_MAX

cloud.from_array(points)

# pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree (resolution);
# octree.setInputCloud (cloud);
# octree.addPointsFromInputCloud ();

# resolution = 128.0
# x,y,z Area Filter
resolution = 0.2
octree = cloud.make_octreeSearch(resolution)
octree.add_points_from_input_cloud()

# pcl::PointXYZ searchPoint;
# searchPoint.x = 1024.0f * rand () / (RAND_MAX + 1.0f);
# searchPoint.y = 1024.0f * rand () / (RAND_MAX + 1.0f);
# searchPoint.z = 1024.0f * rand () / (RAND_MAX + 1.0f);
searchPoint = pcl.PointCloud()
searchPoints = np.zeros((1, 3), dtype=np.float32)
# searchPoints[0][0] = 1024 * random.random () / (RAND_MAX + 1.0)
# searchPoints[0][1] = 1024 * random.random () / (RAND_MAX + 1.0)
# searchPoints[0][2] = 1024 * random.random () / (RAND_MAX + 1.0)
searchPoints[0][0] = 1024 * random.random () / (RAND_MAX + 1.0)
searchPoints[0][1] = 1024 * random.random () / (RAND_MAX + 1.0)
searchPoints[0][2] = 1024 * random.random () / (RAND_MAX + 1.0)

searchPoint.from_array(searchPoints)

##
# // Neighbors within voxel search
# std::vector<int> pointIdxVec;
# 
#   if (octree.voxelSearch (searchPoint, pointIdxVec))
#   {
#     std::cout << "Neighbors within voxel search at (" << searchPoint.x 
#      << " " << searchPoint.y 
#      << " " << searchPoint.z << ")" 
#      << std::endl;
#               
#     for (size_t i = 0; i < pointIdxVec.size (); ++i)
#    std::cout << "    " << cloud->points[pointIdxVec[i]].x 
#        << " " << cloud->points[pointIdxVec[i]].y 
#        << " " << cloud->points[pointIdxVec[i]].z << std::endl;
#   }
ind = octree.VoxelSearch(searchPoint)

print ('Neighbors within voxel search at (' + str(searchPoint[0][0]) + ' ' + str(searchPoint[0][1]) + ' ' + str(searchPoint[0][2]) + ')')
# for i in range(0, ind.size):
for i in range(0, ind.size):
    print ('index = ' + str(ind[i]))
    print ('(' + str(cloud[ind[i]][0]) + ' ' + str(cloud[ind[i]][1]) + ' ' + str(cloud[ind[i]][2]))

##
# // K nearest neighbor search
# std::vector<int> pointIdxNKNSearch;
# std::vector<float> pointNKNSquaredDistance;
# 
# std::cout << "K nearest neighbor search at (" << searchPoint.x 
#           << " " << searchPoint.y 
#           << " " << searchPoint.z
#           << ") with K=" << K << std::endl;
K = 10
print ('K nearest neighbor search at (' + str(searchPoint[0][0]) + ' ' + str(searchPoint[0][1]) + ' ' + str(searchPoint[0][2]) + ') with K=' + str(K))

# if (octree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
# {
#   for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
#     std::cout << "    "  <<   cloud->points[ pointIdxNKNSearch[i] ].x 
#               << " " << cloud->points[ pointIdxNKNSearch[i] ].y 
#               << " " << cloud->points[ pointIdxNKNSearch[i] ].z 
#               << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
# }
# // Neighbors within radius search
[ind, sqdist] = octree.nearest_k_search_for_cloud(searchPoint, K)
# if nearest_k_search_for_cloud
for i in range(0, ind.size):
    print ('(' + str(cloud[ind[0][i]][0]) + ' ' + str(cloud[ind[0][i]][1]) + ' ' + str(cloud[ind[0][i]][2]) + ' (squared distance: ' + str(sqdist[0][i]) + ')')

## 
# std::vector<int> pointIdxRadiusSearch;
# std::vector<float> pointRadiusSquaredDistance;
# float radius = 256.0f * rand () / (RAND_MAX + 1.0f);
# std::cout << "Neighbors within radius search at (" << searchPoint.x 
#     << " " << searchPoint.y 
#     << " " << searchPoint.z
#     << ") with radius=" << radius << std::endl;
#
radius = 256.0 * random.random () / (RAND_MAX + 1.0)
print ('Neighbors within radius search at (' + str(searchPoint[0][0]) + ' ' + str(searchPoint[0][1]) + ' ' + str(searchPoint[0][2]) + ') with radius=' + str(radius))

# if (octree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
# {
#   for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
#        std::cout << "    "  <<   cloud->points[ pointIdxRadiusSearch[i] ].x 
#                  << " " << cloud->points[ pointIdxRadiusSearch[i] ].y 
#                  << " " << cloud->points[ pointIdxRadiusSearch[i] ].z 
#                  << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
# }
###
# [ind, sqdist] = octree.radius_search_for_cloud (searchPoint, radius)
# Exception ignored in: 'pcl._pcl.to_point_t'
# [ind, sqdist] = octree.radius_search (searchPoint, radius, 10)
searchPoints = (searchPoint[0][0], searchPoint[0][1], searchPoint[0][2])
[ind, sqdist] = octree.radius_search (searchPoints, radius, 10)

# Function radius_search
for i in range(0, ind.size):
   print ('(' + str(cloud[ind[i]][0]) + ' ' + str(cloud[ind[i]][1]) + ' ' + str(cloud[ind[i]][2]) + ' (squared distance: ' + str(sqdist[i]) + ')')

# Function radius_search_for_cloud
# for i in range(0, ind.size):
#    print ('(' + str(cloud[ind[0][i]][0]) + ' ' + str(cloud[ind[0][i]][1]) + ' ' + str(cloud[ind[0][i]][2]) + ' (squared distance: ' + str(sqdist[0][i]) + ')')

