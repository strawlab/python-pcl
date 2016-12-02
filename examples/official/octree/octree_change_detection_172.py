# -*- coding: utf-8 -*-
# Spatial change detection on unorganized point cloud data
# http://pointclouds.org/documentation/tutorials/octree_change.php#octree-change-detection

import pcl
import numpy as np
import random

# // Octree resolution - side length of octree voxels
resolution = 32.0

# // Instantiate octree-based point cloud change detection class
# pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZ> octree (resolution);
# pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA (new pcl::PointCloud<pcl::PointXYZ> );
# // Generate pointcloud data for cloudA
# cloudA->width = 128;
# cloudA->height = 1;
# cloudA->points.resize (cloudA->width * cloudA->height);
# for (size_t i = 0; i < cloudA->points.size (); ++i)
# {
#     cloudA->points[i].x = 64.0f * rand () / (RAND_MAX + 1.0f);
#     cloudA->points[i].y = 64.0f * rand () / (RAND_MAX + 1.0f);
#     cloudA->points[i].z = 64.0f * rand () / (RAND_MAX + 1.0f);
# }
# 
# // Add points from cloudA to octree
# octree.setInputCloud (cloudA);
# octree.addPointsFromInputCloud ();
cloudA = pcl.PointCloud()

points = np.zeros((128, 3), dtype=np.float32)
RAND_MAX = 1024.0
for i in range(0, 5):
    points[i][0] = 64.0 * random.random () / RAND_MAX
    points[i][1] = 64.0 * random.random () / RAND_MAX
    points[i][2] = 64.0 * random.random () / RAND_MAX

cloudA.from_array(points)
octree = cloudA.make_octreeChangeDetector(resolution)
octree.add_points_from_input_cloud ()

# // Switch octree buffers: This resets octree but keeps previous tree structure in memory.
# octree.switchBuffers ();
octree.switchBuffers ()

# pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB (new pcl::PointCloud<pcl::PointXYZ> );
cloudB = pcl.PointCloud()

# // Generate pointcloud data for cloudB 
# cloudB->width = 128;
# cloudB->height = 1;
# cloudB->points.resize (cloudB->width * cloudB->height);
# 
# for (size_t i = 0; i < cloudB->points.size (); ++i)
# {
#   cloudB->points[i].x = 64.0f * rand () / (RAND_MAX + 1.0f);
#   cloudB->points[i].y = 64.0f * rand () / (RAND_MAX + 1.0f);
#   cloudB->points[i].z = 64.0f * rand () / (RAND_MAX + 1.0f);
# }
# // Add points from cloudB to octree
#  octree.setInputCloud (cloudB);
#  octree.addPointsFromInputCloud ();
points2 = np.zeros((128, 3), dtype=np.float32)
for i in range(0, 5):
    points2[i][0] = 64.0 * random.random () / RAND_MAX
    points2[i][1] = 64.0 * random.random () / RAND_MAX
    points2[i][2] = 64.0 * random.random () / RAND_MAX

cloudB.from_array(points2)

octree.setInputCloud (cloudB)
octree.addPointsFromInputCloud ()

# std::vector<int> newPointIdxVector;
# // Get vector of point indices from octree voxels which did not exist in previous buffer
# octree.getPointIndicesFromNewVoxels (newPointIdxVector);
# // Output points
# std::cout << "Output from getPointIndicesFromNewVoxels:" << std::endl;
# for (size_t i = 0; i < newPointIdxVector.size (); ++i)
#   std::cout << i << "# Index:" << newPointIdxVector[i]
#             << "  Point:" << cloudB->points[newPointIdxVector[i]].x << " "
#             << cloudB->points[newPointIdxVector[i]].y << " "
#             << cloudB->points[newPointIdxVector[i]].z << std::endl;
[ind, sqdist] = kdtree.radius_search_for_cloud (searchPoint, radius)
for i in range(0, ind.size):
    print ('(' + str(i) + '# Index:' + str(ind[0][i]) + '  Point:' + str(cloudB[ind[0][i]][0]) + ' ' + str(cloudB[ind[0][i]][1]) + ' ' + str(cloudB[ind[0][i]][2]) + ' (squared distance: ' + str(sqdist[0][i]) + ')')

