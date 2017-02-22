# -*- coding: utf-8 -*-
# Smoothing and normal estimation based on polynomial reconstruction
# http://pointclouds.org/documentation/tutorials/resampling.php#moving-least-squares

import numpy as np
import pcl
import random

# // Load input file into a PointCloud<T> with an appropriate type
# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
# // Load bun0.pcd -- should be available with the PCL archive in test 
# pcl::io::loadPCDFile ("bun0.pcd", *cloud);
cloud = pcl.load('./examples/official/Surface/bun0.pcd')
print('cloud(size) = ' + str(cloud.size))

# // Create a KD-Tree
# pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
tree = cloud.make_kdtree()
# tree = cloud.make_kdtree_flann()
# blankCloud = pcl.PointCloud()
# tree = blankCloud.make_kdtree()

# // Output has the PointNormal type in order to store the normals calculated by MLS
# pcl::PointCloud<pcl::PointNormal> mls_points;
# mls_points = pcl.PointCloudNormal()
# // Init object (second point type is for the normals, even if unused)
# pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
# mls.setComputeNormals (true);
# 
# // Set parameters
# mls.setInputCloud (cloud);
# mls.setPolynomialFit (true);
# mls.setSearchMethod (tree);
# mls.setSearchRadius (0.03);
#
# // Reconstruct
# mls.process (mls_points);
mls = cloud.make_moving_least_squares()
# print('make_moving_least_squares')
mls.set_Compute_Normals (True)
mls.set_polynomial_fit (True)
mls.set_Search_Method (tree)
mls.set_search_radius (0.03)
print('set parameters')
mls_points = mls.process ()

# Save output
# pcl::io::savePCDFile ("bun0-mls.pcd", mls_points);
pcl.save_PointNormal(mls_points, 'bun0-mls.pcd')
