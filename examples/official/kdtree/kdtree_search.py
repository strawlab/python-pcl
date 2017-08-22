# -*- coding: utf-8 -*-
# http://pointclouds.org/documentation/tutorials/kdtree_search.php#kdtree-search

import numpy as np
import pcl
import random

# int main (int argc, char** argv)
# srand (time (NULL));
# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
cloud = pcl.PointCloud()

# // Generate pointcloud data
# cloud->width = 1000;
# cloud->height = 1;
# cloud->points.resize (cloud->width * cloud->height);
# 
# for (size_t i = 0; i < cloud->points.size (); ++i)
# {
# cloud->points[i].x = 1024.0f * rand () / (RAND_MAX + 1.0f);
# cloud->points[i].y = 1024.0f * rand () / (RAND_MAX + 1.0f);
# cloud->points[i].z = 1024.0f * rand () / (RAND_MAX + 1.0f);
# }
points = np.zeros((1000, 3), dtype=np.float32)
RAND_MAX = 1024
for i in range(0, 1000):
    points[i][0] = 1024 * random.random () / (RAND_MAX + 1.0)
    points[i][1] = 1024 * random.random () / (RAND_MAX + 1.0)
    points[i][2] = 1024 * random.random () / (RAND_MAX + 1.0)

cloud.from_array(points)


# pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
# kdtree.setInputCloud (cloud);
kdtree = cloud.make_kdtree_flann()

# pcl::PointXYZ searchPoint;
# 
# searchPoint.x = 1024.0f * rand () / (RAND_MAX + 1.0f);
# searchPoint.y = 1024.0f * rand () / (RAND_MAX + 1.0f);
# searchPoint.z = 1024.0f * rand () / (RAND_MAX + 1.0f);
searchPoint = pcl.PointCloud()
searchPoints = np.zeros((1, 3), dtype=np.float32)
searchPoints[0][0] = 1024 * random.random () / (RAND_MAX + 1.0)
searchPoints[0][1] = 1024 * random.random () / (RAND_MAX + 1.0)
searchPoints[0][2] = 1024 * random.random () / (RAND_MAX + 1.0)

searchPoint.from_array(searchPoints)

# // K nearest neighbor search
# int K = 10;
K = 10

# std::vector<int> pointIdxNKNSearch(K);
# std::vector<float> pointNKNSquaredDistance(K);
# 
# std::cout << "K nearest neighbor search at (" << searchPoint.x 
#         << " " << searchPoint.y 
#         << " " << searchPoint.z
#         << ") with K=" << K << std::endl;
# print ('K nearest neighbor search at (' + searchPoint[0][0] + ' ' + searchPoint[0][1] + ' ' + searchPoint[0][2] + ') with K=' + str(K))
print ('K nearest neighbor search at (' + str(searchPoint[0][0]) + ' ' + str(searchPoint[0][1]) + ' ' + str(searchPoint[0][2]) + ') with K=' + str(K))

# if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
# {
# for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
#   std::cout << "    "  <<   cloud->points[ pointIdxNKNSearch[i] ].x 
#             << " " << cloud->points[ pointIdxNKNSearch[i] ].y 
#             << " " << cloud->points[ pointIdxNKNSearch[i] ].z 
#             << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
# }
[ind, sqdist] = kdtree.nearest_k_search_for_cloud(searchPoint, K)
# if nearest_k_search_for_cloud
for i in range(0, ind.size):
    print ('(' + str(cloud[ind[0][i]][0]) + ' ' + str(cloud[ind[0][i]][1]) + ' ' + str(cloud[ind[0][i]][2]) + ' (squared distance: ' + str(sqdist[0][i]) + ')')



# Neighbors within radius search
# std::vector<int> pointIdxRadiusSearch;
# std::vector<float> pointRadiusSquaredDistance;
# float radius = 256.0f * rand () / (RAND_MAX + 1.0f);
# std::cout << "Neighbors within radius search at (" << searchPoint.x 
#         << " " << searchPoint.y 
#       << " " << searchPoint.z
#        << ") with radius=" << radius << std::endl;
radius = 256.0 * random.random () / (RAND_MAX + 1.0)
print ('Neighbors within radius search at (' + str(searchPoint[0][0]) + ' ' + str(searchPoint[0][1]) + ' ' + str(searchPoint[0][2]) + ') with radius=' + str(radius))

# if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
# {
# for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
#   std::cout << "    "  <<   cloud->points[ pointIdxRadiusSearch[i] ].x 
#             << " " << cloud->points[ pointIdxRadiusSearch[i] ].y 
#             << " " << cloud->points[ pointIdxRadiusSearch[i] ].z 
#             << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
# }
# NotImplement radiusSearch
[ind, sqdist] = kdtree.radius_search_for_cloud (searchPoint, radius)
for i in range(0, ind.size):
    print ('(' + str(cloud[ind[0][i]][0]) + ' ' + str(cloud[ind[0][i]][1]) + ' ' + str(cloud[ind[0][i]][2]) + ' (squared distance: ' + str(sqdist[0][i]) + ')')

