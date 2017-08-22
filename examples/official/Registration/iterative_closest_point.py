# -*- coding: utf-8 -*-
# How to use iterative closest point
# http://pointclouds.org/documentation/tutorials/iterative_closest_point.php#iterative-closest-point

import pcl
import random
import numpy as np

# from pcl import icp, gicp, icp_nl

cloud_in = pcl.PointCloud()
cloud_out = pcl.PointCloud()

# Fill in the CloudIn data
# cloud_in->width    = 5;
# cloud_in->height   = 1;
# cloud_in->is_dense = false;
# cloud_in->points.resize (cloud_in->width * cloud_in->height);
# for (size_t i = 0; i < cloud_in->points.size (); ++i)
# {
#   cloud_in->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
#   cloud_in->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
#   cloud_in->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
# }
points_in = np.zeros((5, 3), dtype=np.float32)
RAND_MAX = 1024.0
for i in range(0, 5):
    points_in[i][0] = 1024 * random.random () / RAND_MAX
    points_in[i][1] = 1024 * random.random () / RAND_MAX
    points_in[i][2] = 1024 * random.random () / RAND_MAX

cloud_in.from_array(points_in)

# std::cout << "Saved " << cloud_in->points.size () << " data points to input:" << std::endl;
# for (size_t i = 0; i < cloud_in->points.size (); ++i) std::cout << "    " <<
#   cloud_in->points[i].x << " " << cloud_in->points[i].y << " " <<
#   cloud_in->points[i].z << std::endl;
# *cloud_out = *cloud_in;
print('Saved ' + str(cloud_in.size) + ' data points to input:')
points_out = np.zeros((5, 3), dtype=np.float32)


# std::cout << "size:" << cloud_out->points.size() << std::endl;
# for (size_t i = 0; i < cloud_in->points.size (); ++i)
# cloud_out->points[i].x = cloud_in->points[i].x + 0.7f;

# print('size:' + str(cloud_out.size))
# for i in range(0, cloud_in.size):
print('size:' + str(points_out.size))
for i in range(0, cloud_in.size):
    points_out[i][0] = points_in[i][0] + 0.7
    points_out[i][1] = points_in[i][1]
    points_out[i][2] = points_in[i][2]

cloud_out.from_array(points_out)

# std::cout << "Transformed " << cloud_in->points.size () << " data points:" << std::endl;
print('Transformed ' + str(cloud_in.size) + ' data points:')

# for (size_t i = 0; i < cloud_out->points.size (); ++i)
#   std::cout << "    " << cloud_out->points[i].x << " " << cloud_out->points[i].y << " " << cloud_out->points[i].z << std::endl;
for i in range(0, cloud_out.size):
    print('     ' + str(cloud_out[i][0]) + ' ' + str(cloud_out[i][1]) +  ' '  + str(cloud_out[i][2]) + ' data points:')


# pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
# icp.setInputCloud(cloud_in);
# icp.setInputTarget(cloud_out);
# pcl::PointCloud<pcl::PointXYZ> Final;
# icp.align(Final);
icp = cloud_in.make_IterativeClosestPoint()
# Final = icp.align()
converged, transf, estimate, fitness = icp.icp(cloud_in, cloud_out)

# std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
# std::cout << icp.getFinalTransformation() << std::endl;
# print('has converged:' + str(icp.hasConverged()) + ' score: ' + str(icp.getFitnessScore()) )
# print(str(icp.getFinalTransformation()))
print('has converged:' + str(converged) + ' score: ' + str(fitness) )
print(str(transf))
