# -*- coding: utf-8 -*-
# 
# #include <iostream>
# #include <pcl/io/pcd_io.h>
# #include <pcl/point_types.h>
# 
# int main (int argc, char** argv)
# {
#   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
# 
#   if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd.pcd", *cloud) == -1) //* load the file
#   {
#     PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
#     return (-1);
#   }
#   std::cout << "Loaded "
#             << cloud->width * cloud->height
#             << " data points from test_pcd.pcd with the following fields: "
#             << std::endl;
#   for (size_t i = 0; i < cloud->points.size (); ++i)
#     std::cout << "    " << cloud->points[i].x
#               << " "    << cloud->points[i].y
#               << " "    << cloud->points[i].z << std::endl;
# 
#   return (0);
# }

import pcl

cloud = pcl.load('./examples/official/IO/test_pcd.pcd')

print ('Loaded ' + str(cloud.width * cloud.height) + ' data points from test_pcd.pcd with the following fields: ')
for i in range(0, cloud.size):
    print ('x: '  + str(cloud[i][0]) + ', y : ' + str(cloud[i][1])  + ', z : ' + str(cloud[i][2]))
