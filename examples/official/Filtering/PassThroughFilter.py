# -*- coding: utf-8 -*-
# Filtering a PointCloud using a PassThrough filter
# http://pointclouds.org/documentation/tutorials/passthrough.php#passthrough


#
import numpy as np
import pcl
import random

# int main (int argc, char** argv)
# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
cloud = pcl.PointCloud()
# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
# cloud_filtered = pcl.PointCloud()

# // Fill in the cloud data
# cloud->width  = 5;
# cloud->height = 1;
# cloud->points.resize (cloud->width * cloud->height);
points = np.zeros((5, 3), dtype=np.float32)
RAND_MAX = 1024

for i in range(0, 5):
    points[i][0] = 1024 * random.random () / RAND_MAX
    points[i][1] = 1024 * random.random () / RAND_MAX
    points[i][2] = 1024 * random.random () / RAND_MAX

cloud.from_array(points)

# std::cerr << "Cloud before filtering: " << std::endl;
# for (size_t i = 0; i < cloud->points.size (); ++i)
# for i in range(0, 5):
# std::cerr << "    " << cloud->points[i].x << " " 
#                     << cloud->points[i].y << " " 
#                     << cloud->points[i].z << std::endl;
print ('Cloud before filtering: ')
for i in range(0, cloud.size):
    print ('x: '  + str(cloud[i][0]) + ', y : ' + str(cloud[i][1])  + ', z : ' + str(cloud[i][2]))

# Create the filtering object
# pcl::PassThrough<pcl::PointXYZ> pass;
# pass.setInputCloud (cloud);
# define pass , NG
passthrough = cloud.make_passthrough_filter()
passthrough.set_filter_field_name ("z")
passthrough.set_filter_limits (0.0, 0.5)
# //pass.setFilterLimitsNegative (true)
cloud_filtered = passthrough.filter ()

# std::cerr << "Cloud after filtering: " << std::endl;
# for (size_t i = 0; i < cloud_filtered->points.size (); ++i)
# std::cerr << "    " << cloud_filtered->points[i].x << " " 
#                     << cloud_filtered->points[i].y << " " 
#                     << cloud_filtered->points[i].z << std::endl;
print ('Cloud after filtering: ')
for i in range(0, cloud_filtered.size):
    print ('x: '  + str(cloud_filtered[i][0]) + ', y : ' + str(cloud_filtered[i][1])  + ', z : ' + str(cloud_filtered[i][2]))
