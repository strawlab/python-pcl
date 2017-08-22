# Projecting points using a parametric model
# http://pointclouds.org/documentation/tutorials/project_inliers.php#project-inliers

import pcl
import numpy as np
import random

# int main (int argc, char** argv)
# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);
cloud = pcl.PointCloud()
cloud_projected = pcl.PointCloud()

# // Fill in the cloud data
# cloud->width  = 5;
# cloud->height = 1;
# cloud->points.resize (cloud->width * cloud->height);
# 
# for (size_t i = 0; i < cloud->points.size (); ++i)
# {
# cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
# cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
# cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
# }
# x,y,z
points = np.zeros((5, 3), dtype=np.float32)
RAND_MAX = 1.0
for i in range(0, 5):
    points[i][0] = 1024 * random.random () / RAND_MAX
    points[i][1] = 1024 * random.random () / RAND_MAX
    points[i][2] = 1024 * random.random () / RAND_MAX

cloud.from_array(points)

# std::cerr << "Cloud before projection: " << std::endl;
# for (size_t i = 0; i < cloud->points.size (); ++i)
# std::cerr << "    " << cloud->points[i].x << " " 
#                     << cloud->points[i].y << " " 
#                     << cloud->points[i].z << std::endl;
print ('Cloud before projection: ')
for i in range(0, cloud.size):
    print ('x: '  + str(cloud[i][0]) + ', y : ' + str(cloud[i][1])  + ', z : ' + str(cloud[i][2]))

# segment parts
# // Create a set of planar coefficients with X=Y=0, Z=1
# pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
# coefficients->values.resize (4);
# coefficients->values[0] = coefficients->values[1] = 0;
# coefficients->values[2] = 1.0;
# coefficients->values[3] = 0;
# 
# Create the filtering object
# pcl::ProjectInliers<pcl::PointXYZ> proj;
# proj.setModelType (pcl::SACMODEL_PLANE);
# proj.setInputCloud (cloud);
# proj.setModelCoefficients (coefficients);
# proj.filter (*cloud_projected);
proj = cloud.make_ProjectInliers()
proj.set_model_type (pcl.SACMODEL_PLANE)
# proj.setModelCoefficients (coefficients);
cloud_projected = proj.filter()

# std::cerr << "Cloud after projection: " << std::endl;
# for (size_t i = 0; i < cloud_projected->points.size (); ++i)
# std::cerr << "    " << cloud_projected->points[i].x << " " 
#                     << cloud_projected->points[i].y << " " 
#                     << cloud_projected->points[i].z << std::endl;
print ('Cloud after projection: ')
for i in range(0, cloud_projected.size):
    print ('x: '  + str(cloud_projected[i][0]) + ', y : ' + str(cloud_projected[i][1])  + ', z : ' + str(cloud_projected[i][2]))

