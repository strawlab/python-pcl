# -*- coding: utf-8 -*-
# Construct a concave or convex hull polygon for a plane model
# http://pointclouds.org/documentation/tutorials/hull_2d.php#hull-2d

import numpy as np
import pcl
import random

#  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), 
#                                      cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>), 
#                                      cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);
# cloud = pcl.PointCloud()
# cloud_filtered = pcl.PointCloud()
# cloud_projected = pcl.PointCloud()

#  pcl::PCDReader reader;
#  reader.read ("table_scene_mug_stereo_textured.pcd", *cloud);
cloud = pcl.load("./examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd")

# // Build a filter to remove spurious NaNs
# pcl::PassThrough<pcl::PointXYZ> pass;
# pass.setInputCloud (cloud);
# pass.setFilterFieldName ("z");
# pass.setFilterLimits (0, 1.1);
# pass.filter (*cloud_filtered);
# std::cerr << "PointCloud after filtering has: "
#           << cloud_filtered->points.size () << " data points." << std::endl;
passthrough = cloud.make_passthrough_filter()
passthrough.set_filter_field_name ("z")
passthrough.set_filter_limits (0.0, 1.1)
cloud_filtered = passthrough.filter ()
print ('PointCloud after filtering has: ' + str(cloud_filtered.size) + ' data points.')

# pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
# pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
# // Create the segmentation object
# pcl::SACSegmentation<pcl::PointXYZ> seg;
# // Optional
# seg.setOptimizeCoefficients (true);
# // Mandatory
# seg.setModelType (pcl::SACMODEL_PLANE);
# seg.setMethodType (pcl::SAC_RANSAC);
# seg.setDistanceThreshold (0.01);
# seg.setInputCloud (cloud_filtered);
# seg.segment (*inliers, *coefficients);
# std::cerr << "PointCloud after segmentation has: "
#           << inliers->indices.size () << " inliers." << std::endl;
seg = cloud_filtered.make_segmenter_normals(ksearch=50)
seg.set_optimize_coefficients(True)
seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(0.01)
indices, model = seg.segment()

print ('PointCloud after segmentation has: ' + str(indices.count) + ' inliers.')

#   // Project the model inliers
#   pcl::ProjectInliers<pcl::PointXYZ> proj;
#   proj.setModelType (pcl::SACMODEL_PLANE);
#   proj.setIndices (inliers);
#   proj.setInputCloud (cloud_filtered);
#   proj.setModelCoefficients (coefficients);
#   proj.filter (*cloud_projected);
#   std::cerr << "PointCloud after projection has: "
#             << cloud_projected->points.size () << " data points." << std::endl;
proj = cloud_filtered.make_ProjectInliers()
proj.set_model_type (pcl.SACMODEL_PLANE);
#   proj.setIndices (inliers);
#   proj.setModelCoefficients (coefficients)
cloud_projected = proj.filter ()

print ('PointCloud after projection has: ' + str(cloud_projected.size) + ' data points.')

#   // Create a Concave Hull representation of the projected inliers
#   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
#   pcl::ConcaveHull<pcl::PointXYZ> chull;
#   chull.setInputCloud (cloud_projected);
#   chull.setAlpha (0.1);
#   chull.reconstruct (*cloud_hull);
#   std::cerr << "Concave hull has: " << cloud_hull->points.size ()
#             << " data points." << std::endl;
# cloud_projected = pcl.PointCloud()
chull = cloud_projected.make_ConcaveHull()
chull.set_Alpha (0.1)
cloud_hull = chull.reconstruct ()
print ('Concave hull has: ' + str(cloud_hull.size) + ' data points.')

#   pcl::PCDWriter writer;
#   writer.write ("table_scene_mug_stereo_textured_hull.pcd", *cloud_hull, false);

if cloud_hull.size != 0:
    pcl.save(cloud_hull, 'table_scene_mug_stereo_textured_hull.pcd')

