# -*- coding: utf-8 -*-
# Cylinder model segmentation
# http://pointclouds.org/documentation/tutorials/cylinder_segmentation.php#cylinder-segmentation
# dataset : https://raw.github.com/PointCloudLibrary/data/master/tutorials/table_scene_mug_stereo_textured.pcd

import pcl

# typedef pcl::PointXYZ PointT;
# int main (int argc, char** argv)
# // All the objects needed
# pcl::PCDReader reader;
# pcl::PassThrough<PointT> pass;
# pcl::NormalEstimation<PointT, pcl::Normal> ne;
# pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; 
# pcl::PCDWriter writer;
# pcl::ExtractIndices<PointT> extract;
# pcl::ExtractIndices<pcl::Normal> extract_normals;
# pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
# 
# // Datasets
# pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
# pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
# pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
# pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
# pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
# pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
# pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);
# 
# // Read in the cloud data
# reader.read ("table_scene_mug_stereo_textured.pcd", *cloud);
# std::cerr << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;
cloud = pcl.load("./examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd")
print('PointCloud has: ' + str(cloud.size) + ' data points.')

# Build a passthrough filter to remove spurious NaNs
# pass.setInputCloud (cloud);
# pass.setFilterFieldName ("z");
# pass.setFilterLimits (0, 1.5);
# pass.filter (*cloud_filtered);
# std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;
passthrough = cloud.make_passthrough_filter()
passthrough.set_filter_field_name ('z')
passthrough.set_filter_limits (0, 1.5)
cloud_filtered = passthrough.filter()
print('PointCloud has: ' + str(cloud_filtered.size) + ' data points.')

# Estimate point normals
# ne.setSearchMethod (tree);
# ne.setInputCloud (cloud_filtered);
# ne.setKSearch (50);
# ne.compute (*cloud_normals);
ne = cloud_filtered.make_NormalEstimation()
tree = cloud_filtered.make_kdtree()
ne.set_SearchMethod (tree)
ne.set_KSearch (50)
# cloud_normals = ne.compute ()


# Create the segmentation object for the planar model and set all the parameters
# seg.setOptimizeCoefficients (true);
# seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
# seg.setNormalDistanceWeight (0.1);
# seg.setMethodType (pcl::SAC_RANSAC);
# seg.setMaxIterations (100);
# seg.setDistanceThreshold (0.03);
# seg.setInputCloud (cloud_filtered);
# seg.setInputNormals (cloud_normals);
# // Obtain the plane inliers and coefficients
# seg.segment (*inliers_plane, *coefficients_plane);
# std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

# SACSegmentationFromNormals
# seg = cloud_filtered.make_segmenter_normals(ksearch=50)
seg = cloud_filtered.make_segmenter_normals(ksearch=50)
seg.set_optimize_coefficients (True)
seg.set_model_type (pcl.SACMODEL_NORMAL_PLANE)
seg.set_normal_distance_weight (0.1)
seg.set_method_type (pcl.SAC_RANSAC)
seg.set_max_iterations (100)
seg.set_distance_threshold (0.03)
# seg.set_InputNormals (cloud_normals)
[inliers_plane, coefficients_plane] = seg.segment ()

# // Extract the planar inliers from the input cloud
# extract.setInputCloud (cloud_filtered);
# extract.setIndices (inliers_plane);
# extract.setNegative (false);
#
# // Write the planar inliers to disk
# pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
# extract.filter (*cloud_plane);
# std::cerr << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
# writer.write ("table_scene_mug_stereo_textured_plane.pcd", *cloud_plane, false);
cloud_plane = cloud_filtered.extract(inliers_plane, False)
print('PointCloud representing the planar component: ' + str(cloud_plane.size) + ' data points.\n')
pcl.save(cloud_plane, 'table_scene_mug_stereo_textured_plane.pcd')

# // Remove the planar inliers, extract the rest
# extract.setNegative (true);
# extract.filter (*cloud_filtered2);
cloud_filtered2 = cloud_filtered.extract(inliers_plane, True)

# extract_normals.setNegative (true);
# extract_normals.setInputCloud (cloud_normals);
# extract_normals.setIndices (inliers_plane);
# extract_normals.filter (*cloud_normals2);
# cloud_normals2 = cloud_normals.extract(inliers_plane, True)

# 
# // Create the segmentation object for cylinder segmentation and set all the parameters
# seg.setOptimizeCoefficients (true);
# seg.setModelType (pcl::SACMODEL_CYLINDER);
# seg.setMethodType (pcl::SAC_RANSAC);
# seg.setNormalDistanceWeight (0.1);
# seg.setMaxIterations (10000);
# seg.setDistanceThreshold (0.05);
# seg.setRadiusLimits (0, 0.1);
# seg.setInputCloud (cloud_filtered2);
# seg.setInputNormals (cloud_normals2);
# 
# // Obtain the cylinder inliers and coefficients
# seg.segment (*inliers_cylinder, *coefficients_cylinder);
# std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;
seg = cloud_filtered2.make_segmenter_normals(ksearch=50)
seg.set_optimize_coefficients (True)
seg.set_model_type (pcl.SACMODEL_CYLINDER)
seg.set_normal_distance_weight (0.1)
seg.set_method_type (pcl.SAC_RANSAC)
seg.set_max_iterations (10000)
seg.set_distance_threshold (0.05)
seg.set_radius_limits (0, 0.1)
# seg.set_InputNormals (cloud_normals2)
[inliers_cylinder, coefficients_cylinder] = seg.segment ()

#   // Write the cylinder inliers to disk
#   extract.setInputCloud (cloud_filtered2);
#   extract.setIndices (inliers_cylinder);
#   extract.setNegative (false);
#   pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT> ());
#   extract.filter (*cloud_cylinder);
cloud_cylinder = cloud_filtered2.extract(inliers_cylinder, False)

#   if (cloud_cylinder->points.empty ()) 
#     std::cerr << "Can't find the cylindrical component." << std::endl;
#   else
#   {
#     std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size () << " data points." << std::endl;
#     writer.write ("table_scene_mug_stereo_textured_cylinder.pcd", *cloud_cylinder, false);
#   }
# 
if cloud_cylinder.size == 0:
    print("Can't find the cylindrical component.")
else:
    print("PointCloud representing the cylindrical component: " + str(cloud_cylinder.size) + " data points.")
    pcl.save(cloud_cylinder, 'table_scene_mug_stereo_textured_cylinder.pcd')


