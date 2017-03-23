# -*- coding: utf-8 -*-
# 3D Object Recognition based on Correspondence Grouping
# http://pointclouds.org/documentation/tutorials/correspondence_grouping.php#correspondence-grouping
# python correspondence_grouping.py milk.pcd milk_cartoon_all_small_clorox.pcd
# python correspondence_grouping.py milk.pcd milk_cartoon_all_small_clorox.pcd milk.pcd milk_cartoon_all_small_clorox.pcd -r --model_ss 7.5 --scene_ss 20 --rf_rad 10 --descr_rad 15 --cg_size 10
import pcl
import numpy as np
import random
import argparse
import sys

# typedef pcl::PointXYZRGBA PointType;
# typedef pcl::Normal NormalType;
# typedef pcl::ReferenceFrame RFType;
# typedef pcl::SHOT352 DescriptorType;

# string model_filename_ = 'milk.pcd'
# string scene_filename_ = 'milk_cartoon_all_small_clorox.pcd'

model_filename_ = ''
scene_filename_ = ''

# Algorithm params
# bool show_keypoints_ (false)
# bool show_correspondences_ (false)
# bool use_cloud_resolution_ (false)
# bool use_hough_ (true)
# float model_ss_ (0.01f)
# float scene_ss_ (0.03f)
# float rf_rad_ (0.015f)
# float descr_rad_ (0.02f)
# float cg_size_ (0.01f)
# float cg_thresh_ (5.0f)
show_keypoints_ = False
show_correspondences_ = False
use_cloud_resolution_ = False
use_hough_ = True
model_ss_ = 0.01
scene_ss_ = 0.03
rf_rad_ = 0.015
descr_rad_ = 0.02
cg_size_ = 0.01
cg_thresh_ = 5.0

# void showHelp (char *filename)
# {
#   std::cout << std::endl;
#   std::cout << "***************************************************************************" << std::endl;
#   std::cout << "*                                                                         *" << std::endl;
#   std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
#   std::cout << "*                                                                         *" << std::endl;
#   std::cout << "***************************************************************************" << std::endl << std::endl;
#   std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
#   std::cout << "Options:" << std::endl;
#   std::cout << "     -h:                     Show this help." << std::endl;
#   std::cout << "     -k:                     Show used keypoints." << std::endl;
#   std::cout << "     -c:                     Show used correspondences." << std::endl;
#   std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
#   std::cout << "                             each radius given by that value." << std::endl;
#   std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
#   std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
#   std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
#   std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
#   std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
#   std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
#   std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
# }
# 

# void parseCommandLine (int argc, char *argv[])
# {
#   //Show help
#   if (pcl::console::find_switch (argc, argv, "-h"))
#   {
#     showHelp (argv[0]);
#     exit (0);
#   }
# 
#   //Model & scene filenames
#   std::vector<int> filenames;
#   filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
#   if (filenames.size () != 2)
#   {
#     std::cout << "Filenames missing.\n";
#     showHelp (argv[0]);
#     exit (-1);
#   }
#
#   model_filename_ = argv[filenames[0]];
#   scene_filename_ = argv[filenames[1]];
# 
#   //Program behavior
#   if (pcl::console::find_switch (argc, argv, "-k"))
#   {
#     show_keypoints_ = true;
#   }
#   if (pcl::console::find_switch (argc, argv, "-c"))
#   {
#     show_correspondences_ = true;
#   }
#   if (pcl::console::find_switch (argc, argv, "-r"))
#   {
#     use_cloud_resolution_ = true;
#   }
# 
#   std::string used_algorithm;
#   if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
#   {
#     if (used_algorithm.compare ("Hough") == 0)
#     {
#       use_hough_ = true;
#     }else if (used_algorithm.compare ("GC") == 0)
#     {
#       use_hough_ = false;
#     }
#     else
#     {
#       std::cout << "Wrong algorithm name.\n";
#       showHelp (argv[0]);
#       exit (-1);
#     }
#   }
# 
#   //General parameters
#   pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
#   pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
#   pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
#   pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
#   pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
#   pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
# }

# def double computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
#     double res = 0.0
#     int n_points = 0
#     int nres
#     std::vector<int> indices (2);
#     std::vector<float> sqr_distances (2);
#     pcl::search::KdTree<PointType> tree;
#     tree.setInputCloud (cloud);
# 
#     for (size_t i = 0; i < cloud->size (); ++i)
#         if (! pcl_isfinite ((*cloud)[i].x))
#             continue;
#         end
# 
#         //Considering the second neighbor since the first is the point itself.
#         nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
#         if (nres == 2)
#             res += sqrt (sqr_distances[1]);
#             ++n_points;
#         end
#     end
# 
#     if (n_points != 0)
#         res /= n_points
#     end
# 
#     return res
# end

# main
# int main (int argc, char *argv[])
# parse
# parseCommandLine (argc, argv);
argvs = sys.argv  # ÉRÉ}ÉìÉhÉâÉCÉìà¯êîÇäiî[ÇµÇΩÉäÉXÉgÇÃéÊìæ
argc = len(argvs) # à¯êîÇÃå¬êî

# string model_filename_ = 'milk.pcd'
# string scene_filename_ = 'milk_cartoon_all_small_clorox.pcd'
model_filename_ = argvs[1]
scene_filename_ = argvs[2]

parser = argparse.ArgumentParser(description='PointCloudLibrary example: correspondence_grouping correspondence_grouping')
parser.add_argument('--UnseenToMaxRange', '-m', default=True, type=bool,
                    help='Setting unseen values in range image to maximum range readings')
parser.add_argument('--algorithm', '-algorithm', choices=('Hough', 'GC'), default='',
                    help='Using algorithm Hough|GC.')
parser.add_argument('--model_ss', '-s', default=0.01, type=double,
                    help='Model uniform sampling radius (default 0.01)')
parser.add_argument('--scene_ss', '-s', default=0.03, type=double,
                    help='Scene uniform sampling radius (default 0.03)')
parser.add_argument('--rf_rad', '-rf', default=0.01, type=double,
                    help='Reference frame radius (default 0.015)\n')
parser.add_argument('--descr_rad', '-s', default=0.02, type=double,
                    help='Descriptor radius (default 0.02)\n')
parser.add_argument('--cg_size', '-s', default=0.01, type=double,
                    help='Descriptor radius (default 0.02)\n')
parser.add_argument('--cg_thresh', '-cg_thresh', default=5, type=int,
                    help='Clustering threshold (default 5)\n')
parser.add_argument('--Help', 
                    help='Usage: model_filename.pcd scene_filename.pcd [Options]\n\n'
                    'Options:\n'
                    '------------------------------------------\n'
                    '-h:                     Show this help.\n'
                    '-k:                     Show used keypoints.\n'
                    '-c:                     Show used correspondences.\n'
                    '-r:                     Compute the model cloud resolution and multiply\n'
                    '                        each radius given by that value.\n'
                    '--rf_rad val:           Reference frame radius (default 0.015)\n'
                    '--descr_rad val:        Descriptor radius (default 0.02)\n'
                    '--cg_size val:          Cluster size (default 0.01)\n'
                    '--cg_thresh val:        Clustering threshold (default 5)\n\n;')

args = parser.parse_args()

# Program behavior
# if (pcl::console::find_switch (argc, argv, "-k"))
#   show_keypoints_ = true;
#
# if (pcl::console::find_switch (argc, argv, "-c"))
#   show_correspondences_ = true;
#
# if (pcl::console::find_switch (argc, argv, "-r"))
#   use_cloud_resolution_ = true;
show_keypoints_ = args.show_keypoints_;
# show_correspondences_ = args.
use_cloud_resolution_ = args.use_cloud_resolution
use_hough_  = args.use_hough
model_ss_  = args.model_ss
scene_ss_ = args.scene_ss
rf_rad_ = args.rf_rad
descr_rad_ = args.descr_rad
cg_size_ = args.cg_size
cg_thresh_ = args.cg_thresh


# settings
model = pcl.PointCloud_XYZRGBA()
model_keypoints = pcl.PointCloud_XYZRGBA()
scene = pcl.PointCloud_XYZRGBA()
scene_keypoints = pcl.PointCloud_XYZRGBA()
model_normals = pcl.PointCloud_Normal()
scene_normals = pcl.PointCloud_Normal()
model_descriptors = pcl.PointCloud_SHOT352()
scene_descriptors = pcl.PointCloud_SHOT352()


# Load clouds
model = pcl.load_XYZRGBA(model_filename_)
scene = pcl.load_XYZRGBA(scene_filename_)

# Set up resolution invariance
if use_cloud_resolution_ == True:
    # float resolution = static_cast<float> (computeCloudResolution (model))
    resolution = 0.0

    if resolution != 0.0:
        model_ss_   *= resolution;
        scene_ss_   *= resolution;
        rf_rad_     *= resolution;
        descr_rad_  *= resolution;
        cg_size_    *= resolution;

    print('Model resolution:       ' + resolution )
    print('Model sampling size:    ' + model_ss_ )
    print('Scene sampling size:    ' + scene_ss_ )
    print('LRF support radius:     ' + rf_rad_ )
    print('SHOT descriptor radius: ' + descr_rad_ )
    print('Clustering bin size:    ' + cg_size_ )


# Compute Normals
# pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
# norm_est.setKSearch (10);
# norm_est.setInputCloud (model);
# norm_est.compute (*model_normals);
# model_normals = norm_est.Å`
norm_est = model.make_segmenter_normals(10)
norm_est.setKSearch
model_normals = 

# scene_normals = norm_est2.Å`
# norm_est.setInputCloud (scene);
# norm_est.compute (*scene_normals);
norm_est = norm_est.set_InputCloud(scene)
scene_normals =  norm_est.make_segmenter_normals(10)

# Downsample Clouds to Extract keypoints
# pcl::UniformSampling<PointType> uniform_sampling;
# uniform_sampling = pcl.UniformSampling_XYZRGBA()
# uniform_sampling.setInputCloud (model);
# uniform_sampling.setRadiusSearch (model_ss_);
# uniform_sampling.filter (*model_keypoints);
# std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
uniform_sampling = pcl.UniformSampling_XYZRGBA()
uniform_sampling.set_RadiusSearch (model_ss_);
model_keypoints = uniform_sampling.filter()
print("Model total points: " + str(model.size()) + "; Selected Keypoints: " + str(model_keypoints.size()) + "\n")

# uniform_sampling.setInputCloud (scene)
# uniform_sampling.setRadiusSearch (scene_ss_)
# uniform_sampling.filter (*scene_keypoints)
# std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;
uniform_sampling.setInputCloud (scene)
uniform_sampling.setRadiusSearch (scene_ss_)
scene_keypoints = uniform_sampling.filter ()
print("Model total points: " + str(scene.size()) + "; Selected Keypoints: " + str(scene_keypoints.size()) + "\n")

# Compute Descriptor for keypoints
# pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
# descr_est.setRadiusSearch (descr_rad_);
# descr_est.setInputCloud (model_keypoints);
# descr_est.setInputNormals (model_normals);
# descr_est.setSearchSurface (model);
# descr_est.compute (*model_descriptors);
descr_est = model_keypoints.make_SHOTEstimationOMP()
descr_est.setRadiusSearch (descr_rad_)
descr_est.setSearchSurface (model)
model_descriptors = descr_est.compute()

# descr_est.setInputCloud (scene_keypoints);
# descr_est.setInputNormals (scene_normals);
# descr_est.setSearchSurface (scene);
# descr_est.compute (*scene_descriptors)
descr_est.setInputCloud (scene_keypoints)
descr_est.setInputNormals (scene_normals)
descr_est.setSearchSurface (scene)
scene_descriptors = descr_est.compute ()

# Find Model-Scene Correspondences with KdTree
# pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
model_scene_corrs = pcl.Correspondences()

# pcl::KdTreeFLANN<DescriptorType> match_search;
# match_search.setInputCloud (model_descriptors);
match_search = model_descriptors.make_KdTreeFLANN()

# For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
# for (size_t i = 0; i < scene_descriptors->size (); ++i)
# {
#     std::vector<int> neigh_indices (1);
#     std::vector<float> neigh_sqr_dists (1);
#     if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
#     {
#         continue;
#     }
#     int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
#     if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
#     {
#         pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
#         model_scene_corrs->push_back (corr);
#     }
# }

for i in range(i, scene_descriptors.size):
    pass
#     std::vector<int> neigh_indices (1);
#     std::vector<float> neigh_sqr_dists (1);
#     if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
#     {
#         continue;
#     }
#     int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
#     if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
#     {
#         pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
#         model_scene_corrs->push_back (corr);
#     }


# std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl
print ("Correspondences found: " + str(model_scene_corrs.size))

# //  Actual Clustering
# std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
# std::vector<pcl::Correspondences> clustered_corrs;

# Using Hough3D
# if use_hough_ == True:
#     # Compute (Keypoints) Reference Frames only for Hough
#     pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
#     pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());
# 
#     pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
#     rf_est.setFindHoles (true);
#     rf_est.setRadiusSearch (rf_rad_);
# 
#     rf_est.setInputCloud (model_keypoints);
#     rf_est.setInputNormals (model_normals);
#     rf_est.setSearchSurface (model);
#     rf_est.compute (*model_rf);
# 
#     rf_est.setInputCloud (scene_keypoints);
#     rf_est.setInputNormals (scene_normals);
#     rf_est.setSearchSurface (scene);
#     rf_est.compute (*scene_rf);
# 
#     //  Clustering
#     pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
#     clusterer.setHoughBinSize (cg_size_);
#     clusterer.setHoughThreshold (cg_thresh_);
#     clusterer.setUseInterpolation (true);
#     clusterer.setUseDistanceWeight (false);
# 
#     clusterer.setInputCloud (model_keypoints);
#     clusterer.setInputRf (model_rf);
#     clusterer.setSceneCloud (scene_keypoints);
#     clusterer.setSceneRf (scene_rf);
#     clusterer.setModelSceneCorrespondences (model_scene_corrs);
# 
#     //clusterer.cluster (clustered_corrs);
#     clusterer.recognize (rototranslations, clustered_corrs);
# else: 
#     // Using GeometricConsistency
#     pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
#     gc_clusterer.setGCSize (cg_size_);
#     gc_clusterer.setGCThreshold (cg_thresh_);
# 
#     gc_clusterer.setInputCloud (model_keypoints);
#     gc_clusterer.setSceneCloud (scene_keypoints);
#     gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);
# 
#     //gc_clusterer.cluster (clustered_corrs);
#     gc_clusterer.recognize (rototranslations, clustered_corrs);

# Using Hough3D
if use_hough_ == True:
    # Compute (Keypoints) Reference Frames only for Hough
    # pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    # pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());
    
    # 1.7.2
    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est
    rf_est.setFindHoles (True)
    rf_est.setRadiusSearch (rf_rad_)
    
    rf_est.setInputCloud (model_keypoints)
    rf_est.setInputNormals (model_normals)
    rf_est.setSearchSurface (model)
    model_rf = rf_est.compute ()
    
    rf_est.setInputCloud (scene_keypoints)
    rf_est.setInputNormals (scene_normals)
    rf_est.setSearchSurface (scene)
    scene_rf = rf_est.compute ()
    
    # Clustering
    # pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_)
    clusterer.setHoughThreshold (cg_thresh_)
    clusterer.setUseInterpolation (True)
    clusterer.setUseDistanceWeight (False)
    
    clusterer.setInputCloud (model_keypoints)
    clusterer.setInputRf (model_rf)
    clusterer.setSceneCloud (scene_keypoints)
    clusterer.setSceneRf (scene_rf)
    clusterer.setModelSceneCorrespondences (model_scene_corrs)
    
    # //clusterer.cluster (clustered_corrs)
    clusterer.recognize (rototranslations, clustered_corrs)
else: 
    # // Using GeometricConsistency
    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer
    gc_clusterer.setGCSize (cg_size_)
    gc_clusterer.setGCThreshold (cg_thresh_)
    
    gc_clusterer.setInputCloud (model_keypoints)
    gc_clusterer.setSceneCloud (scene_keypoints)
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs)
    
    # //gc_clusterer.cluster (clustered_corrs)
    gc_clusterer.recognize (rototranslations, clustered_corrs)
    

# Output results
# std::cout << "Model instances found: " << rototranslations.size () << std::endl;
print("Model instances found: " + str(rototranslations.size()) + "\n")

# for (size_t i = 0; i < rototranslations.size (); ++i)
# {
#     std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
#     std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;
# 
#     // Print the rotation matrix and translation vector
#     Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
#     Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);
# 
#     printf ("\n");
#     printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
#     printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
#     printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
#     printf ("\n");
#     printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
# }

for i in range(i, rototranslations.size)
    print('\n    Instance ' + str(i + 1) + ':')
    print('        Correspondences belonging to this instance: ' + str(clustered_corrs[i].size) )
    
    # Print the rotation matrix and translation vector
    eigen3.Matrix3f rotation = rototranslations[i].block<3, 3>(0, 0)
    eigen3.Vector3f translation = rototranslations[i].block<3, 1>(0, 3)
    
    printf ('\n')
    printf ('            | %6.3f %6.3f %6.3f | \n', rotation (0,0), rotation (0,1), rotation (0,2))
    printf ('        R = | %6.3f %6.3f %6.3f | \n', rotation (1,0), rotation (1,1), rotation (1,2))
    printf ('            | %6.3f %6.3f %6.3f | \n', rotation (2,0), rotation (2,1), rotation (2,2))
    printf ('\n')
    printf ('        t = < %0.3f, %0.3f, %0.3f >\n', translation (0), translation (1), translation (2))


# Visualization
# pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
# viewer.addPointCloud (scene, "scene_cloud");
viewer = pcl.PCLVisualizer('Correspondence Grouping')
viewer.AddPointCloud (scene, 'scene_cloud')

# pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
# pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());
# if (show_correspondences_ || show_keypoints_)
# {
#     # We are translating the model so that it doesn't end in the middle of the scene representation
#     pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
#     pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
# 
#     pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
#     viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
# }
if (show_correspondences_ || show_keypoints_) == True:
    # We are translating the model so that it doesn't end in the middle of the scene representation
    pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));


# if (show_keypoints_)
# {
#     pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
#     viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
#     viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");
# 
#     pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
#     viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
#     viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
# }

if show_keypoints_ == True:
    # scene_keypoints_color_handler = pcl::visualization::PointCloudColorHandlerCustom<PointType>(scene_keypoints, 0, 0, 255)
    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints")
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints")
	
    off_scene_model_keypoints_color_handler = pcl::visualization::PointCloudColorHandlerCustom<PointType>(off_scene_model_keypoints, 0, 0, 255)
    viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints")
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints")
	

# for (size_t i = 0; i < rototranslations.size (); ++i)
# {
#     pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
#     pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);
# 
#     std::stringstream ss_cloud;
#     ss_cloud << "instance" << i;
# 
#     pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
#     viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());
# 
#     if (show_correspondences_)
#     {
#         for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
#         {
#             std::stringstream ss_line;
#             ss_line << "correspondence_line" << i << "_" << j;
#             PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
#             PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);
# 
#             //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
#             viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
#         }
#     }
# }

for i = 0 in range(i, rototranslations.size):
    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);
    
    print('instance' + str(i))
    
    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
    viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());
    
    if show_correspondences_ == True:
        for j = 0 in range(j, clustered_corrs[i].size)
            # ss_line << "correspondence_line" << i << "_" << j;
            # PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
            # PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);
            # //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
            # viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
            pass


# while (!viewer.wasStopped ())
# {
#     viewer.spinOnce ();
# }

while viewer.wasStopped() == True:
    viewer.spinOnce ()

