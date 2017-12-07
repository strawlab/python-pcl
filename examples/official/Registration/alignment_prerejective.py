# -*- coding: utf-8 -*-
# Robust pose estimation of rigid objects
# http://pointclouds.org/documentation/tutorials/alignment_prerejective.php#alignment-prerejective

import pcl
import argparse
parser = argparse.ArgumentParser(description='PointCloudLibrary example: Remove outliers')
parser.add_argument('--Removal', '-r', choices=('Radius', 'Condition'), default='',
                    help='RadiusOutlier/Condition Removal')
args = parser.parse_args()

# // Types
# typedef pcl::PointNormal PointNT;
# typedef pcl::PointCloud<PointNT> PointCloudT;
# typedef pcl::FPFHSignature33 FeatureT;
# typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
# typedef pcl::PointCloud<FeatureT> FeatureCloudT;
# typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;
# 
# Align a rigid object to a scene with clutter and occlusions
# 
# // Point clouds
# PointCloudT::Ptr object (new PointCloudT);
# PointCloudT::Ptr object_aligned (new PointCloudT);
# PointCloudT::Ptr scene (new PointCloudT);
# FeatureCloudT::Ptr object_features (new FeatureCloudT);
# FeatureCloudT::Ptr scene_features (new FeatureCloudT);

# // Get input object and scene
# if (argc != 3)
# {
#   pcl::console::print_error ("Syntax is: %s object.pcd scene.pcd\n", argv[0]);
#   return (1);
# }
if args.n != 3:
    print('Syntax is: " + "" +" object.pcd scene.pcd\n')
    return (1)

# // Load object and scene
# pcl::console::print_highlight ("Loading point clouds...\n");
# if (pcl::io::loadPCDFile<PointNT> (argv[1], *object) < 0 ||
#       pcl::io::loadPCDFile<PointNT> (argv[2], *scene) < 0)
#   {
#     pcl::console::print_error ("Error loading object/scene file!\n");
#     return (1);
#   }
print('Loading point clouds...\n')
object = pcl.load('')
scene = pcl.load('')

# // Downsample
# pcl::console::print_highlight ("Downsampling...\n");
# pcl::VoxelGrid<PointNT> grid;
# const float leaf = 0.005f;
# grid.setLeafSize (leaf, leaf, leaf);
# grid.setInputCloud (object);
# grid.filter (*object);
# grid.setInputCloud (scene);
# grid.filter (*scene);
print('Downsampling...\n')
grid_obj = object.make_voxel_grid_filter()
leaf = 0.005
grid_obj.set_leaf_size (leaf, leaf, leaf)
object = grid_obj.filter()
scene_obj = scene.make_voxel_grid_filter()
grid_sce = scene_obj.filter ()

# // Estimate normals for scene
# pcl::console::print_highlight ("Estimating scene normals...\n");
# pcl::NormalEstimationOMP<PointNT,PointNT> nest;
# nest.setRadiusSearch (0.01);
# nest.setInputCloud (scene);
# nest.compute (*scene);
print('Estimating scene normals...\n')
nest = scene.make_NormalEstimationOMP()
nest.set_RadiusSearch (0.01);
scene = nest.compute ()

# // Estimate features
# pcl::console::print_highlight ("Estimating features...\n");
# FeatureEstimationT fest;
# fest.setRadiusSearch (0.025);
# fest.setInputCloud (object);
# fest.setInputNormals (object);
# fest.compute (*object_features);
# fest.setInputCloud (scene);
# fest.setInputNormals (scene);
# fest.compute (*scene_features);
print('Estimating features...\n')
fest_obj = object.make_FeatureEstimation()
fest_obj.setRadiusSearch (0.025)
object_features = fest_obj.compute ()

fest_sce = scene.make_FeatureEstimation()
fest_sce.setRadiusSearch (0.025)
scene_features = fest_sce.compute ()


# // Perform alignment
# pcl::console::print_highlight ("Starting alignment...\n");
# pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
# align.setInputSource (object);
# align.setSourceFeatures (object_features);
# align.setInputTarget (scene);
# align.setTargetFeatures (scene_features);
# align.setMaximumIterations (50000); // Number of RANSAC iterations
# align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
# align.setCorrespondenceRandomness (5); // Number of nearest features to use
# align.setSimilarityThreshold (0.9f); // Polygonal edge length similarity threshold
# align.setMaxCorrespondenceDistance (2.5f * leaf); // Inlier threshold
# align.setInlierFraction (0.25f); // Required inlier fraction for accepting a pose hypothesis
# {
#   pcl::ScopeTime t("Alignment");
#   align.align (*object_aligned);
# }
print('Starting alignment...\n')
align = object.make_SampleConsensusPrerejective()
align.setSourceFeatures (object_features)
align.setTargetFeatures (scene_features)
# Number of RANSAC iterations
align.set_MaximumIterations (50000)
# Number of points to sample for generating/prerejecting a pose
align.set_NumberOfSamples (3)
# Number of nearest features to use
align.set_CorrespondenceRandomness (5)
# Polygonal edge length similarity threshold
align.set_SimilarityThreshold (0.9)
# Inlier threshold
align.set_MaxCorrespondenceDistance (2.5 * leaf)

# if (align.hasConverged ())
# {
#   // Print results
#   printf ("\n");
#   Eigen::Matrix4f transformation = align.getFinalTransformation ();
#   pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
#   pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
#   pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
#   pcl::console::print_info ("\n");
#   pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
#   pcl::console::print_info ("\n");
#   pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), object->size ());
#   
#   // Show alignment
#   pcl::visualization::PCLVisualizer visu("Alignment");
#   visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
#   visu.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), "object_aligned");
#   visu.spin ();
# }
# else
# {
#   pcl::console::print_error ("Alignment failed!\n");
#   return (1);
# }

if align.hasConverged () == True:
    # Print results
    print ('\n');
    # Eigen::Matrix4f transformation = align.getFinalTransformation ()
    transformation = align.getFinalTransformation ()

    # print ('    | %6.3f %6.3f %6.3f | \n', transformation [0, 0], transformation [0, 1], transformation [0, 2])
    # print ('R = | %6.3f %6.3f %6.3f | \n', transformation [1, 0], transformation [1, 1], transformation [1, 2])
    # print ('    | %6.3f %6.3f %6.3f | \n', transformation [2, 0], transformation [2, 1], transformation [2, 2])
    # print ('\n');
    # print ('t = < %0.3f, %0.3f, %0.3f >\n', transformation[0, 3], transformation[1, 3], transformation[2, 3])
    # print ('\n');
    # print ('Inliers: %i/%i\n', align.getInliers ().size (), object->size ());

    # Show alignment
    visu = pcl.PCLVisualization('Alignment')
    visu.add_PointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), 'scene')
    visu.add_PointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), 'object_aligned')
    visu.spin ()
else:
    print('Alignment failed!\n')
    return (1)

