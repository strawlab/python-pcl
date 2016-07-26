# author Bastian Steder 
# http://pointclouds.org/documentation/tutorials/narf_keypoint_extraction.php#narf-keypoint-extraction

#coding: UTF-8

import pcl
import numpy as np
import random

import argparse

# Parameters
angular_resolution = 0.5
support_size = 0.2
coordinate_frame = pcl.CythonCoordinateFrame_Type.CAMERA_FRAME
setUnseenToMaxRange = false

import argparse

# void setViewerPose (pclvisualizationPCLVisualizer& viewer, const EigenAffine3f& viewer_pose)
#	EigenVector3f pos_vector = viewer_pose  EigenVector3f (0, 0, 0);
#	EigenVector3f look_at_vector = viewer_pose.rotation ()  EigenVector3f (0, 0, 1) + pos_vector;
#	EigenVector3f up_vector = viewer_pose.rotation ()  EigenVector3f (0, -1, 0);
#	viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
#                             look_at_vector[0], look_at_vector[1], look_at_vector[2],
#                             up_vector[0], up_vector[1], up_vector[2]);

# -----Main-----
# -----Parse Command Line Arguments-----
parser = argparse.ArgumentParser(description='StrawPCL example: narf keyPoint')
parser.add_argument('--UnseenToMaxRange', '-m', default=true, type=bool,
                    help='Setting unseen values in range image to maximum range readings')
parser.add_argument('--CoordinateFrame', '-c', default=-1, type=int,
                    help='Using coordinate frame = ')
parser.add_argument('--SupportSize', '-s', default=0, type=int,
                    help='Setting support size to = ')
parser.add_argument('--AngularResolution', '-r', default=0, type=int,
                    help='Usage: narf_keypoint_extraction.py [options] <scene.pcd>nn'
                    'Options:n'
                    '-------------------------------------------n'
                    '-r <float>   angular resolution in degrees (default = angular_resolution)n'
                    '-c <int>     coordinate frame (default = coordinate_frame)n'
                    '-m           Treat all unseen points as maximum range readingsn'
                    '-s <float>   support size for the interest points (diameter of the used sphere - default = support_size)n'
                    '-h           this helpnnn;')
args = parser.parse_args()

setUnseenToMaxRange = args.UnseenToMaxRange
coordinate_frame = pclRangeImageCoordinateFrame (args.CoordinateFrame)
angular_resolution = pcl.deg2rad (args.AngularResolution)

# -----Read pcd file or create example point cloud if not given-----
# pclPointCloudPointTypePtr point_cloud_ptr (new pclPointCloudPointType);
# pclPointCloudPointType& point_cloud = point_cloud_ptr
point_cloud = pcl.PointCloud()

# pcl::PointCloud<pcl::PointWithViewpoint> far_ranges
# EigenAffine3f scene_sensor_pose (EigenAffine3fIdentity ())
scene_sensor_pose = (EigenAffine3fIdentity ())
# vector[int] pcd_filename_indices = pcl::console::parse_file_extension_argument (argc, argv, pcd)
# pcd_filename_indices = pcl::console::parse_file_extension_argument (argc, argv, pcd)
pcd_filename_indices = [0, 0, 0]

# if pcd_filename_indices.empty() == False

if len(pcd_filename_indices) != 0:
    string filename = argv[pcd_filename_indices[0]]
    point_cloud = pcl.load(argv[0])

    scene_sensor_pose = Eigen::Affine3f (Eigen::Translation3f (point_cloud.sensor_origin_[0],
                                                               point_cloud.sensor_origin_[1],
                                                               point_cloud.sensor_origin_[2])) *
                        Eigen::Affine3f (point_cloud.sensor_orientation_);

    stdstring far_ranges_filename = pclgetFilenameWithoutExtension (filename)+_far_ranges.pcd;

    if (pclioloadPCDFile (far_ranges_filename.c_str (), far_ranges) == -1)
        stdcout  Far ranges file far_ranges_filename does not exists.n;
else:
    setUnseenToMaxRange = true
    print ('nNo *.pcd file given = Genarating example point cloud.nn')
    for (float x = -0.5f; x = 0.5f; x += 0.01f)
        for (float y = -0.5f; y = 0.5f; y += 0.01f)
            points = np.zeros((1, 3), dtype=np.float32)
            points[0][0] = x  
            points[0][1] = y
            points[0][2] = 2.0f - y
        end
    end
	point_cloud.points.push_back (point);

    
    point_cloud.width  = (int) point_cloud.points.size ()
    point_cloud.height = 1;

# # -----Create RangeImage from the PointCloud-----
# noise_level = 0.0
# min_range = 0.0f
# 
# int border_size = 1
# boostshared_ptrpclRangeImage range_image_ptr (new pclRangeImage);
# pclRangeImage& range_image = range_image_ptr;
# range_image.createFromPointCloud (
#                             point_cloud, angular_resolution, pcldeg2rad (360.0f), pcldeg2rad (180.0f),
#                             scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
# range_image.integrateFarRanges (far_ranges);
# 
# if (setUnseenToMaxRange)
#     range_image.setUnseenToMaxRange ();
# 
# # -----Open 3D viewer and add point cloud-----
# pclvisualizationPCLVisualizer viewer ("3D Viewer")
# viewer.setBackgroundColor (1, 1, 1)
# pclvisualizationPointCloudColorHandlerCustompclPointWithRange range_image_color_handler (range_image_ptr, 0, 0, 0);
# viewer.addPointCloud (range_image_ptr, range_image_color_handler, "range image");
# viewer.setPointCloudRenderingProperties (pclvisualizationPCL_VISUALIZER_POINT_SIZE, 1, "range image");
# viewer.initCameraParameters ();
# 
# # -----Show range image-----
# pclvisualizationRangeImageVisualizer range_image_widget (Range image);
# range_image_widget.showRangeImage (range_image);
# 
# # -----Extract NARF keypoints-----
# pclRangeImageBorderExtractor range_image_border_extractor;
# pclNarfKeypoint narf_keypoint_detector (&range_image_border_extractor);
# narf_keypoint_detector.setRangeImage (&range_image);
# narf_keypoint_detector.getParameters ().support_size = support_size;
# narf_keypoint_detector.getParameters ().add_points_on_straight_edges = true;
# narf_keypoint_detector.getParameters ().distance_for_additional_points = 0.5;
# 
# pclPointCloudint keypoint_indices;
# narf_keypoint_detector.compute (keypoint_indices);
# stdcout  Found keypoint_indices.points.size () key points.n;
# 
# # -----Show keypoints in range image widget-----
# for (size_t i=0; ikeypoint_indices.points.size (); ++i)
# range_image_widget.markPoint (keypoint_indices.points[i]%range_image.width,
#                               keypoint_indices.points[i]range_image.width);
# 
# # -----Show keypoints in 3D viewer-----
# pclPointCloudpclPointXYZPtr keypoints_ptr (new pclPointCloudpclPointXYZ);
# pclPointCloudpclPointXYZ& keypoints = keypoints_ptr;
# keypoints.points.resize (keypoint_indices.points.size ());
# for (size_t i=0; ikeypoint_indices.points.size (); ++i)
# keypoints.points[i].getVector3fMap () = range_image.points[keypoint_indices.points[i]].getVector3fMap ();
# 
# pclvisualizationPointCloudColorHandlerCustompclPointXYZ keypoints_color_handler (keypoints_ptr, 0, 255, 0);
# viewer.addPointCloudpclPointXYZ (keypoints_ptr, keypoints_color_handler, keypoints);
# viewer.setPointCloudRenderingProperties (pclvisualizationPCL_VISUALIZER_POINT_SIZE, 7, keypoints);
# 
# # -----Main loop-----
# # while (!viewer.wasStopped ())
# while True
#     # process GUI events
#     range_image_widget.spinOnce ()
#     viewer.spinOnce ()
#     # pcl_sleep(0.01);
# end

