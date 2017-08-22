# -*- coding: utf-8 -*-
# author : Bastian Steder 
# http://pointclouds.org/documentation/tutorials/narf_keypoint_extraction.php#narf-keypoint-extraction

import pcl
import pcl.pcl_visualization
import numpy as np
import random
import argparse
import time

# Parameters
angular_resolution = 0.5
support_size = 0.2
coordinate_frame = pcl.CythonCoordinateFrame_Type.CAMERA_FRAME
setUnseenToMaxRange = False

# void setViewerPose (pcl::visualization::PCLVisualizer& viewer, const EigenAffine3f& viewer_pose)
#   EigenVector3f pos_vector = viewer_pose  EigenVector3f (0, 0, 0);
#   EigenVector3f look_at_vector = viewer_pose.rotation ()  EigenVector3f (0, 0, 1) + pos_vector;
#   EigenVector3f up_vector = viewer_pose.rotation ()  EigenVector3f (0, -1, 0);
#   viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
#                             look_at_vector[0], look_at_vector[1], look_at_vector[2],
#                             up_vector[0], up_vector[1], up_vector[2]);

# -----Main-----
# -----Parse Command Line Arguments-----
parser = argparse.ArgumentParser(description='PointCloudLibrary example: narf keyPoint extraction')
parser.add_argument('--UnseenToMaxRange', '-m', default=True, type=bool,
                    help='Setting unseen values in range image to maximum range readings')
parser.add_argument('--CoordinateFrame', '-c', default=-1, type=int,
                    help='Using coordinate frame = ')
parser.add_argument('--SupportSize', '-s', default=0, type=int,
                    help='Setting support size to = ')
parser.add_argument('--AngularResolution', '-r', default=0, type=int,
                    help='Setting angular resolution to = ')
parser.add_argument('--Help', 
                    help='Usage: narf_keypoint_extraction.py [options] <scene.pcd>\n\n'
                    'Options:\n'
                    '-------------------------------------------\n'
                    '-r <float>   angular resolution in degrees (default = angular_resolution)\n'
                    '-c <int>     coordinate frame (default = coordinate_frame)\n'
                    '-m           Treat all unseen points as maximum range readings\n'
                    '-s <float>   support size for the interest points (diameter of the used sphere - default = support_size)\n'
                    '-h           this help\n\n\n')

args = parser.parse_args()

# args setting
setUnseenToMaxRange = args.UnseenToMaxRange
# coordinate_frame = pcl.RangeImage.CoordinateFrame (args.CoordinateFrame)
# angular_resolution = pcl.deg2rad (args.AngularResolution)

# -----Read pcd file or create example point cloud if not given-----
# pcl::PointCloudPointTypePtr point_cloud_ptr (new pcl::PointCloud::PointType);
# pcl::PointCloudPointType& point_cloud = point_cloud_ptr
# pcl::PointCloud<pcl::PointWithViewpoint> far_ranges
##
# point_cloud = pcl.PointCloud()

# Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ())
# scene_sensor_pose = (eigen3.Affine3f.Identity ())

# vector[int] pcd_filename_indices = pcl::console::parse_file_extension_argument (argc, argv, pcd)
# pcd_filename_indices = './examples/official/IO/test_pcd.pcd'
# pcd_filename_indices = [0, 0, 0]
# if pcd_filename_indices.empty() == False

pcd_filename_indices = ''
if len(pcd_filename_indices) != 0:
    # # string filename = argv[pcd_filename_indices[0]]
    # filename = argv[pcd_filename_indices[0]]
    # point_cloud = pcl.load(argv[0])
    point_cloud = pcl.load('./examples/official/IO/test_pcd.pcd')
    
    # scene_sensor_pose = Eigen::Affine3f (Eigen::Translation3f (point_cloud.sensor_origin_[0],
    #                                                            point_cloud.sensor_origin_[1],
    #                                                            point_cloud.sensor_origin_[2])) *
    #                     Eigen::Affine3f (point_cloud.sensor_orientation_);
    # Python
    # origin = point_cloud.sensor_origin
    # sensor_orientation = eigen3.Affine3f(origin[0], origin[1], origin[2]) * eigen3.Affine3f(point_cloud.sensor_orientation)
    
    # std::string far_ranges_filename = pcl::getFilenameWithoutExtension (filename)+_far_ranges.pcd;
    # if (pcl::io::loadPCDFile (far_ranges_filename.c_str (), far_ranges) == -1)
    #     stdcout  Far ranges file far_ranges_filename does not exists.n;
    far_ranges_filename = os.path.splitext(pcd_filename_indices) + '_far_ranges.pcd'
    far_ranges = pcl.load_PointWithViewpoint(far_ranges_filename)
    
    # Error
    # print('Far ranges file ' + far_ranges_filename + 'does not exists.\n')
    
else:
    setUnseenToMaxRange = True
    print ('No *.pcd file given = Genarating example point cloud.\n')
    
    # for (float x = -0.5f; x = 0.5f; x += 0.01f)
    #     for (float y = -0.5f; y = 0.5f; y += 0.01f)
    #         points = np.zeros((1, 3), dtype=np.float32)
    #         points[0][0] = x  
    #         points[0][1] = y
    #         points[0][2] = 2.0f - y
    #     end
    # end
    
    count = 0
    points = np.zeros((100 * 100, 3), dtype=np.float32)
    
    # float NG
    # TypeError: range() integer end argument expected, got float.
    # for x in range(-0.5, 0.5, 0.01):
    #     for y in range(-0.5, 0.5, 0.01):
    for x in range(-50, 50, 1):
        for y in range(-50, 50, 1):
            points[count][0] = x * 0.01
            points[count][1] = y * 0.01
            points[count][2] = 2.0 - y * 0.01
            count = count + 1
    
    # point_cloud.points.push_back (point);
    # point_cloud.width  = (int) point_cloud.points.size ()
    # point_cloud.height = 1;
    point_cloud = pcl.PointCloud()
    point_cloud.from_array(points)
    
    far_ranges = pcl.PointCloud_PointWithViewpoint()

# Create RangeImage from the PointCloud
noise_level = 0.0
min_range = 0.0
border_size = 1

# boost::shared_ptr<pcl::RangeImage> range_image_ptr (new pcl::RangeImage);
# pcl::RangeImage& range_image = *range_image_ptr;
range_image = point_cloud.make_RangeImage()

print ('range_image::createFromPointCloud.\n')
print ('point_cloud(size  ) = ' + str(point_cloud.size  ) )
print ('point_cloud(width ) = ' + str(point_cloud.width ) )
print ('point_cloud(height) = ' + str(point_cloud.height) )

# range_image.createFromPointCloud (
#                             point_cloud, angular_resolution, pcl.deg2rad (360.0f), pcl.deg2rad (180.0f),
#                             scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
range_image.CreateFromPointCloud (point_cloud, 
                                    angular_resolution, pcl.deg2rad (360.0), pcl.deg2rad (180.0), 
                                    coordinate_frame, noise_level, min_range, border_size)

# NG
# print ('range_image::integrateFarRanges.\n')
# range_image.IntegrateFarRanges (far_ranges)

# if (setUnseenToMaxRange)
#     range_image.setUnseenToMaxRange ();
print ('range_image::setUnseenToMaxRange.\n')
if setUnseenToMaxRange == True:
   range_image.SetUnseenToMaxRange ()

# Open 3D viewer and add point cloud
# pcl::visualization::PCLVisualizer viewer ("3D Viewer")
# viewer.setBackgroundColor (1, 1, 1)
# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler (range_image_ptr, 0, 0, 0);
# viewer.addPointCloud (range_image_ptr, range_image_color_handler, "range image");
# viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "range image");
# viewer.initCameraParameters ();
##
# viewer = pcl.PCLVisualizering("3D Viewer")
viewer = pcl.pcl_visualization.PCLVisualizering('3D Viewer')
viewer.SetBackgroundColor (1, 1, 1)
# NG
# range_image_color_handler = pcl.PointCloudColorHandlerCustoms[cpp.PointWithRange] (range_image, 0, 0, 0)
# range_image_color_handler = pcl.PointCloudColorHandlerCustoms (range_image, 0, 0, 0)
# range_image_color_handler = pcl.pcl_visualization.PointCloudColorHandleringCustom (range_image, 0, 0, 0)
range_image_color_handler = pcl.pcl_visualization.PointCloudColorHandleringCustom (point_cloud, 0, 0, 0)

viewer.AddPointCloud_ColorHandler (point_cloud, range_image_color_handler, b'range image')
# viewer.AddPointCloud (point_cloud, 'range image', 0)
# viewer.AddPointCloud (point_cloud)

time.sleep(1)
viewer.SetPointCloudRenderingProperties (pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 1, b'range image')
time.sleep(1)
viewer.InitCameraParameters ()
time.sleep(1)

# Show range image
# pcl::visualization::RangeImageVisualizer range_image_widget ("Range image");
# range_image_widget.showRangeImage (range_image);
range_image_widget = pcl.pcl_visualization.RangeImageVisualization()
range_image_widget.ShowRangeImage (range_image)

# Extract NARF keypoints
# pcl::RangeImageBorderExtractor range_image_border_extractor;
# pcl::NarfKeypoint narf_keypoint_detector (&range_image_border_extractor);
# narf_keypoint_detector.setRangeImage (&range_image);
# narf_keypoint_detector.getParameters ().support_size = support_size;
# narf_keypoint_detector.getParameters ().add_points_on_straight_edges = true;
# narf_keypoint_detector.getParameters ().distance_for_additional_points = 0.5;
# pcl::PointCloud<int> keypoint_indices;
# narf_keypoint_detector.compute (keypoint_indices);
# std::cout << "Found" << keypoint_indices.points.size () << "key points.\n";
range_image_border_extractor = pcl.RangeImageBorderExtractor()
narf_keypoint_detector = pcl.NarfKeypoint(range_image_border_extractor)
# narf_keypoint_detector.SetRangeImage (&range_image)

# pcl::PointCloud<int> keypoint_indices;
# narf_keypoint_detector.compute (keypoint_indices);
print("Found" + str(keypoint_indices.size) + "key points.\n")

# Show keypoints in range image widget
### Comment ###
# for (size_t i=0; ikeypoint_indices.points.size (); ++i)
# range_image_widget.markPoint (keypoint_indices.points[i] % range_image.width,
#                               keypoint_indices.points[i], range_image.width);
# for size_t i=0; ikeypoint_indices.points.size (); ++i:
#     range_image_widget.markPoint (keypoint_indices.points[i] % range_image.width, keypoint_indices.points[i], range_image.width)
###

# Show keypoints in 3D viewer
# pcl::PointCloud<pcl::PointXYZPtr> keypoints_ptr (new pclPointCloudpclPointXYZ);
# pcl::PointCloud<pcl::PointXYZ> &keypoints = keypoints_ptr;
# keypoints.points.resize (keypoint_indices.points.size ());
# for (size_t i=0; ikeypoint_indices.points.size (); ++i)
# keypoints.points[i].getVector3fMap () = range_image.points[keypoint_indices.points[i]].getVector3fMap ();
##
keypoints = pcl.KeyPoints()
keypoints.resize(keypoint_indices.size)
# for i in range(0, keypoint_indices.size, 1):
#     keypoints.points[i].getVector3fMap () = range_image[keypoint_indices.points[i]].getVector3fMap ()

# for x in range(-50, 50, 1):
# for y in range(-50, 50, 1):


# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (keypoints_ptr, 0, 255, 0);
# viewer.addPointCloud<pcl::PointXYZ> (keypoints_ptr, keypoints_color_handler, keypoints);
# viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, keypoints);
# keypoints_color_handler = pcl.PointCloudColorHandlerCustom (0, 255, 0)
# viewer.AddPointCloud<pcl::PointXYZ} (keypoints_ptr, keypoints_color_handler, keypoints)
# viewer.SetPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, keypoints);

## pcl
# keypoints_color_handler = pcl.visualization.PointCloudColorHandlerCustom[pcl.PointXYZ](keypoints_ptr, 0, 255, 0)
viewer.addPointCloud (keypoints_ptr, keypoints_color_handler, keypoints)
viewer.setPointCloudRenderingProperties (pcl.pcl_visualization.PCL_VISUALIZER_POINT_SIZE, 7, keypoints)
keypoints_color_handler = pcl.PointCloudColorHandlerCustom (0, 255, 0)
viewer.AddPointCloud (keypoints_ptr, keypoints_color_handler, keypoints)
viewer.SetPointCloudRenderingProperties (pcl.pcl_visualization.PCL_VISUALIZER_POINT_SIZE, 7, keypoints)

# Main loop
# # while (!viewer.wasStopped ())
#     # process GUI events
#     range_image_widget.spinOnce ()
#     viewer.spinOnce ()
#     # pcl_sleep(0.01);
# end

print("while")
while True:
    # process GUI events
    range_image_widget.SpinOnce ()
    viewer.SpinOnce ()


