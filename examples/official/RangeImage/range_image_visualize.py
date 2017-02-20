# -*- coding: utf-8 -*-
# How to visualize a range image
# http://www.pointclouds.org/documentation/tutorials/range_image_visualization.php#range-image-visualization

import pcl
import numpy as np
import random
import argparse

import pcl.pcl_visualization

# -----Parameters-----
angular_resolution_x = 0.5
angular_resolution_y = 0.5
coordinate_frame = pcl.CythonCoordinateFrame_Type.CAMERA_FRAME
live_update = False

# -----Help-----
# void printUsage (const char* progName)
# {
#   std::cout << "\n\nUsage: "<<progName<<" [options] <scene.pcd>\n\n"
#             << "Options:\n"
#             << "-------------------------------------------\n"
#             << "-r <float>   angular resolution in degrees (default "<<angular_resolution<<")\n"
#             << "-c <int>     coordinate frame (default "<< (int)coordinate_frame<<")\n"
#             << "-m           Treat all unseen points to max range\n"
#             << "-h           this help\n"
#             << "\n\n";
# }

# void setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
# {
#   Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
#   Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f(0, 0, 1) + pos_vector;
#   Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f(0, -1, 0);
#   viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
#                             look_at_vector[0], look_at_vector[1], look_at_vector[2],
#                             up_vector[0], up_vector[1], up_vector[2]);
# }

# -----Main-----
# int main (int argc, char** argv)
# // -----Parse Command Line Arguments-----
#   if (pcl::console::find_argument (argc, argv, "-h") >= 0)
#   {
#     printUsage (argv[0]);
#     return 0;
#   }
#   if (pcl::console::find_argument (argc, argv, "-l") >= 0)
#   {
#     live_update = true;
#     std::cout << "Live update is on.\n";
#   }
#   if (pcl::console::parse (argc, argv, "-rx", angular_resolution_x) >= 0)
#     std::cout << "Setting angular resolution in x-direction to "<<angular_resolution_x<<"deg.\n";
#   if (pcl::console::parse (argc, argv, "-ry", angular_resolution_y) >= 0)
#     std::cout << "Setting angular resolution in y-direction to "<<angular_resolution_y<<"deg.\n";
#   int tmp_coordinate_frame;
#   if (pcl::console::parse (argc, argv, "-c", tmp_coordinate_frame) >= 0)
#   {
#     coordinate_frame = pcl::RangeImage::CoordinateFrame (tmp_coordinate_frame);
#     std::cout << "Using coordinate frame "<< (int)coordinate_frame<<".\n";
#   }
#   angular_resolution_x = pcl::deg2rad (angular_resolution_x);
#   angular_resolution_y = pcl::deg2rad (angular_resolution_y);

parser = argparse.ArgumentParser(description='StrawPCL example: How to visualize a range image')
parser.add_argument('--UnseenToMaxRange', '-m', default=True, type=bool,
                    help='Setting unseen values in range image to maximum range readings')
parser.add_argument('--CoordinateFrame', '-c', default=-1, type=int,
                    help='Using coordinate frame = ')
parser.add_argument('--AngularResolution', '-r', default=0, type=int,
                    help='Setting angular resolution to = ')
parser.add_argument('--Help',
                    help='Usage: narf_keypoint_extraction.py [options] <scene.pcd>\n\n'
                    'Options:\n'
                    '-------------------------------------------\n'
                    '-r <float>   angular resolution in degrees (default = angular_resolution)\n'
                    '-c <int>     coordinate frame (default = coordinate_frame)\n'
                    '-m           Treat all unseen points as max range\n'
                    '-h           this help\n\n\n;')

args = parser.parse_args()

#   // -----Read pcd file or create example point cloud if not given-----
#   pcl::PointCloud<PointType>::Ptr point_cloud_ptr (new pcl::PointCloud<PointType>);
#   pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;
#   Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
#   std::vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument (argc, argv, "pcd");
#   if (!pcd_filename_indices.empty ())
#   {
#     std::string filename = argv[pcd_filename_indices[0]];
#     if (pcl::io::loadPCDFile (filename, point_cloud) == -1)
#     {
#       std::cout << "Was not able to open file \""<<filename<<"\".\n";
#       printUsage (argv[0]);
#       return 0;
#     }
#     scene_sensor_pose = Eigen::Affine3f (Eigen::Translation3f (point_cloud.sensor_origin_[0],
#                                                              point_cloud.sensor_origin_[1],
#                                                              point_cloud.sensor_origin_[2])) *
#                         Eigen::Affine3f (point_cloud.sensor_orientation_);
#   }
#   else
#   {
#     std::cout << "\nNo *.pcd file given => Genarating example point cloud.\n\n";
#     for (float x=-0.5f; x<=0.5f; x+=0.01f)
#     {
#       for (float y=-0.5f; y<=0.5f; y+=0.01f)
#       {
#         PointType point;  point.x = x;  point.y = y;  point.z = 2.0f - y;
#         point_cloud.points.push_back (point);
#       }
#     }
#     point_cloud.width = (int) point_cloud.points.size ();  point_cloud.height = 1;
#   }
pcd_filename_indices = ''
if len(pcd_filename_indices) != 0:
    # point_cloud = pcl.load(argv[0])
    point_cloud = pcl.load('./examples/official/IO/test_pcd.pcd')
    far_ranges_filename = 'test_pcd.pcd'

    far_ranges = pcl.load_PointWithViewpoint(far_ranges_filename)
else:
    setUnseenToMaxRange = True
    print ('No *.pcd file given = Genarating example point cloud.\n')

    count = 0
    points = np.zeros((100 * 100, 3), dtype=np.float32)

    # float NG
    for x in range(-50, 50, 1):
        for y in range(-50, 50, 1):
            points[count][0] = x * 0.01
            points[count][1] = y * 0.01
            points[count][2] = 2.0 - y * 0.01
            count = count + 1
    
    point_cloud = pcl.PointCloud()
    point_cloud.from_array(points)
    
    far_ranges = pcl.PointCloud_PointWithViewpoint()

# // -----------------------------------------------
# // -----Create RangeImage from the PointCloud-----
# // -----------------------------------------------
# float noise_level = 0.0;
# float min_range = 0.0f;
# int border_size = 1;
# boost::shared_ptr<pcl::RangeImage> range_image_ptr(new pcl::RangeImage);
# pcl::RangeImage& range_image = *range_image_ptr;   
# range_image.createFromPointCloud (point_cloud, angular_resolution_x, angular_resolution_y,
#                                 pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
#                                 scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
noise_level = 0.0
min_range = 0.0
border_size = 1
range_image = point_cloud.make_RangeImage()
range_image.CreateFromPointCloud (point_cloud, 
                        angular_resolution_x, pcl.deg2rad (360.0), pcl.deg2rad (180.0), 
                        coordinate_frame, noise_level, min_range, border_size)
print ('range_image::integrateFarRanges.\n')

# // --------------------------------------------
# // -----Open 3D viewer and add point cloud-----
# // --------------------------------------------
# pcl::visualization::PCLVisualizer viewer ("3D Viewer");
# viewer.setBackgroundColor (1, 1, 1);
# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler (range_image_ptr, 0, 0, 0);
# viewer.addPointCloud (range_image_ptr, range_image_color_handler, "range image");
# viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "range image");
# //viewer.addCoordinateSystem (1.0f, "global");
# //PointCloudColorHandlerCustom<PointType> point_cloud_color_handler (point_cloud_ptr, 150, 150, 150);
# //viewer.addPointCloud (point_cloud_ptr, point_cloud_color_handler, "original point cloud");
# viewer.initCameraParameters ();
# setViewerPose(viewer, range_image.getTransformationToWorldSystem ());
viewer = pcl.pcl_visualization.PCLVisualizering()
viewer.SetBackgroundColor (1.0, 1.0, 1.0)
range_image_color_handler = pcl.pcl_visualization.PointCloudColorHandleringCustom (point_cloud, 0, 0, 0)
cloudname = str('cloud')
viewer.AddPointCloud (range_image, range_image_color_handler, cloudname)
viewer.SetPointCloudRenderingProperties (pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 1, cloudname)


# // --------------------------
# // -----Show range image-----
# // --------------------------
# pcl::visualization::RangeImageVisualizer range_image_widget ("Range image");
# range_image_widget.showRangeImage (range_image);
range_image_widget = pcl.pcl_visualization.RangeImageVisualization()
range_image_widget.ShowRangeImage (range_image)


#   //--------------------
#   // -----Main loop-----
#   //--------------------
#   while (!viewer.wasStopped ())
#   {
#     range_image_widget.spinOnce ();
#     viewer.spinOnce ();
#     pcl_sleep (0.01);
#     
#     if (live_update)
#     {
#       scene_sensor_pose = viewer.getViewerPose();
#       range_image.createFromPointCloud (point_cloud, angular_resolution_x, angular_resolution_y,
#                                         pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
#                                         scene_sensor_pose, pcl::RangeImage::LASER_FRAME, noise_level, min_range, border_size);
#       range_image_widget.showRangeImage (range_image);
#     }
#   }

