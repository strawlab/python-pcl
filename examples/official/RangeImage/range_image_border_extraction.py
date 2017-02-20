# -*- coding: utf-8 -*-
# How to extract borders from range images
# http://pointclouds.org/documentation/tutorials/range_image_border_extraction.php#range-image-border-extraction

import pcl
import numpy as np
import random
import argparse

import pcl.pcl_visualization

# -----Parameters-----
angular_resolution = 0.5
# pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
coordinate_frame = pcl.CythonCoordinateFrame_Type.CAMERA_FRAME
setUnseenToMaxRange = False;

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

# -----Main-----
# // --------------
# int main (int argc, char** argv)
# // -----Parse Command Line Arguments-----
# if (pcl::console::find_argument (argc, argv, "-h") >= 0)
# {
# printUsage (argv[0]);
# return 0;
# }
# if (pcl::console::find_argument (argc, argv, "-m") >= 0)
# {
# setUnseenToMaxRange = true;
# cout << "Setting unseen values in range image to maximum range readings.\n";
# }
# int tmp_coordinate_frame;
# if (pcl::console::parse (argc, argv, "-c", tmp_coordinate_frame) >= 0)
# {
# coordinate_frame = pcl::RangeImage::CoordinateFrame (tmp_coordinate_frame);
# cout << "Using coordinate frame "<< (int)coordinate_frame<<".\n";
# }
# if (pcl::console::parse (argc, argv, "-r", angular_resolution) >= 0)
# cout << "Setting angular resolution to "<<angular_resolution<<"deg.\n";

parser = argparse.ArgumentParser(description='StrawPCL example: range image border extraction')
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

setUnseenToMaxRange = args.UnseenToMaxRange
# coordinate_frame = pcl.RangeImage.CoordinateFrame (args.CoordinateFrame)
# angular_resolution = pcl.deg2rad (args.AngularResolution)

# // -----Read pcd file or create example point cloud if not given-----
# pcl::PointCloud<PointType>::Ptr point_cloud_ptr (new pcl::PointCloud<PointType>);
# pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;
point_cloud = pcl.PointCloud()

# pcl::PointCloud<pcl::PointWithViewpoint> far_ranges;
# Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
# scene_sensor_pose = (Eigen::Affine3f::Identity ())

# std::vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument (argc, argv, "pcd");
pcd_filename_indices = ''
# pcd_filename_indices = [0, 0, 0]

##
# if (!pcd_filename_indices.empty ())
# {
# std::string filename = argv[pcd_filename_indices[0]];
# if (pcl::io::loadPCDFile (filename, point_cloud) == -1)
# {
#   cout << "Was not able to open file \""<<filename<<"\".\n";
#   printUsage (argv[0]);
#   return 0;
# }
# scene_sensor_pose = Eigen::Affine3f (Eigen::Translation3f (point_cloud.sensor_origin_[0],
#                                                            point_cloud.sensor_origin_[1],
#                                                            point_cloud.sensor_origin_[2])) *
#                     Eigen::Affine3f (point_cloud.sensor_orientation_);
# std::string far_ranges_filename = pcl::getFilenameWithoutExtension (filename)+"_far_ranges.pcd";
# if (pcl::io::loadPCDFile(far_ranges_filename.c_str(), far_ranges) == -1)
#   std::cout << "Far ranges file \""<<far_ranges_filename<<"\" does not exists.\n";
# }
# else
# {
# cout << "\nNo *.pcd file given => Genarating example point cloud.\n\n";
# for (float x=-0.5f; x<=0.5f; x+=0.01f)
# {
#   for (float y=-0.5f; y<=0.5f; y+=0.01f)
#   {
#     PointType point;  point.x = x;  point.y = y;  point.z = 2.0f - y;
#     point_cloud.points.push_back (point);
#   }
# }
# point_cloud.width = (int) point_cloud.points.size ();  point_cloud.height = 1;
# }

if len(pcd_filename_indices) != 0:
    # point_cloud = pcl.load(argv[0])
    point_cloud = pcl.load('test_pcd.pcd')
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


##
# Create RangeImage from the PointCloud
noise_level = 0.0
min_range = 0.0
border_size = 1

# boost::shared_ptr<pcl::RangeImage> range_image_ptr (new pcl::RangeImage);
# pcl::RangeImage& range_image = *range_image_ptr;   
# range_image.createFromPointCloud (point_cloud, angular_resolution, pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
#                                scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
# range_image.integrateFarRanges (far_ranges);
# if (setUnseenToMaxRange)
# range_image.setUnseenToMaxRange ();
##
range_image = point_cloud.make_RangeImage()
range_image.CreateFromPointCloud (point_cloud, 
                        angular_resolution, pcl.deg2rad (360.0), pcl.deg2rad (180.0), 
                        coordinate_frame, noise_level, min_range, border_size)
print ('range_image::integrateFarRanges.\n')
if setUnseenToMaxRange == True:
    range_image.SetUnseenToMaxRange ()


# -----Open 3D viewer and add point cloud-----
# pcl::visualization::PCLVisualizer viewer ("3D Viewer");
# viewer.setBackgroundColor (1, 1, 1);
# viewer.addCoordinateSystem (1.0f, "global");
# pcl::visualization::PointCloudColorHandlerCustom<PointType> point_cloud_color_handler (point_cloud_ptr, 0, 0, 0);
# viewer.addPointCloud (point_cloud_ptr, point_cloud_color_handler, "original point cloud");
# // PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler (range_image_ptr, 150, 150, 150);
# // viewer.addPointCloud (range_image_ptr, range_image_color_handler, "range image");
# // viewer.setPointCloudRenderingProperties (PCL_VISUALIZER_POINT_SIZE, 2, "range image");
##
# viewer = pcl.pcl_visualization.PCLVisualizering('3D Viewer')
viewer = pcl.pcl_visualization.PCLVisualizering()
viewer.SetBackgroundColor (1.0, 1.0, 1.0)
range_image_color_handler = pcl.pcl_visualization.PointCloudColorHandleringCustom (point_cloud, 0, 0, 0)
# viewer.AddPointCloud (range_image, range_image_color_handler, 'range image')
viewer.AddPointCloud (point_cloud, 'range image')
# viewer.AddPointCloud_ColorHandler
# viewer.SetPointCloudRenderingProperties (pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 1, propName = 'range image')
# NG - ([setPointCloudRenderingProperties] Could not find any PointCloud datasets with id <cloud>!)
# viewer.SetPointCloudRenderingProperties (pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 1)


##
# Extract borders
# pcl::RangeImageBorderExtractor border_extractor (&range_image);
# pcl::PointCloud<pcl::BorderDescription> border_descriptions;
# border_extractor.compute (border_descriptions);

##
# // ----------------------------------
# // -----Show points in 3D viewer-----
# // ----------------------------------
# pcl::PointCloud<pcl::PointWithRange>::Ptr border_points_ptr(new pcl::PointCloud<pcl::PointWithRange>),
#                                         veil_points_ptr(new pcl::PointCloud<pcl::PointWithRange>),
#                                         shadow_points_ptr(new pcl::PointCloud<pcl::PointWithRange>);
# pcl::PointCloud<pcl::PointWithRange>& border_points = *border_points_ptr,
#                                   & veil_points = * veil_points_ptr,
#                                   & shadow_points = *shadow_points_ptr;
# for (int y=0; y< (int)range_image.height; ++y)
# {
# for (int x=0; x< (int)range_image.width; ++x)
# {
#   if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__OBSTACLE_BORDER])
#     border_points.points.push_back (range_image.points[y*range_image.width + x]);
#   if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__VEIL_POINT])
#     veil_points.points.push_back (range_image.points[y*range_image.width + x]);
#   if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__SHADOW_BORDER])
#     shadow_points.points.push_back (range_image.points[y*range_image.width + x]);
# }
# }
# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> border_points_color_handler (border_points_ptr, 0, 255, 0);
# viewer.addPointCloud<pcl::PointWithRange> (border_points_ptr, border_points_color_handler, "border points");
# viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "border points");
# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> veil_points_color_handler (veil_points_ptr, 255, 0, 0);
# viewer.addPointCloud<pcl::PointWithRange> (veil_points_ptr, veil_points_color_handler, "veil points");
# viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "veil points");
# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> shadow_points_color_handler (shadow_points_ptr, 0, 255, 255);
# viewer.addPointCloud<pcl::PointWithRange> (shadow_points_ptr, shadow_points_color_handler, "shadow points");
# viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "shadow points");
# 



##
# //-------------------------------------
# // -----Show points on range image-----
# // ------------------------------------
# pcl::visualization::RangeImageVisualizer* range_image_borders_widget = NULL;
# range_image_borders_widget =
# pcl::visualization::RangeImageVisualizer::getRangeImageBordersWidget (range_image, -std::numeric_limits<float>::infinity (), std::numeric_limits<float>::infinity (), false,
#                                                                       border_descriptions, "Range image with borders");
# 



##
# //--------------------
# // -----Main loop-----
# //--------------------
# while (!viewer.wasStopped ())
# {
# range_image_borders_widget->spinOnce ();
# viewer.spinOnce ();
# pcl_sleep(0.01);
# }

flag = True
while flag:
    flag != viewer.WasStopped ()
    viewer.SpinOnce ()
end

