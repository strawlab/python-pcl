
# // Point cloud library 
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/pcl_visualizer.h>
import pcl

# // Opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
import opencv

# These are track bar initial settings adjusted to the given pointcloud to make it completely visible.
# Need to be adjusted depending on the pointcloud and its xyz limits if used with new pointclouds.
int a = 22;
int b = 12;
int c=  10;

# // PCL Visualizer to view the pointcloud
# pcl::visualization::PCLVisualizer viewer ("Simple visualizing window");

# int   main (int argc, char** argv)
# {
#   pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
#   pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);

#   if (pcl::io::loadPLYFile<pcl::PointXYZRGBA> (argv[1], *cloud) == -1) //* load the ply file from command line
#   {
#       PCL_ERROR ("Couldn't load the file\n");
#       return (-1);
#   }
cloud = pcl.loadPLYFile("ism_test_cat.pcd")

# pcl::copyPointCloud( *cloud,*cloud_filtered);
cloud_filtered = cloud.copyPointCloud()

# float i
# float j
# float k

# cv::namedWindow("picture");
# // Creating trackbars uisng opencv to control the pcl filter limits
# cvCreateTrackbar("X_limit", "picture", &a, 30, NULL);
# cvCreateTrackbar("Y_limit", "picture", &b, 30, NULL);
# cvCreateTrackbar("Z_limit", "picture", &c, 30, NULL);

# // Starting the while loop where we continually filter with limits using trackbars and display pointcloud
char last_c = 0;
    while(true && (last_c != 27))
        # pcl::copyPointCloud(*cloud_filtered, *cloud);
        # // i,j,k Need to be adjusted depending on the pointcloud and its xyz limits if used with new pointclouds.
        i = 0.1 * a;
        j = 0.1 * b;
        k = 0.1 * c;
        # // Printing to ensure that the passthrough filter values are changing if we move trackbars. 
        # cout << "i = " << i << " j = " << j << " k = " << k << endl;
        # // Applying passthrough filters with XYZ limits

        pcl::PassThrough<pcl::PointXYZRGBA> pass;
        pass.setInputCloud (cloud);
        pass.setFilterFieldName ("y");
        //  pass.setFilterLimits (-0.1, 0.1);
        pass.setFilterLimits (-k, k);
        pass.filter (*cloud);

        pass.setInputCloud (cloud);
        pass.setFilterFieldName ("x");
        // pass.setFilterLimits (-0.1, 0.1);
        pass.setFilterLimits (-j, j);
        pass.filter (*cloud);

        pass.setInputCloud (cloud);
        pass.setFilterFieldName ("z");
        //  pass.setFilterLimits (-10, 10);
        pass.setFilterLimits (-i, i);
        pass.filter (*cloud);

        // Visualizing pointcloud
        viewer.addPointCloud (cloud, "scene_cloud");
        viewer.spinOnce();
        viewer.removePointCloud("scene_cloud");
    }

    return (0);
}
