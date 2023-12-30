# -*- coding: utf-8 -*-
# Point cloud library
import pcl
import pcl.pcl_visualization

# Opencv
# import opencv
import cv2


def main():
    # These are track bar initial settings adjusted to the given pointcloud to make it completely visible.
    # Need to be adjusted depending on the pointcloud and its xyz limits if used with new pointclouds.
    # int a = 22;
    # int b = 12;
    # int c=  10;
    a = 22
    b = 12
    c = 10

    # PCL Visualizer to view the pointcloud
    # pcl::visualization::PCLVisualizer viewer ("Simple visualizing window");
    viewer = pcl.pcl_visualization.PCLVisualizering()

    # int main (int argc, char** argv)
    # {
    #   pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    #   pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);
    #   if (pcl::io::loadPLYFile<pcl::PointXYZRGBA> (argv[1], *cloud) == -1) //* load the ply file from command line
    #   {
    #       PCL_ERROR ("Couldn't load the file\n");
    #       return (-1);
    #   }
    # cloud = pcl.load("lamppost.pcd")
    cloud = pcl.load("Tile_173078_LD_010_017_L22.obj")

    # pcl::copyPointCloud( *cloud,*cloud_filtered);
    # cloud_filtered = cloud.copyPointCloud()
    cloud_filtered = cloud

    # float i
    # float j
    # float k

    # cv::namedWindow("picture");
    # // Creating trackbars uisng opencv to control the pcl filter limits
    # cvCreateTrackbar("X_limit", "picture", &a, 30, NULL);
    # cvCreateTrackbar("Y_limit", "picture", &b, 30, NULL);
    # cvCreateTrackbar("Z_limit", "picture", &c, 30, NULL);
    # cv2.CreateTrackbar("X_limit", "picture", a, 30)
    # cv2.CreateTrackbar("Y_limit", "picture", b, 30)
    # cv2.CreateTrackbar("Z_limit", "picture", c, 30)

    # // Starting the while loop where we continually filter with limits using trackbars and display pointcloud
    # char last_c = 0;
    last_c = 0

    # while(true && (last_c != 27))
    while last_c != 27:

        # pcl::copyPointCloud(*cloud_filtered, *cloud);
        # // i,j,k Need to be adjusted depending on the pointcloud and its xyz limits if used with new pointclouds.
        i = 0.1 * a
        j = 0.1 * b
        k = 0.1 * c

        # Printing to ensure that the passthrough filter values are changing if we move trackbars.
        # cout << "i = " << i << " j = " << j << " k = " << k << endl;
        print("i = " + str(i) + " j = " + str(j) + " k = " + str(k))

        # Applying passthrough filters with XYZ limits
        # pcl::PassThrough<pcl::PointXYZRGBA> pass;
        # pass.setInputCloud (cloud);
        # pass.setFilterFieldName ("y");
        # //  pass.setFilterLimits (-0.1, 0.1);
        # pass.setFilterLimits (-k, k);
        # pass.filter (*cloud);
        pass_th = cloud.make_passthrough_filter()
        pass_th.set_filter_field_name("y")
        pass_th.set_filter_limits(-k, k)
        cloud = pass_th.filter()

        # pass.setInputCloud (cloud);
        # pass.setFilterFieldName ("x");
        # // pass.setFilterLimits (-0.1, 0.1);
        # pass.setFilterLimits (-j, j);
        # pass.filter (*cloud);
        # pass_th.setInputCloud(cloud)
        pass_th.set_filter_field_name("x")
        pass_th.set_filter_limits(-j, j)
        cloud = pass_th.filter()

        # pass.setInputCloud (cloud);
        # pass.setFilterFieldName ("z");
        # //  pass.setFilterLimits (-10, 10);
        # pass.setFilterLimits (-i, i);
        # pass.filter (*cloud);
        # pass_th.setInputCloud(cloud)
        pass_th.set_filter_field_name("z")
        pass_th.set_filter_limits(-10, 10)
        cloud = pass_th.filter()

        # // Visualizing pointcloud
        # viewer.addPointCloud (cloud, "scene_cloud");
        # viewer.spinOnce();
        # viewer.removePointCloud("scene_cloud");
        viewer.AddPointCloud(cloud, b'scene_cloud', 0)
        viewer.SpinOnce()
        # viewer.Spin()
        viewer.RemovePointCloud(b'scene_cloud', 0)


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
