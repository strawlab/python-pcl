# -*- coding: utf-8 -*-
# How to use Random Sample Consensus model
# http://pointclouds.org/documentation/tutorials/random_sample_consensus.php#random-sample-consensus

import numpy as np
import pcl
import random
import pcl.pcl_visualization
import math
import sys # モジュール属性 argv を取得するため


# boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
# {
#   // -----Open 3D viewer and add point cloud-----
#   boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
#   viewer->setBackgroundColor (0, 0, 0);
#   viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
#   viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
#   //viewer->addCoordinateSystem (1.0, "global");
#   viewer->initCameraParameters ();
#   return (viewer);
# }

argvs = sys.argv  # コマンドライン引数を格納したリストの取得
argc = len(argvs) # 引数の個数

# // initialize PointClouds
# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
# pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);
cloud = pcl.PointCloud()
final = pcl.PointCloud()

# // populate our PointCloud with points
# cloud->width    = 500;
# cloud->height   = 1;
# cloud->is_dense = false;
# cloud->points.resize (cloud->width * cloud->height);
# for (size_t i = 0; i < cloud->points.size (); ++i)
# {
#   if (pcl::console::find_argument (argc, argv, "-s") >= 0 || pcl::console::find_argument (argc, argv, "-sf") >= 0)
#   {
#     cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
#     cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
#     if (i % 5 == 0)
#       cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0);
#     else if(i % 2 == 0)
#       cloud->points[i].z =  sqrt( 1 - (cloud->points[i].x * cloud->points[i].x)
#                                     - (cloud->points[i].y * cloud->points[i].y));
#     else
#       cloud->points[i].z =  - sqrt( 1 - (cloud->points[i].x * cloud->points[i].x)
#                                       - (cloud->points[i].y * cloud->points[i].y));
#   }
#   else
#   {
#     cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
#     cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
#     if( i % 2 == 0)
#       cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0);
#     else
#       cloud->points[i].z = -1 * (cloud->points[i].x + cloud->points[i].y);
#   }
# }
points = np.zeros((1000, 3), dtype=np.float32)
RAND_MAX = 1024

for i in range(0, 1000):
    if argc > 1:
        if argvs[1] == "-s" or argvs[1] == "-sf":
            points[i][0] = 1024 * random.random () / (RAND_MAX + 1.0)
            points[i][1] = 1024 * random.random () / (RAND_MAX + 1.0)
            # print("x : " + str(points[i][0]))
            # print("y : " + str(points[i][1]))
            if i % 5 == 0:
                points[i][2] = 1024 * random.random () / (RAND_MAX + 1.0)
            elif i % 2 == 0:
                points[i][2] = math.sqrt(math.fabs(1 - (points[i][0] * points[i][0]) - (points[i][1] * points[i][1])))
                # points[i][2] = math.sqrt(1 )
            else:
                points[i][2] = -1 * math.sqrt(math.fabs(1 - (points[i][0] * points[i][0]) - (points[i][1] * points[i][1])))
                # points[i][2] = -1 * math.sqrt( 1 )
        else:
            points[i][0] = 1024 * random.random () / RAND_MAX
            points[i][1] = 1024 * random.random () / RAND_MAX
            if i % 2 == 0:
                points[i][2] = 1024 * random.random () / RAND_MAX
            else:
                points[i][2] = -1 * (points[i][0] + points[i][1])
    else:
        points[i][0] = 1024 * random.random () / RAND_MAX
        points[i][1] = 1024 * random.random () / RAND_MAX
        if i % 2 == 0:
            points[i][2] = 1024 * random.random () / RAND_MAX
        else:
            points[i][2] = -1 * (points[i][0] + points[i][1])

cloud.from_array(points)

# std::vector<int> inliers;
# // created RandomSampleConsensus object and compute the appropriated model
# pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ> (cloud));
# pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));
# if(pcl::console::find_argument (argc, argv, "-f") >= 0)
# {
# pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
# ransac.setDistanceThreshold (.01);
# ransac.computeModel();
# ransac.getInliers(inliers);
# }
# else if (pcl::console::find_argument (argc, argv, "-sf") >= 0 )
# {
# pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_s);
# ransac.setDistanceThreshold (.01);
# ransac.computeModel();
# ransac.getInliers(inliers);
# }
###
# inliers = vector[int]
model_s = pcl.SampleConsensusModelSphere(cloud)
model_p = pcl.SampleConsensusModelPlane(cloud)
if argc > 1:
    if argvs[1] == "-f":
        ransac = pcl.RandomSampleConsensus (model_p)
        ransac.set_DistanceThreshold (.01)
        ransac.computeModel()
        inliers = ransac.get_Inliers()
    elif argvs[1] == "-sf":
        ransac = pcl.RandomSampleConsensus (model_s)
        ransac.set_DistanceThreshold (.01)
        ransac.computeModel()
        inliers = ransac.get_Inliers()
    else:
        inliers = []
else:
    inliers = []


# // copies all inliers of the model computed to another PointCloud
# pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *final);
# final = pcl.copyPointCloud(cloud, inliers)
# pcl.copyPointCloud(cloud, inliers, final)
# final = cloud
print(str(len(inliers)))
if len(inliers) != 0:
    finalpoints = np.zeros((len(inliers), 3), dtype=np.float32)
    
    for i in range(0, len(inliers)):
        finalpoints[i][0] = cloud[inliers[i]][0]
        finalpoints[i][1] = cloud[inliers[i]][1]
        finalpoints[i][2] = cloud[inliers[i]][2]

    final.from_array(finalpoints)


# current(0.3.0) Windows Only Test
isWindows = False
if isWindows == True:
    # creates the visualization object and adds either our orignial cloud or all of the inliers
    # depending on the command line arguments specified.
    # boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    # if (pcl::console::find_argument (argc, argv, "-f") >= 0 || pcl::console::find_argument (argc, argv, "-sf") >= 0)
    #   viewer = simpleVis(final);
    # else
    #   viewer = simpleVis(cloud);

    if argc > 1:
        if argvs[1] == "-f" or argvs[1] == "-sf":
            viewer = pcl.pcl_visualization.PCLVisualizering('3D Viewer')
            viewer.SetBackgroundColor (0, 0, 0)
            viewer.AddPointCloud (final, b'sample cloud')
            viewer.SetPointCloudRenderingProperties (pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'sample cloud')
            viewer.InitCameraParameters ()
        else:
            viewer = pcl.pcl_visualization.PCLVisualizering('3D Viewer')
            viewer.SetBackgroundColor (0, 0, 0)
            viewer.AddPointCloud (cloud, b'sample cloud')
            viewer.SetPointCloudRenderingProperties (pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'sample cloud')
            viewer.InitCameraParameters ()
    else:
        viewer = pcl.pcl_visualization.PCLVisualizering('3D Viewer')
        viewer.SetBackgroundColor (0, 0, 0)
        viewer.AddPointCloud (cloud, b'sample cloud')
        viewer.SetPointCloudRenderingProperties (pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'sample cloud')
        viewer.InitCameraParameters ()


    # while (!viewer->wasStopped ())
    # {
    # viewer->spinOnce (100);
    # boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    # }
    isStopped = False
    while isStopped == False:
        isStopped = viewer.WasStopped()
        viewer.SpinOnce (100)
        # boost::this_thread::sleep (boost::posix_time::microseconds (100000));
else:
    pass

