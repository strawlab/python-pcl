# -*- coding: utf-8 -*-
# http://virtuemarket-lab.blogspot.jp/2015/03/harris.html
import pcl
import numpy as np
import pcl.pcl_visualization

# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
# pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *cloud);
# cloud = pcl.load("table_scene_mug_stereo_textured.pcd")
# cloud = pcl.load('./examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')
cloud = pcl.load('./tests/tutorials/bunny.pcd')
print("cloud points : " + str(cloud.size))

# pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> detector;
# detector.setNonMaxSupression (true);
# detector.setRadius (0.01);
# //detector.setRadiusSearch (100);
# detector.setInputCloud(cloud);
# pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());
# detector.compute(*keypoints);
###
detector = cloud.make_HarrisKeypoint3D()
detector.set_NonMaxSupression (True)
detector.set_Radius (0.01)
# detector.set_NonMaxSupression (False)
# detector.set_RadiusSearch (100)
keypoints = detector.compute()

# std::cout << "keypoints detected: " << keypoints->size() << std::endl;
print("keypoints detected: " + str(keypoints.size))

# pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints3D(new pcl::PointCloud<pcl::PointXYZ>());
# pcl::PointXYZ tmp;
# double max = 0,min=0;
# for(pcl::PointCloud<pcl::PointXYZI>::iterator i = keypoints->begin(); i!= keypoints->end(); i++)
#     tmp = pcl::PointXYZ((*i).x,(*i).y,(*i).z);
#     if ((*i).intensity>max )
#         std::cout << (*i) << " coords: " << (*i).x << ";" << (*i).y << ";" << (*i).z << std::endl;
#         max = (*i).intensity;
#     if ((*i).intensity<min)
#         min = (*i).intensity;
#     keypoints3D->push_back(tmp);
# 
# std::cout << "maximal responce: "<< max << " min responce:  "<< min<<std::endl;
###
keypoints3D = pcl.PointCloud()
max = -999
min = 999

count = 0
points = np.zeros((keypoints.size, 3), dtype=np.float32)
# Generate the data
for i in range(0, keypoints.size):
    # set Point Plane
    points[i][0] = keypoints[i][0]
    points[i][1] = keypoints[i][1]
    points[i][2] = keypoints[i][2]
    intensity = keypoints[i][3]
    if intensity > max:
        print("coords: " + str(keypoints[i][0]) + ";" + str(keypoints[i][1]) + ";" + str(keypoints[i][2]) )
        max = intensity
    
    if intensity < min:
        min = intensity
    
    count = count + 1

points.resize(count, 3)
print(points)
keypoints3D.from_array(points)
print("maximal responce: " + str(max) + " min responce:  " +  str(min) )

# //show point cloud
# pcl::visualization::PCLVisualizer viewer ("3D Viewer");
# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pccolor(cloud, 255, 255, 255);
# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> kpcolor(keypoints3D, 255, 0, 0);
# viewer.addPointCloud(cloud, pccolor, "testimg.png");
# viewer.addPointCloud(keypoints3D, kpcolor,"keypoints.png");
# viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints.png");
##
viewer = pcl.pcl_visualization.PCLVisualizering('3D Viewer')
pccolor = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 255, 255, 255)
kpcolor = pcl.pcl_visualization.PointCloudColorHandleringCustom(keypoints3D, 255, 0, 0)
# OK
viewer.AddPointCloud_ColorHandler(cloud, pccolor)
viewer.AddPointCloud_ColorHandler(keypoints3D, kpcolor, b'keypoints')

# viewer.AddPointCloud_ColorHandler(cloud, pccolor, "testimg.png", 0)
# viewer.AddPointCloud_ColorHandler(keypoints3D, kpcolor, str('keypoints.png'), 0)
# need? : AddPointCloud_ColorHandler Function Succeded
# viewer.SetPointCloudRenderingProperties (pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 7, b'keypoints.png')
###


# while (!viewer.wasStopped ())
# {
#     viewer.spinOnce();
#     pcl_sleep (0.01);
# }
flag = True
while flag:
    flag != viewer.WasStopped()
    viewer.SpinOnce()
    # pcl_sleep (0.01)
    # pass
end
