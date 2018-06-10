# -*- coding: utf-8 -*-
# http://virtuemarket-lab.blogspot.jp/2015/03/sift.html
import pcl
import numpy as np
import pcl.pcl_visualization

# pcl::PointCloud<pcl::PointNormal>::Ptr Surface_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
# {
#     pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
#     ne.setInputCloud (cloud);//法線の計算を行いたい点群を指定する
# 
#     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());//KDTREEを作る
#     ne.setSearchMethod (tree);//検索方法にKDTREEを指定する
# 
#     pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointNormal>);//法線情報を入れる変数
# 
#     ne.setRadiusSearch (0.5);//検索する半径を指定する
# 
#     ne.compute (*cloud_normals);//法線情報の出力先を指定する
# 
#     return cloud_normals;
# }
def Surface_normals(cloud):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(0.5)
    # NG
    print('test - a')
    print(ne)
    cloud_normals = ne.compute()
    print('test - b')
    return cloud_normals

###

# pcl::PointCloud<pcl::PointWithScale> Extract_SIFT(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals)
# {    
#     // SIFT特徴量計算のためのパラメータ
#     const float min_scale = 0.01f;
#     const int n_octaves = 3;
#     const int n_scales_per_octave = 4;
#     const float min_contrast = 0.001f;
#     pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
#     pcl::PointCloud<pcl::PointWithScale> result;
#     pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal> ());
#     sift.setSearchMethod(tree);
#     sift.setScales(min_scale, n_octaves, n_scales_per_octave);
#     sift.setMinimumContrast(0.00);
#     sift.setInputCloud(cloud_normals);
#     sift.compute(result);
#     std::cout << "No of SIFT points in the result are " << result.points.size () << std::endl;
# 
#     return result;
# }
def Extract_SIFT(cloud, cloud_normals):
    min_scale = 0.01
    n_octaves = 3
    n_scales_per_octave = 4
    min_contrast = 0.001
    
    sift = cloud_makeSIFTKeypoint()
    sift.set_SearchMethod(tree)
    sift.set_Scales(min_scale, n_octaves, n_scales_per_octave)
    sift.set_MinimumContrast(0.00)
    result = sift.compute()
    print('No of SIFT points in the result are ' + str(result.size))
    return result

###

# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
# pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *cloud);
# cloud = pcl.load("table_scene_mug_stereo_textured.pcd")
# cloud = pcl.load('./examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')
cloud = pcl.load('./tests/tutorials/bunny.pcd')
print("cloud points : " + str(cloud.size))

# pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointNormal>);
cloud_normals = Surface_normals(cloud)

# XYZの情報をcloudからSurface_normals(cloud)にXYZとして加える
# for(size_t i = 0; i < cloud_normals->points.size(); ++i)
# {
#     cloud_normals->points[i].x = cloud->points[i].x;
#     cloud_normals->points[i].y = cloud->points[i].y;
#     cloud_normals->points[i].z = cloud->points[i].z;
# }

# // 視覚化のためSIFT計算の結果をcloud_tempにコピー
# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZ>);
# copyPointCloud(Extract_SIFT(cloud, cloud_normals), *cloud_temp);
# std::cout << "SIFT points in the cloud_temp are " << cloud_temp->points.size () << std::endl;
cloud_temp = Extract_SIFT(cloud, cloud_normals)
print('SIFT points in the cloud_temp are ' + str(cloud_temp.size))

# // 入力点群と計算された特徴点を表示
# pcl::visualization::PCLVisualizer viewer("PCL Viewer");
# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (cloud_temp, 0, 255, 0);
# pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler (cloud, 255, 0, 0);
# viewer.setBackgroundColor( 0.0, 0.0, 0.0 );
# viewer.addPointCloud(cloud, cloud_color_handler, "cloud");
# viewer.addPointCloud(cloud_temp, keypoints_color_handler, "keypoints");
# viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");

flag = True
while flag:
    flag != viewer.WasStopped()
    viewer.SpinOnce()
    # pcl_sleep (0.01)
    # pass
end
