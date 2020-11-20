# -*- coding: utf-8 -*-
# http://pointclouds.org/documentation/tutorials/voxel_grid.php#voxelgrid
# http://derivecv.tumblr.com/post/13631147204
# http://nisot0710.blogspot.jp/2014/09/pclvoxelgridpclpointcloud2.html
# PCLPointCloud2 is 1.7.2

import pcl


def main():
    # pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    # pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ> ());
    #
    # pcl::PCDReader reader;
    # reader.read("pcdfilename", *cloud);
    cloud = pcl.load('./examples/pcldata/tutorials/table_scene_lms400.pcd')

    # std::cerr<<"PointCloud befor filtering: " << cloud->width * cloud->height << "data points ( " << pcl::getFieldsList (*cloud) << ").";
    # print('PointCloud befor filtering: ' + str(cloud.width * cloud.height) + 'data points ( ' + pcl.getFieldsList (cloud) + ').')

    # pcl::VoxelGrid<pcl::PointXYZ> sor;
    # sor.setInputCloud(cloud);
    # sor.setLeafSize(0.1f, 0.1f, 0.1f);
    # sor.filter(*cloud_filtered);
    sor = cloud.make_voxel_grid_filter()
    sor.set_leaf_size(0.01, 0.01, 0.01)
    cloud_filtered = sor.filter()

    # std::cerr<<"PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << "data points (" << pcl::getFieldsList(*cloud_filtered) <<").";
    # print('PointCloud after filtering: ' + str(cloud_filtered.width * cloud_filtered.height) + 'data points ( ' + pcl.getFieldsList (cloud) + ').')

    # pcl::PCDWriter writer;
    # writer.write("savefilename", *cloud_filtered, false);
    pcl.save(cloud_filtered, 'table_scene_lms400_voxelfilter.pcd')


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
