# Normal Estimation Using Integral Images
# http://pointclouds.org/documentation/tutorials/normal_estimation_using_integral_images.php#normal-estimation-using-integral-images

import pcl
import pcl.pcl_visualization


def main():
    # cloud = pcl.load('table_scene_mug_stereo_textured.pcd')
    cloud = pcl.load(
        './examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')

    print('load table_scene_mug_stereo_textured.pcd')

    # estimate normals
    # pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    # pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    print('make_IntegralImageNormalEstimation: ')
    ne = cloud.make_IntegralImageNormalEstimation()

    print('set_NormalEstimation_Method_AVERAGE_3D_GRADIENT: ')
    ne.set_NormalEstimation_Method_AVERAGE_3D_GRADIENT()
    print('set_MaxDepthChange_Factor: ')
    ne.set_MaxDepthChange_Factor(0.02)
    print('set_NormalSmoothingSize: ')
    ne.set_NormalSmoothingSize(10.0)
    print('set OK')
    print('compute - start')
    normals = ne.compute()
    print('compute - end')
    print(str(normals.size))
    print(normals.to_array())

    # visualize normals
    viewer = pcl.pcl_visualization.PCLVisualizering()
    viewer.SetBackgroundColor(0.0, 0.0, 0.5)
    # viewer.addPointCloudNormals<pcl::PointXYZ,pcl::Normal>(cloud, normals, b'normals');
    viewer.AddPointCloud(cloud)
    viewer.AddPointCloudNormals(cloud, normals, 10, 0.05, b'normals')

    flag = True
    while (flag):
        flag = not(viewer.WasStopped())
        viewer.SpinOnce()


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
