# Removing outliers using a StatisticalOutlierRemoval filter
# http://pointclouds.org/documentation/tutorials/statistical_outlier.php#statistical-outlier-removal

import pcl

p = pcl.load('./examples/pcldata/tutorials/table_scene_lms400.pcd')


fil = p.make_statistical_outlier_filter()
fil.set_mean_k(50)
fil.set_std_dev_mul_thresh(1.0)

pcl.save(fil.filter(), "table_scene_lms400_inliers.pcd")

fil.set_negative(True)
pcl.save(fil.filter(), "table_scene_lms400_outliers.pcd")
