#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>

#include "minipcl.h"

void mpcl_compute_normals(pcl::PointCloud<pcl::PointXYZ> &cloud,
                          int ksearch,
                          pcl::PointCloud<pcl::Normal> &out)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud.makeShared());
    ne.setKSearch (ksearch);
    ne.compute (out);
}

