#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>

#include <Eigen/Dense>

#include "minipcl.h"

// set ksearch and radius to < 0 to disable 
void mpcl_compute_normals(pcl::PointCloud<pcl::PointXYZ> &cloud,
                          int ksearch,
                          double searchRadius,
                          pcl::PointCloud<pcl::Normal> &out)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud.makeShared());
    if (ksearch >= 0)
        ne.setKSearch (ksearch);
    if (searchRadius >= 0.0)
        ne.setRadiusSearch (searchRadius);
    ne.compute (out);
}

void mpcl_sacnormal_set_axis(pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> &sac,
                             double ax, double ay, double az)
{
    Eigen::Vector3f vect(ax,ay,az);
    sac.setAxis(vect);
}

void mpcl_extract(pcl::PointCloud<pcl::PointXYZ>::Ptr &incloud,
                  pcl::PointCloud<pcl::PointXYZ> *outcloud,
                  pcl::PointIndices *indices,
                  bool negative)
{
    pcl::PointIndices::Ptr indicesptr (indices);
    pcl::ExtractIndices<pcl::PointXYZ> ext;
    ext.setInputCloud(incloud);
    ext.setIndices(indicesptr);
    ext.setNegative(negative);
    ext.filter(*outcloud);
}
