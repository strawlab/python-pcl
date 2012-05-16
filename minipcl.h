#ifndef _MINIPCL_H_
#define _MINIPCL_H_

#include <pcl/PointIndices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

void mpcl_simple_segment_plane(pcl::PointIndices &inliers,
                               pcl::ModelCoefficients &coefficients,
                               pcl::PointCloud<pcl::PointXYZ> &cloud);

void mpcl_compute_normals(pcl::PointCloud<pcl::PointXYZ> &cloud,
                          int ksearch,
                          pcl::PointCloud<pcl::Normal> &out);

#endif
