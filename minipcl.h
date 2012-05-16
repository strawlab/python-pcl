#ifndef _MINIPCL_H_
#define _MINIPCL_H_

#include <pcl/point_types.h>

void mpcl_compute_normals(pcl::PointCloud<pcl::PointXYZ> &cloud,
                          int ksearch,
                          pcl::PointCloud<pcl::Normal> &out);

#endif
