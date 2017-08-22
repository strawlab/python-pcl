#ifndef _ProjectInliers_H_
#define _ProjectInliers_H_

#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

//
void mpcl_ProjectInliers_setModelCoefficients(pcl::ProjectInliers<pcl::PointXYZ> &inliers);

#endif
