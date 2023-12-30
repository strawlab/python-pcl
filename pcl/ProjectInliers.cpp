#include <pcl/point_types.h>

#include "ProjectInliers.h"

// set ksearch and radius to < 0 to disable 
void mpcl_ProjectInliers_setModelCoefficients(pcl::ProjectInliers<pcl::PointXYZ> &inliers)
{
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    coefficients->values.resize (4);
    coefficients->values[0] = coefficients->values[1] = 0;
    coefficients->values[2] = 1.0;
    coefficients->values[3] = 0;

    inliers.setModelCoefficients (coefficients);
}