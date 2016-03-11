#ifndef _MINIPCL_H_
#define _MINIPCL_H_

#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/Octree/octree_pointcloud.h>
//
void mpcl_compute_normals(pcl::PointCloud<pcl::PointXYZ> cloud,
                          int ksearch,
                          double searchRadius,
                          pcl::PointCloud<pcl::Normal> &out);

void mpcl_compute_normals_PointXYZI(pcl::PointCloud<pcl::PointXYZI> cloud,
                          int ksearch,
                          double searchRadius,
                          pcl::PointCloud<pcl::Normal> &out);

void mpcl_compute_normals_PointXYZRGB(pcl::PointCloud<pcl::PointXYZRGB> cloud,
                          int ksearch,
                          double searchRadius,
                          pcl::PointCloud<pcl::Normal> &out);

void mpcl_compute_normals_PointXYZRGBA(pcl::PointCloud<pcl::PointXYZRGBA> cloud,
                          int ksearch,
                          double searchRadius,
                          pcl::PointCloud<pcl::Normal> &out);

// 
void mpcl_sacnormal_set_axis(pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> &sac,
                             double ax, double ay, double az);

void mpcl_sacnormal_set_axis_PointXYZI(pcl::SACSegmentationFromNormals<pcl::PointXYZI, pcl::Normal> &sac,
                             double ax, double ay, double az);

void mpcl_sacnormal_set_axis_PointXYZRGB(pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> &sac,
                             double ax, double ay, double az);

void mpcl_sacnormal_set_axis_PointXYZRGBA(pcl::SACSegmentationFromNormals<pcl::PointXYZRGBA, pcl::Normal> &sac,
                             double ax, double ay, double az);


//
void mpcl_extract(pcl::PointCloud<pcl::PointXYZ>::Ptr &incloud,
                  pcl::PointCloud<pcl::PointXYZ> *outcloud,
                  pcl::PointIndices *indices,
                  bool negative);

void mpcl_extract_PointXYZI(pcl::PointCloud<pcl::PointXYZI>::Ptr &incloud,
                  pcl::PointCloud<pcl::PointXYZI> *outcloud,
                  pcl::PointIndices *indices,
                  bool negative);

void mpcl_extract_PointXYZRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &incloud,
                  pcl::PointCloud<pcl::PointXYZRGB> *outcloud,
                  pcl::PointIndices *indices,
                  bool negative);

void mpcl_extract_PointXYZRGBA(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &incloud,
                  pcl::PointCloud<pcl::PointXYZRGBA> *outcloud,
                  pcl::PointIndices *indices,
                  bool negative);

// Octree
void mpcl_deleteVoxelAtPoint(pcl::PointXYZ incloud);
void mpcl_deleteVoxelAtPoint(pcl::PointXYZI incloud);
void mpcl_deleteVoxelAtPoint(pcl::PointXYZRGB incloud);
void mpcl_deleteVoxelAtPoint(pcl::PointXYZRGBA incloud);

#endif
