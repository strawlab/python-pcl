#ifndef _MINIPCL_H_
#define _MINIPCL_H_

#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/octree/octree_pointcloud.h>

#include <vector>
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

// Octree(OctreePointCloud)
// void mpcl_deleteVoxelAtPoint(pcl::octree::OctreePointCloud<pcl::PointXYZ>& inOctree, pcl::PointXYZ incloud);
// void mpcl_deleteVoxelAtPoint_PointXYZI(pcl::octree::OctreePointCloud<pcl::PointXYZI>& inOctree, pcl::PointXYZI incloud);
// void mpcl_deleteVoxelAtPoint_PointXYZRGB(pcl::octree::OctreePointCloud<pcl::PointXYZRGB>& inOctree, pcl::PointXYZRGB incloud);
// void mpcl_deleteVoxelAtPoint_PointXYZRGBA(pcl::octree::OctreePointCloud<pcl::PointXYZRGBA>& inOctree, pcl::PointXYZRGBA incloud);

// int mpcl_getOccupiedVoxelCenters(pcl::octree::OctreePointCloud<pcl::PointXYZ>& inOctree, std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > alignPoint);
// int mpcl_getOccupiedVoxelCenters_PointXYZI(pcl::octree::OctreePointCloud<pcl::PointXYZI>& inOctree, std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI> > alignPoint);
// int mpcl_getOccupiedVoxelCenters_PointXYZRGB(pcl::octree::OctreePointCloud<pcl::PointXYZRGB>& inOctree, std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> > alignPoint);
// int mpcl_getOccupiedVoxelCenters_PointXYZRGBA(pcl::octree::OctreePointCloud<pcl::PointXYZRGBA>& inOctree, std::vector<pcl::PointXYZRGBA, Eigen::aligned_allocator<pcl::PointXYZRGBA> > alignPoint);

#endif
