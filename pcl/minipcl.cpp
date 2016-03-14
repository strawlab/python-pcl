#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/octree/octree_pointcloud.h>

#include <Eigen/Dense>

#include "minipcl.h"

// set ksearch and radius to < 0 to disable 
void mpcl_compute_normals(pcl::PointCloud<pcl::PointXYZ> cloud,
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

void mpcl_compute_normals_PointXYZI(pcl::PointCloud<pcl::PointXYZI> cloud,
                          int ksearch,
                          double searchRadius,
                          pcl::PointCloud<pcl::Normal> &out)
{
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI> ());
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;

    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud.makeShared());
    if (ksearch >= 0)
        ne.setKSearch (ksearch);
    if (searchRadius >= 0.0)
        ne.setRadiusSearch (searchRadius);
    ne.compute (out);
}

void mpcl_compute_normals_PointXYZRGB(pcl::PointCloud<pcl::PointXYZRGB> cloud,
                          int ksearch,
                          double searchRadius,
                          pcl::PointCloud<pcl::Normal> &out)
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;

    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud.makeShared());
    if (ksearch >= 0)
        ne.setKSearch (ksearch);
    if (searchRadius >= 0.0)
        ne.setRadiusSearch (searchRadius);
    ne.compute (out);
}


void mpcl_compute_normals_PointXYZRGBA(pcl::PointCloud<pcl::PointXYZRGBA> cloud,
                          int ksearch,
                          double searchRadius,
                          pcl::PointCloud<pcl::Normal> &out)
{
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA> ());
    pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;

    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud.makeShared());
    if (ksearch >= 0)
        ne.setKSearch (ksearch);
    if (searchRadius >= 0.0)
        ne.setRadiusSearch (searchRadius);
    ne.compute (out);
}

// set ksearch and radius to < 0 to disable 
void mpcl_sacnormal_set_axis(pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> &sac,
                             double ax, double ay, double az)
{
    Eigen::Vector3f vect(ax,ay,az);
    sac.setAxis(vect);
}

void mpcl_sacnormal_set_axis_PointXYZI(pcl::SACSegmentationFromNormals<pcl::PointXYZI, pcl::Normal> &sac,
                             double ax, double ay, double az)
{
    Eigen::Vector3f vect(ax,ay,az);
    sac.setAxis(vect);
}

void mpcl_sacnormal_set_axis_PointXYZRGB(pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> &sac,
                             double ax, double ay, double az)
{
    Eigen::Vector3f vect(ax,ay,az);
    sac.setAxis(vect);
}

void mpcl_sacnormal_set_axis_PointXYZRGBA(pcl::SACSegmentationFromNormals<pcl::PointXYZRGBA, pcl::Normal> &sac,
                             double ax, double ay, double az)
{
    Eigen::Vector3f vect(ax,ay,az);
    sac.setAxis(vect);
}

// 
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

void mpcl_extract_PointXYZI(pcl::PointCloud<pcl::PointXYZI>::Ptr &incloud,
                   pcl::PointCloud<pcl::PointXYZI> *outcloud,
                   pcl::PointIndices *indices,
                   bool negative)
{
    pcl::PointIndices::Ptr indicesptr (indices);
    pcl::ExtractIndices<pcl::PointXYZI> ext;
    ext.setInputCloud(incloud);
    ext.setIndices(indicesptr);
    ext.setNegative(negative);
    ext.filter(*outcloud);
}

void mpcl_extract_PointXYZRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &incloud,
                   pcl::PointCloud<pcl::PointXYZRGB> *outcloud,
                   pcl::PointIndices *indices,
                   bool negative)
{
    pcl::PointIndices::Ptr indicesptr (indices);
    pcl::ExtractIndices<pcl::PointXYZRGB> ext;
    ext.setInputCloud(incloud);
    ext.setIndices(indicesptr);
    ext.setNegative(negative);
    ext.filter(*outcloud);
}

void mpcl_extract_PointXYZRGBA(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &incloud,
                   pcl::PointCloud<pcl::PointXYZRGBA> *outcloud,
                   pcl::PointIndices *indices,
                   bool negative)
{
    pcl::PointIndices::Ptr indicesptr (indices);
    pcl::ExtractIndices<pcl::PointXYZRGBA> ext;
    ext.setInputCloud(incloud);
    ext.setIndices(indicesptr);
    ext.setNegative(negative);
    ext.filter(*outcloud);
}

/// Octree
// void mpcl_deleteVoxelAtPoint(pcl::octree::OctreePointCloud<pcl::PointXYZ>& inOctree, pcl::PointXYZ incloud)
// {
//     inOctree.deleteVoxelAtPoint(incloud);
// }
// 
// void mpcl_deleteVoxelAtPoint(pcl::octree::OctreePointCloud<pcl::PointXYZI>& inOctree, pcl::PointXYZI incloud)
// {
//     inOctree.deleteVoxelAtPoint(incloud);
// }
// 
// void mpcl_deleteVoxelAtPoint(pcl::octree::OctreePointCloud<pcl::PointXYZRGB>& inOctree, pcl::PointXYZRGB incloud)
// {
//     inOctree.deleteVoxelAtPoint(incloud);
// }
// 
// void mpcl_deleteVoxelAtPoint(pcl::octree::OctreePointCloud<pcl::PointXYZRGBA>& inOctree, pcl::PointXYZRGBA incloud)
// {
//     inOctree.deleteVoxelAtPoint(incloud);
// }
// 
// 
// int mpcl_getOccupiedVoxelCenters(pcl::octree::OctreePointCloud<pcl::PointXYZ>& inOctree, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &incloud, vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > alignPoint)
// {
//     return inOctree.getOccupiedVoxelCenters(alignPoint);
// }
// 
// int mpcl_getOccupiedVoxelCenters(pcl::octree::OctreePointCloud<pcl::PointXYZI>& inOctree, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &incloud, vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI> > alignPoint)
// {
//     return inOctree.getOccupiedVoxelCenters(alignPoint);
// }
// 
// int mpcl_getOccupiedVoxelCenters(pcl::octree::OctreePointCloud<pcl::PointXYZRGB>& inOctree, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &incloud, vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> > alignPoint)
// {
//     return inOctree.getOccupiedVoxelCenters(alignPoint);
// }
// 
// int mpcl_getOccupiedVoxelCenters(pcl::octree::OctreePointCloud<pcl::PointXYZRGBA>& inOctree, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &incloud, vector<pcl::PointXYZRGBA, Eigen::aligned_allocator<pcl::PointXYZRGBA> > alignPoint)
// {
//     return inOctree.getOccupiedVoxelCenters(alignPoint);
// }
// 
