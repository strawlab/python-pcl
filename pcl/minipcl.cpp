#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/octree/octree_pointcloud.h>

#include <pcl/features/vfh.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Dense>

#include <pcl/features/integral_image_normal.h>

#include "minipcl.h"

// set ksearch and radius to < 0 to disable 
void mpcl_compute_normals(const pcl::PointCloud<pcl::PointXYZ>& cloud,
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

void mpcl_compute_normals_PointXYZI(const pcl::PointCloud<pcl::PointXYZI>& cloud,
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

void mpcl_compute_normals_PointXYZRGB(const pcl::PointCloud<pcl::PointXYZRGB>& cloud,
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


void mpcl_compute_normals_PointXYZRGBA(const pcl::PointCloud<pcl::PointXYZRGBA>& cloud,
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

// EuclideanClusterExtraction
// Octree
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

void mpcl_extract_VFH(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // http://virtuemarket-lab.blogspot.jp/2015/03/viewpoint-feature-histogram.html
    // pcl::PointCloud<pcl::VFHSignature308>::Ptr Extract_VFH(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal> ());
    pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
    
    // cloud_normals = surface_normals(cloud);
    vfh.setInputCloud (cloud);
    vfh.setInputNormals (cloud_normals);
    
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    vfh.setSearchMethod (tree);
    
    vfh.compute (*vfhs);
    // return vfhs;
}

/*
// pcl1.6 
#include <pcl/keypoints/harris_keypoint3D.h>
// use 1.7
#include <pcl/keypoints/harris_3d.h>

// HarrisKeypoint3D
// NG 
// outcloud set pcl::PointXYZI
void mpcl_extract_HarrisKeypoint3D(pcl::PointCloud<pcl::PointXYZ>::Ptr &incloud,
                                   pcl::PointCloud<pcl::PointXYZ> *outcloud)
{
    // pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> detector;
    pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> detector;

    detector.setInputCloud(incloud);

    detector.setNonMaxSupression (true);
    detector.setRadius (0.01);
    // detector.setRadiusSearch (100);
    // detector.setIndices(indicesptr);
    
    // NG
    // detector.compute(*outcloud);

    // OK
    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());
    detector.compute(*keypoints);
}

// HarrisKeypoint3D
void mpcl_extract_HarrisKeypoint3D(pcl::PointCloud<pcl::PointXYZ>::Ptr &incloud,
                                   pcl::PointCloud<pcl::PointXYZI> *outcloud)
{
    pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> detector;

    detector.setInputCloud(incloud);

    detector.setNonMaxSupression (true);
    detector.setRadius (0.01);
    //detector.setRadiusSearch (100);

    // detector.setIndices(indicesptr);
    // detector.compute(*outcloud);
    detector.compute(*outcloud);
}
*/

// features
// integral_image_normal.h
void mpcl_features_NormalEstimationMethod_AVERAGE_3D_GRADIENT(pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne)
{
    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
}

void mpcl_features_NormalEstimationMethod_COVARIANCE_MATRIX(pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne)
{
    ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
}

void mpcl_features_NormalEstimationMethod_AVERAGE_DEPTH_CHANGE(pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne)
{
    ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE);
}

void mpcl_features_NormalEstimationMethod_SIMPLE_3D_GRADIENT(pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne)
{
    ne.setNormalEstimationMethod (ne.SIMPLE_3D_GRADIENT);
}

void mpcl_features_NormalEstimationMethod_compute(pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne, pcl::PointCloud<pcl::Normal> &out)
{
    // NG : out Variant Function end error
    printf("compute start.\n");
    ne.compute (out);
    // pcl 1.7.2 error
    // printf("out = %p.\n", out);
    printf("compute end.\n");
}
