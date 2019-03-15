# Removing outliers using a Conditional or RadiusOutlier removal
# http://pointclouds.org/documentation/tutorials/remove_outliers.php#remove-outliers

import pcl
import numpy as np
import random

import argparse

parser = argparse.ArgumentParser(description='PointCloudLibrary example: Remove outliers')
parser.add_argument('--Removal', '-r', choices=('Radius', 'Condition'), default='',
                    help='RadiusOutlier/Condition Removal')
args = parser.parse_args()


# int main (int argc, char** argv)
# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
# pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
cloud = pcl.PointCloud()
cloud_filtered = pcl.PointCloud()

# // Fill in the cloud data
# cloud->width  = 5;
# cloud->height = 1;
# cloud->points.resize (cloud->width * cloud->height);
# 
# for (size_t i = 0; i < cloud->points.size (); ++i)
# {
# cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
# cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
# cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
# }
# 
# x,y,z
points = np.zeros((5, 3), dtype=np.float32)
RAND_MAX = 1024.0
for i in range(0, 5):
    points[i][0] = 1024 * random.random () / RAND_MAX
    points[i][1] = 1024 * random.random () / RAND_MAX
    points[i][2] = 1024 * random.random () / RAND_MAX

cloud.from_array(points)


# if (strcmp(argv[1], "-r") == 0)
# {
# pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    # // build the filter
    # outrem.setInputCloud(cloud);
    # outrem.setRadiusSearch(0.8);
    # outrem.setMinNeighborsInRadius (2);
    # // apply filter
    # outrem.filter (*cloud_filtered);
# }
# else if (strcmp(argv[1], "-c") == 0)
# {
    #   // build the condition
    #   pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond (new pcl::ConditionAnd<pcl::PointXYZ> ());
    #   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (
    #       new pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::GT, 0.0)));
    # 
    #   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (
    #       new pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::LT, 0.8)));
    # 
    #   // build the filter
    #   pcl::ConditionalRemoval<pcl::PointXYZ> condrem (range_cond);
    #   condrem.setInputCloud (cloud);
    #   condrem.setKeepOrganized(true);
    #   // apply filter
    #   condrem.filter (*cloud_filtered);
# }
# else
# {
#   std::cerr << "please specify command line arg '-r' or '-c'" << std::endl;
#   exit(0);
# }
if args.Removal == 'Radius':
    outrem = cloud.make_RadiusOutlierRemoval()
    outrem.set_radius_search(0.8)
    outrem.set_MinNeighborsInRadius(2)
    cloud_filtered = outrem.filter ()
elif args.Removal == 'Condition':
    range_cond = cloud.make_ConditionAnd()

    range_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.GT, 0.0)
    range_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.LT, 0.8)

    # build the filter
    condrem = cloud.make_ConditionalRemoval()
    condrem.set_Condition(range_cond)
    condrem.set_KeepOrganized(True)
    # apply filter
    cloud_filtered = condrem.filter ()
    
    # Test
    # cloud_filtered = cloud
else:
    print("please specify command line arg paramter 'Radius' or 'Condition'")


# std::cerr << "Cloud before filtering: " << std::endl;
# for (size_t i = 0; i < cloud->points.size (); ++i)
# std::cerr << "    " << cloud->points[i].x << " "
#                     << cloud->points[i].y << " "
#                     << cloud->points[i].z << std::endl;
# // display pointcloud after filtering
print ('Cloud before filtering: ')
for i in range(0, cloud.size):
    print ('x: '  + str(cloud[i][0]) + ', y : ' + str(cloud[i][1])  + ', z : ' + str(cloud[i][2]))

# std::cerr << "Cloud after filtering: " << std::endl;
# for (size_t i = 0; i < cloud_filtered->points.size (); ++i)
# std::cerr << "    " << cloud_filtered->points[i].x << " "
#                     << cloud_filtered->points[i].y << " "
#                     << cloud_filtered->points[i].z << std::endl;
print ('Cloud after filtering: ')
for i in range(0, cloud_filtered.size):
    print ('x: '  + str(cloud_filtered[i][0]) + ', y : ' + str(cloud_filtered[i][1])  + ', z : ' + str(cloud_filtered[i][2]))


