# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import pcl

cloud = pcl.load('./examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')

# pcl::CropBox<PointXYZI> clipper;
# clipper.setInputCloud(cloud);
clipper = cloud.make_cropbox()

# pcl::PCDWriter writer;
# pcl::PointCloud<PointXYZI>::Ptr outcloud;
outcloud = pcl.PointCloud()

# clipper.setTranslation(Eigen::Vector3f(pose->tx, pose->ty, pose->tz));
# clipper.setRotation(Eigen::Vector3f(pose->rx, pose->ry, pose->rz));
# clipper.setMin(-Eigen::Vector4f(tracklet->l/2, tracklet->w/2, 0, 0));
# clipper.setMax(Eigen::Vector4f(tracklet->l/2, tracklet->w/2, tracklet->h, 0));
# clipper.filter(*outcloud);
tx = 0
ty = 0
tz = 0
clipper.set_Translation(tx, ty, tz)
rx = 0
ry = 0
rz = 0
clipper.set_Rotation(rx, ry, rz)
minx = -1.5
miny = -1.5
minz = 2
mins = 0
maxx = 3.5
maxy = 3.5
maxz = 3
maxs = 0
clipper.set_MinMax(minx, miny, minz, mins, maxx, maxy, maxz, maxs)
clipper.Filtering(outcloud)

pcl.save(outcloud, "test.pcd")

# stringstream outfilename;
# outfilename << outfile << tracklet->objectType << i << ".pcd";
# if(!outcloud->empty()){
# 	cout << "Found "<<outcloud->size() << " points, writing to " << outfilename.str() << endl;
# 	writer.write<PointXYZI> (outfilename.str(), *outcloud, false);
# }else{
# 	cerr << "Couldn't find points for tracklet" << tracklet->objectType << i << endl;
# }
