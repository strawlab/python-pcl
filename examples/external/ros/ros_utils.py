import roslib
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import numpy
import pcl


def on_new_point_cloud(data):
    pc = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))
    pc_list = []
    for p in pc:
        pc_list.append([p[0], p[1], p[2]])

    p = pcl.PointCloud()
    p.from_list(pc_list)
    seg = p.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    indices, model = seg.segment()


# reference : http://robonchu.hatenablog.com/entry/2017/09/20/234640
def main():
    rospy.init_node('listener', anonymous=True)
    # rospy.Subscriber("/kinect2/sd/points", PointCloud2, on_new_point_cloud)
    rospy.Subscriber("/kinect2/hd/points", PointCloud2, on_new_point_cloud)
    rospy.spin()


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
