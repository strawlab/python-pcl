import rospy
import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
def on_new_point_cloud(data):
    pc = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))
    pc_list = []
    for p in pc:
        pc_list.append( [p[0],p[1],p[2]] )

    p = pcl.PointCloud()
    p.from_list(pc_list)
    seg = p.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    indices, model = seg.segment()

rospy.init_node('listener', anonymous=True)
rospy.Subscriber("/kinect2/hd/points", PointCloud2, on_new_point_cloud)
rospy.spin()