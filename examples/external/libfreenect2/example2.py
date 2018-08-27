# coding: utf-8
# before install libfreenect2 & pylibfreenect2
# https://qiita.com/leonarudo00/items/9056c8e0a9d08a24a0f7
import pcl
import pcl.pcl_visualization
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel


def main():
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()

    # Create and set logger
    logger = createConsoleLogger(LoggerLevel.Debug)
    setGlobalLogger(logger)

    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No device connected!")
        sys.exit(1)

    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline=pipeline)

    listener = SyncMultiFrameListener(
        FrameType.Color | FrameType.Ir | FrameType.Depth)

    # Register listeners
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    device.start()

    # NOTE: must be called after device.start()
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())

    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)

    # Optinal parameters for registration
    # set True if you need
    need_bigdepth = False
    need_color_depth_map = False

    bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
    color_depth_map = np.zeros((424, 512),  np.int32).ravel() \
        if need_color_depth_map else None

    point = pcl.PointCloud()
    viewer = pcl.pcl_visualization.PCLVisualizering()

    v = True
    while v:
        v = not(viewer.WasStopped())
        viewer.spinOnce()

        frames = listener.waitForNewFrame()

        color = frames["color"]
        ir = frames["ir"]
        depth = frames["depth"]

        registration.apply(color, depth, undistorted, registered,
                           bigdepth=bigdepth,
                           color_depth_map=color_depth_map)

        points = np.zeros((512*424, 3), dtype=np.float32)
        for r in range(0, 512):
            for c in range(0, 424):
                point = registration.getPointXYZ(undistorted, registered, r, c)
                # point = registration.getPointXYZRGB(undistorted, registered, r, c)
                points[r * 424 + c][0] = point[0]
                points[r * 424 + c][1] = point[1]
                points[r * 424 + c][2] = point[2]
                # point B, G, R,
                # points[r * 424 + c][3] = point[5]
                # points[r * 424 + c][4] = point[4]
                # points[r * 424 + c][5] = point[3]

        undistorted_arrray = undistorted.asarray(dtype=np.float32, ndim=2)
        # registered_array = registered.asarray(dtype=np.uint8)
        point = pcl.PointCloud(undistorted_arrray)
        # visual.ShowColorCloud(cloud)

        # cloud_normals = estimateNormal(cloud)

        # Update estimation
        # viewer.removePointCloud(b'normals')
        # viewer.RemovePointCloud(b'normals', 0)
        # viewer.addPointCloudNormals<PointType, pcl::Normal>( cloud, cloud_normals, 100, 0.05, "normals")
        # viewer.AddPointCloudNormals(cloud, cloud_normals, 100, 0.05, b'normals')

        listener.release(frames)

    device.stop()
    device.close()

    sys.exit(0)

    def estimateNormal(cloud):
        ne = pcl.IntegralImageNormalEstimation()

        ne.setNormalEstimationMethod(pcl.pcl_AVERAGE_DEPTH_CHANGE)
        ne.setMaxDepthChangeFactor(0.01)
        ne.setNormalSmoothingSize(5.0)
        ne.setInputCloud(cloud)
        cloud_normals = ne.compute()
        return cloud_normals


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
