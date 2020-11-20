import cv2
import numpy as np
import pcl


# base code
# https://qiita.com/SatoshiGachiFujimoto/items/eb3891116d4f49cd342d
# https://github.com/SatoshiRobatoFujimoto/PointCloudViz
# https://stackoverflow.com/questions/21849512/how-to-align-rgb-and-depth-image-of-kinect-in-opencv/21851642
# https://stackoverflow.com/questions/29270544/how-to-display-a-3d-image-when-we-have-depth-and-rgb-mats-in-opencv-captured-f
# https://github.com/IntelRealSense/librealsense/issues/2090
# https://github.com/daavoo/pyntcloud
def cvtDepth2Cloud(depth, cameraMatrix):
    inv_fx = 1.0 / cameraMatrix[0, 0]
    inv_fy = 1.0 / cameraMatrix[1, 1]
    ox = cameraMatrix[0, 2]
    oy = cameraMatrix[1, 2]
    # print(inv_fx)
    # print(inv_fy)
    # print(ox)
    # print(oy)

    # print(depth.size)
    rows, cols = depth.shape
    # print(cols)
    # print(rows)
    cloud = np.zeros((depth.size, 3), dtype=np.float32)
    # print(cloud)
    for y in range(rows):
        for x in range(cols):
            # print(x)
            # print(y)
            x1 = float(x)
            y1 = float(y)
            # print(x1)
            # print(y1)
            dist = depth[y][x]
            # print(dist)
            # print(cloud[y * cols + x][0])
            cloud[y * cols + x][0] = np.float32((x1 - ox) * dist * inv_fx)
            cloud[y * cols + x][1] = np.float32((y1 - oy) * dist * inv_fy)
            cloud[y * cols + x][2] = np.float32(dist)

    # cloud = []
    # for v in range(height):
    #     for u in range(width):
    #         offset = (v * width + u) * 2
    #         depth = ord(array[offset]) + ord(array[offset+1]) * 256
    #         x = (u - CX) * depth * UNIT_SCALING / FX
    #         y = (v - CY) * depth * UNIT_SCALING / FY
    #         z = depth * UNIT_SCALING
    #         cloud.append((x, y, z))

    return cloud


def cvtDepthColor2Cloud(depth, color, cameraMatrix):
    inv_fx = 1.0 / cameraMatrix[0, 0]
    inv_fy = 1.0 / cameraMatrix[1, 1]
    ox = cameraMatrix[0, 2]
    oy = cameraMatrix[1, 2]
    # print(inv_fx)
    # print(inv_fy)
    # print(ox)
    # print(oy)

    # print(depth.size)
    rows, cols = depth.shape
    # print(cols)
    # print(rows)
    cloud = np.zeros((depth.size, 4), dtype=np.float32)
    # print(cloud)
    for y in range(rows):
        for x in range(cols):
            # print(x)
            # print(y)
            x1 = float(x)
            y1 = float(y)
            # print(x1)
            # print(y1)
            dist = -depth[y][x]
            # print(dist)
            # print(cloud[y * cols + x][0])
            cloud[y * cols + x][0] = -np.float32((x1 - ox) * dist * inv_fx)
            cloud[y * cols + x][1] = np.float32((y1 - oy) * dist * inv_fy)
            cloud[y * cols + x][2] = np.float32(dist)
            red = color[y][x][2]
            green = color[y][x][1]
            blue = color[y][x][0]
            rgb = np.left_shift(red, 16) + np.left_shift(green,
                                                         8) + np.left_shift(blue, 0)
            cloud[y * cols + x][3] = rgb

    return cloud


def main():
    # | fx  0   cx |
    # | 0   fy  cy |
    # | 0   0   1  |
    # vals = np.array(
    #     [525., 0.  , 319.5,
    #      0.  , 525., 239.5,
    #      0.  , 0.  , 1.])
    # cameraMatrix = vals.reshape((3, 3))
    cameraMatrix = np.array(
        [[525., 0., 319.5],
         [0., 525., 239.5],
         [0., 0., 1.]])

    color0 = cv2.imread('rgb/0.png')
    # 16 bit Image
    # https://github.com/eiichiromomma/CVMLAB/wiki/OpenCV-16bitImage
    # depth0 = cv2.imread('depth/0.png', -1)
    depth0 = cv2.imread('depth/0.png', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    print("Color: ", color0.dtype)
    print("Depth: ", depth0.dtype)

    # colorImage1 = cv2.imread('rgb/1.png')
    # depth1 = cv2.imread('depth/1.png', -1)

    # if (color0.empty() || depth0.empty() || colorImage1.empty() || depth1.empty()):
    #     cout << "Data (rgb or depth images) is empty.";
    #     return -1;

    # gray0 = cv2.cvtColor(color0, cv2.COLOR_BGR2GRAY)
    # grayImage1 = cv2.cvtColor(colorImage1, cv2.COLOR_BGR2GRAY)
    # depthFlt0 = depth0.convertTo(cv2.CV_32FC1, 1. / 1000.0)
    # depthFlt1 = depth1.convertTo(cv2.CV_32FC1, 1. / 1000.0)
    depthFlt0 = np.float32(depth0) / 1000.0
    # depthFlt0 = depth0 / 1000.0
    # depthFlt1 = np.float32(depth1) / 1000.0

    import pcl
    # points0 = cvtDepth2Cloud(depthFlt0, cameraMatrix)
    # cloud0 = pcl.PointCloud()
    points0 = cvtDepthColor2Cloud(depthFlt0, color0, cameraMatrix)
    cloud0 = pcl.PointCloud_PointXYZRGBA()
    cloud0.from_array(points0)
    print(cloud0)

    # points1 = cvtDepth2Cloud(depthFlt1, cameraMatrix)
    # cloud1 = pcl.PointCloud()
    # cloud1.from_array(points1)

    # wait
    try:
        import pcl.pcl_visualization
        visual = pcl.pcl_visualization.CloudViewing()
        # xyz only
        # visual.ShowMonochromeCloud(cloud0)
        # visual.ShowMonochromeCloud(cloud1)
        # color(rgba)
        visual.ShowColorACloud(cloud0)
        v = True
        while v:
            v = not(visual.WasStopped())
    except:
        pass


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
