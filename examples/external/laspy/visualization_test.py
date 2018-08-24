import numpy as np
import random
import scipy as sc
from laspy import file
from pcl import pcl_visualization


def main():
    import pcl
    # laspy librairy, read las file
    f = file.File('28XXX10000075-18.las',mode='r')
    # Store pointcloud in array
    ptcloud = np.vstack((f.x, f.y, f.z)).transpose()
    f.close()

    # cloud = pcl.load('./examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')
    # ptcloud = cloud.to_list()

    # Centred the data
    ptcloud_centred = ptcloud - np.mean(ptcloud,0)

    # Simulate an intensity information between 0 and 1
    ptcloud_centred = sc.append(ptcloud_centred, np.zeros((ptcloud.shape[0], 1)), axis=1)
    for i in range(ptcloud_centred.shape[0] - 1):
        ptcloud_centred[i, 3] = random.random()

    p = pcl.PointCloud_PointXYZI()
    p.from_array(np.array(ptcloud_centred, dtype=np.float32))

    ## Visualization
    visual = pcl_visualization.CloudViewing()
    visual.ShowGrayCloud(p, b'cloud')

    v = True
    while v:
        v=not(visual.WasStopped())


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()