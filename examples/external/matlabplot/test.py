import pcl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main():
    pt = pcl.load('bunny.pcd')

    shape = pt.to_array().transpose()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = shape[0]
    y = shape[1]
    z = shape[2]

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()