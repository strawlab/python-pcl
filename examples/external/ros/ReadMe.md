
S-PTAM is a Stereo SLAM system able to compute the camera trajectory in real-time. It heavily exploits the parallel nature of the SLAM problem, separating the time-constrained pose estimation from less pressing matters such as map building and refinement tasks. On the other hand, the stereo setting allows to reconstruct a metric 3D map for each frame of stereo images, improving the accuracy of the mapping process with respect to monocular SLAM and avoiding the well-known bootstrapping problem. Also, the real scale of the environment is an essential feature for robots which have to interact with their surrounding workspace.

## Table of Contents
  - [License](#license)
  - [Dependencies](#dependencies)
    - [ROS](#ros)
    - [ros-utils](#ros-utils)
  - [Installation](#installation)
  - [ROS Package](#ros-package)
    - [ROS Compilation](#ros-compilation)
    - [Tutorials](#tutorials)
      - [KITTI dataset](#kitti-dataset)
      - [EuRoc MAV dataset](#euroc-mav-dataset)
      - [MIT Stata Center dataset](#mit-stata-center-dataset)
      - [Indoor Level 7 S-Block dataset](#indoor-level-7-s-block-dataset)
    - [Node Information](#node-information)
      - [Subscribed Topics](#subscribed-topics)
      - [Published Topics](#published-topics)
      - [ROS Parameters](#ros-parameters)
      - [Involved Coordinates Systems](#involved-coordinates-systems)
  - [CMAKE flags](#cmake-flags)
  - [S-PTAM Parameters](#s-ptam-parameters)
  - [Standalone](#standalone)
    - [Compilation](#compilation)
    - [Run](#run)

# License

# Dependencies

## ROS

We have tested ???? in Ubuntu 16.04 with ROS Kinetic.
To install ROS (Kinetic) use the following command:

`sudo apt-get install ros-kinetic-desktop`

## SuiteSparse

Suitespare is a dependency, so it needs to be installed

`sudo apt-get install libsuitesparse-dev`

## ros-utils

Install our [ros-utils](https://github.com/lrse/ros-utils) library from the source code provided in  

`git clone git@github.com:lrse/ros-utils.git`

# ROS Package

## ROS Compilation

`catkin_make --pkg sptam -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSINGLE_THREAD=OFF -DSHOW_TRACKED_FRAMES=ON -DSHOW_PROFILING=ON -DPARALLELIZE=ON`

To activate Loop Closing capabilities (requires DBoW2 and OpenGV dependencies).

`catkin_make --pkg sptam -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_LOOPCLOSURE=ON -DSINGLE_THREAD=OFF -DSHOW_TRACKED_FRAMES=ON -DSHOW_PROFILING=ON -DPARALLELIZE=ON`

For more information about compilation flags see [CMAKE flags](#cmakeFlags) section.


1-b. Source Code
```cmd
git clone https://github.com/r9y9/pylibfreenect2.git
python setup.py install
```

Reference :
http://wiki.ros.org/rospy/Tutorials

How to use a PCL tutorial in ROS
http://wiki.ros.org/pcl/Overview
http://wiki.ros.org/perception_pcl/Tutorials

Japanese : 
http://ros-recognition-tutorial.readthedocs.io/ja/latest/pcl/


2. execute example

* before Kinect2 device connected

python ros_utils.py

