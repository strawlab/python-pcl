
## Table of Contents
  - [Dependencies](#dependencies)
    - [ROS](#ros)
    - [ros-utils](#ros-utils)
  - [Installation](#installation)
  - [ROS Package](#ros-package)
    - [ROS Compilation](#ros-compilation)
    - [Tutorials](#tutorials)
  - [CMAKE flags](#cmake-flags)
  - [Standalone](#standalone)
    - [Compilation](#compilation)
    - [Run](#run)

# License

# Dependencies

## ROS

We have tested ???? in Ubuntu 16.04 with ROS Kinetic.
To install ROS (Kinetic) use the following command:
`sudo apt-get install ros-kinetic-desktop`

We have tested ???? in Ubuntu 18.04 with ROS Melodic.
To install ROS (Melodic) use the following command:
`sudo apt-get install ros-Melodic-desktop`

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

