ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel

# install basics
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections \
 && apt-get update --fix-missing -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake g++ pkg-config \
 && apt-get install --reinstall python3-apt python3-pip -y


# install python-pcl dependencies
RUN apt-get install -y pkg-config build-essential linux-libc-dev libeigen3-dev \
  && apt-get install -y libpcl-dev libvtk6-dev \
  && python3 -m pip install --upgrade pip wheel numpy cython pkgconfig twine nose
