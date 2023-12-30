# docker build -t ubuntu1804py36
FROM ubuntu:18.04

RUN apt-get update && \
        apt-get install -y software-properties-common vim && \
        add-apt-repository ppa:jonathonf/python-3.6

RUN apt-get update -y

RUN apt-get install cmake -y && \
    apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv && \
    apt-get install -y git && \
    apt-get install openni2-utils -y && \
    apt-get install libpcl-dev -y

# fork module
RUN git clone -b rc_patches4 https://github.com/Sirokujira/python-pcl.git
# main
# RUN git clone -b master https://github.com/strawlab/python-pcl.git

WORKDIR /python-pcl

# update pip
RUN python3.6 -m pip install pip --upgrade && \
    python3.6 -m pip install wheel

RUN pip install -r requirements.txt && \
    python3.6 setup.py build_ext -i && \
    python3.6 setup.py install

