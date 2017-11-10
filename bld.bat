python setup.py build_ext -i
python setup.py install
nosetests -A "not pcl_ver_0_4"
