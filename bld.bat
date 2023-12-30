pip install -r requirements.txt
rem pip install -e .
python setup.py build_ext -i
python setup.py install
rem python setup.py bdist_wheel
nosetests -A "not pcl_ver_0_4"
