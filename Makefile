pcl.so: pcl.pyx setup.py pcl_defs.pxd minipcl.cpp
	python setup.py build_ext --inplace

test: pcl.so tests/test.py
	nosetests -s

clean:
	rm -rf build
	rm -f pcl.cpp pcl.so
