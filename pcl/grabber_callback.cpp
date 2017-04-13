#include "grabber_callback.h"

PyLibCallBack::PyLibCallBack(method_type method, void *callback_func)
{
	is_cy_call = true;
}

PyLibCallBack::~PyLibCallBack()
{
}

double PyLibCallBack::callback_python(void *parameter)
{
    return this->method_(parameter, python_callback_pointer_);
}

double PyLibCallBack::cy_execute (void *parameter)
{
	return _method(parameter, _user_data);
}

// void PyLibCallBack::cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)   
// {   
// 	if (!viewer.wasStopped())
//  		viewer.showCloud (cloud);
// }   
