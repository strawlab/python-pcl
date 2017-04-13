#include "grabber_callback.h"

namespace grabber_callback {

PyLibCallBack::PyLibCallBack() 
{
    is_cy_call = true;
};

PyLibCallBack::PyLibCallBack(method_type method, void *user_data)
{
	is_cy_call = true;
	_method = method;
    _user_data = user_data;
};

PyLibCallBack::~PyLibCallBack()
{
};

// double PyLibCallBack::callback_python(void *parameter)
// {
//     return this->_method(parameter, python_callback_pointer_);
// }

double PyLibCallBack::cy_execute (void *parameter)
{
	return this->_method(parameter, _user_data);
};

// void PyLibCallBack::cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)   
// {   
// 	if (!viewer.wasStopped())
//  		viewer.showCloud (cloud);
// }   

} // namespace grabber_callback