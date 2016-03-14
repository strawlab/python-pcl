#include "grabber_callback.hpp"

cpp_backend::cpp_backend(method_type method, void *callback_func)
    : method_(method), python_callback_pointer_(callback_func)
{}

cpp_backend::~cpp_backend()
{}

double cpp_backend::callback_python(void *parameter)
{
    return this->method_(parameter, python_callback_pointer_);
}

void cpp_backend::cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)   
{   
if (!viewer.wasStopped())   
 viewer.showCloud (cloud);   
}   
