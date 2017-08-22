#include "grabber_callback.h"

namespace grabber_callback {

PyLibCallBack::PyLibCallBack() 
{
    is_cy_call = true;
};

PyLibCallBack::PyLibCallBack(Method method, void *user_data)
{
	is_cy_call = true;
	_method = method;
    _user_data = user_data;
};

PyLibCallBack::~PyLibCallBack()
{
};

double PyLibCallBack::cy_execute (void *parameter)
{
	return this->_method(parameter, _user_data);
};

} // namespace grabber_callback