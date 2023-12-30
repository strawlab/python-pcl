#ifndef __BIND_CPP__
#define __BIND_CPP__

#include <iostream>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/signals2.hpp>

// void some_callback(void* some_ptr);
boost::signals2::connection register_callback(boost::function<void (void*)> callback);

#endif
