#ifndef __GUARD
#define __GUARD

class cpp_backend
{
public:
    // This is wrapper of Python fuction.
    typedef double (*method_type)(void *param, void *callback_func);

    // Constructor
    cpp_backend(method_type, void *user_data);
    // Destructor
    virtual ~cpp_backend();

    double callback_python(void *parameter);

private:
    method_type method_;
    void *python_callback_pointer_;
};

#endif