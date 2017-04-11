#ifndef __GUARD
#define __GUARD

class cpp_backend
{
public:
    // This is wrapper of Python fuction.
    typedef double (*Method)(void *param, void *callback_func);

    // Constructor
    cpp_backend(Method, void *user_data);
    // Destructor
    virtual ~cpp_backend();

    double cy_execute(void *parameter);

private:
    method_type method_;
    void *python_callback_pointer_;
};

#endif