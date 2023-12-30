#ifndef GRABBER_CALLBACK_H_
#define GRABBER_CALLBACK_H_

namespace grabber_callback
{
    class PyLibCallBack
    {
    public:
        // This is wrapper of Python fuction.
        typedef double (*Method)(void *param, void *user_data);

        // Constructor
    	PyLibCallBack();
        PyLibCallBack(Method method, void *user_data);
        // Destructor
        virtual ~PyLibCallBack();

        double cy_execute(void *parameter);

        bool IsCythonCall()
        {
            return is_cy_call;
        }

    protected:
        bool is_cy_call;

    private:
        Method _method;
        void *_user_data;
    };
}
#endif