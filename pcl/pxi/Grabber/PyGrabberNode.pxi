
import grabber_callback

cdef class PyGrabberNode:
    double d_prop

    # def __cinit__(self):
    #     self.thisptr = new PyLibCallBack(callback, <void*>method)

    # def __dealloc__(self):
    #    if self.thisptr:
    #        del self.thisptr

    def Test(self):
        print('PyGrabberNode - Test')
        d_prop = 10.0


