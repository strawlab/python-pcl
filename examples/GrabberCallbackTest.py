import pcl.pcl_grabber

def func(obj):
    print(obj)
    obj.Test()     # Call to a specific method from class 'PyGrabberNode'
    return obj.d_prop

n = pcl.pcl_grabber.PyGrabberNode()    # Custom class of my own
cb = pcl.pcl_grabber.PyGrabberCallback(func)
print(cb.execute(n))