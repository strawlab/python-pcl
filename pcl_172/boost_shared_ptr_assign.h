#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace {
    // Workaround for lack of operator= support in Cython.
    template <typename T>
    void sp_assign(boost::shared_ptr<T> &p, T *v)
    {
        p = boost::shared_ptr<T>(v);
    }
}
