namespace {
    // Workaround for a Cython bug in operator[] with templated types and
    // references. Let's hope the compiler optimizes these functions away.
    template <typename T>
    T *getptr(pcl::PointCloud<T> *pc, size_t i)
    {
        return &(*pc)[i];
    }

    template <typename T>
    T *getptr_at(pcl::PointCloud<T> *pc, size_t i)
    {
        return &(pc->at(i));
    }

    template <typename T>
    T *getptr_at(pcl::PointCloud<T> *pc, int i, int j)
    {
        return &(pc->at(i, j));
    }
    
    //this shouldn't be necessary, but cython wont compile without it.
    template <typename T>
    T *getptrN(pcl::PointCloud<T> *pc, size_t i)
    {
        return &(*pc)[i];
    }

    template <typename T>
    T *getptrN_at(pcl::PointCloud<T> *pc, size_t i)
    {
        return &(pc->at(i));
    }

    template <typename T>
    T *getptrN_at(pcl::PointCloud<T> *pc, int i, int j)
    {
        return &(pc->at(i, j));
    }
}
