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
    T *getptr_at2(pcl::PointCloud<T> *pc, int i, int j)
    {
        return &(pc->at(i, j));
    }
}
