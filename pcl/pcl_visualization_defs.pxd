# -*- coding: utf-8 -*-
from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp
from pcl_range_image cimport RangeImage

# Eigen
cimport eigen as eigen3

# boost
from boost_shared_ptr cimport shared_ptr

###############################################################################
# Types
###############################################################################

### base class ###

# point_cloud_handlers.h(1.6.0)
# point_cloud_handlers.h -> point_cloud_color_handlers.h(1.7.2)
# template <typename PointT>
# class PointCloudColorHandler
cdef extern from "pcl/visualization/point_cloud_handlers.h" namespace "pcl::visualization" nogil:
    cdef cppclass PointCloudColorHandler[T]:
        # brief Constructor.
        # PointCloudColorHandler (const PointCloudConstPtr &cloud)
        PointCloudColorHandler(shared_ptr[const cpp.PointCloud[T]] &cloud)
        
        # public:
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<PointCloudColorHandler<PointT> > Ptr;
        # typedef boost::shared_ptr<const PointCloudColorHandler<PointT> > ConstPtr;
        
        # brief Destructor.
        # virtual ~PointCloudColorHandler () {}
        
        # brief Check if this handler is capable of handling the input data or not.
        # inline bool isCapable () const
        bool isCapable ()
        
        # /** \brief Abstract getName method. */
        # virtual std::string getName () const = 0;
        string getName ()
        
        # /** \brief Abstract getFieldName method. */
        # virtual std::string getFieldName () const = 0;
        string getFieldName ()
        
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        # virtual void getColor (vtkSmartPointer<vtkDataArray> &scalars) const = 0;
        # void getColor (vtkSmartPointer[vtkDataArray] &scalars)


###

# point_cloud_handlers.h(1.6.0)
# point_cloud_handlers.h -> point_cloud_geometry_handlers.h(1.7.2)
# template <typename PointT>
# class PointCloudGeometryHandler
cdef extern from "pcl/visualization/point_cloud_handlers.h" namespace "pcl::visualization" nogil:
    cdef cppclass PointCloudGeometryHandler[T]:
        # brief Constructor.
        # PointCloudGeometryHandler (const PointCloudConstPtr &cloud) :
        PointCloudGeometryHandler(shared_ptr[cpp.PointCloud[T]] &cloud)
        
        # public:
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # typedef typename boost::shared_ptr<PointCloudGeometryHandler<PointT> > Ptr;
        # typedef typename boost::shared_ptr<const PointCloudGeometryHandler<PointT> > ConstPtr;
        
        # brief Abstract getName method.
        # return the name of the class/object.
        # virtual std::string getName () const = 0;
        
        # /** \brief Abstract getFieldName method. */
        # virtual std::string getFieldName () const  = 0;
        
        # /** \brief Checl if this handler is capable of handling the input data or not. */
        # inline bool isCapable () const
        bool isCapable ()
        
        # /** \brief Obtain the actual point geometry for the input dataset in VTK format.
        #   * \param[out] points the resultant geometry
        # virtual void getGeometry (vtkSmartPointer<vtkPoints> &points) const = 0;


###

### Inheritance class ###
### handler class ###

# point_cloud_handlers.h
# template <typename PointT>
# class PointCloudColorHandlerCustom : public PointCloudColorHandler<PointT>
cdef extern from "pcl/visualization/point_cloud_handlers.h" namespace "pcl::visualization" nogil:
    cdef cppclass PointCloudColorHandlerCustom[PointT](PointCloudColorHandler[PointT]):
        # PointCloudColorHandlerCustom ()
        # brief Constructor.
        
        # /** \brief Constructor. */
        # PointCloudColorHandlerCustom (double r, double g, double b)
        PointCloudColorHandlerCustom (double r, double g, double b)
        
        # ctypedef shared_ptr[Vertices const] VerticesConstPtr
        # PointCloudColorHandlerCustom (const PointCloudConstPtr &cloud, double r, double g, double b)
        PointCloudColorHandlerCustom (const shared_ptr[cpp.PointCloud[PointT]] &cloud, double r, double g, double b)
        
        # /** \brief Destructor. */
        # virtual ~PointCloudColorHandlerCustom () {};
        
        # /** \brief Abstract getName method. */
        # virtual inline std::string getName () const
        
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const
        
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        # virtual void getColor (vtkSmartPointer<vtkDataArray> &scalars) const;


ctypedef PointCloudColorHandlerCustom[cpp.PointXYZ] PointCloudColorHandlerCustom_t
ctypedef PointCloudColorHandlerCustom[cpp.PointXYZI] PointCloudColorHandlerCustom_PointXYZI_t
ctypedef PointCloudColorHandlerCustom[cpp.PointXYZRGB] PointCloudColorHandlerCustom_PointXYZRGB_t
ctypedef PointCloudColorHandlerCustom[cpp.PointXYZRGBA] PointCloudColorHandlerCustom_PointXYZRGBA_t
ctypedef shared_ptr[PointCloudColorHandlerCustom[cpp.PointXYZ]] PointCloudColorHandlerCustom_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerCustom[cpp.PointXYZI]] PointCloudColorHandlerCustom_PointXYZI_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerCustom[cpp.PointXYZRGB]] PointCloudColorHandlerCustom_PointXYZRGB_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerCustom[cpp.PointXYZRGBA]] PointCloudColorHandlerCustom_PointXYZRGBA_Ptr_t
ctypedef PointCloudColorHandlerCustom[cpp.PointWithRange] PointCloudColorHandlerCustom_PointWithRange_t
ctypedef shared_ptr[PointCloudColorHandlerCustom[cpp.PointWithRange]] PointCloudColorHandlerCustom_PointWithRange_Ptr_t
###

# point_cloud_handlers.h
# template <typename PointT>
# class PointCloudGeometryHandlerXYZ : public PointCloudGeometryHandler<PointT>
cdef extern from "pcl/visualization/point_cloud_handlers.h" namespace "pcl::visualization" nogil:
    cdef cppclass PointCloudGeometryHandlerXYZ[PointT](PointCloudGeometryHandler[PointT]):
        PointCloudGeometryHandlerXYZ()
        # public:
        # typedef typename PointCloudGeometryHandler<PointT>::PointCloud PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # typedef typename boost::shared_ptr<PointCloudGeometryHandlerXYZ<PointT> > Ptr;
        # typedef typename boost::shared_ptr<const PointCloudGeometryHandlerXYZ<PointT> > ConstPtr;
        
        # /** \brief Constructor. */
        # PointCloudGeometryHandlerXYZ (const PointCloudConstPtr &cloud);
        
        # /** \brief Destructor. */
        # virtual ~PointCloudGeometryHandlerXYZ () {};
        
        # /** \brief Class getName method. */
        # virtual inline std::string getName () const
        
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const
        
        # /** \brief Obtain the actual point geometry for the input dataset in VTK format.
        #   * \param[out] points the resultant geometry
        # virtual void getGeometry (vtkSmartPointer<vtkPoints> &points) const;


ctypedef PointCloudGeometryHandlerXYZ[cpp.PointXYZ] PointCloudGeometryHandlerXYZ_t
ctypedef PointCloudGeometryHandlerXYZ[cpp.PointXYZI] PointCloudGeometryHandlerXYZ_PointXYZI_t
ctypedef PointCloudGeometryHandlerXYZ[cpp.PointXYZRGB] PointCloudGeometryHandlerXYZ_PointXYZRGB_t
ctypedef PointCloudGeometryHandlerXYZ[cpp.PointXYZRGBA] PointCloudGeometryHandlerXYZ_PointXYZRGBA_t
ctypedef shared_ptr[PointCloudGeometryHandlerXYZ[cpp.PointXYZ]] PointCloudGeometryHandlerXYZ_Ptr_t
ctypedef shared_ptr[PointCloudGeometryHandlerXYZ[cpp.PointXYZI]] PointCloudGeometryHandlerXYZ_PointXYZI_Ptr_t
ctypedef shared_ptr[PointCloudGeometryHandlerXYZ[cpp.PointXYZRGB]] PointCloudGeometryHandlerXYZ_PointXYZRGB_Ptr_t
ctypedef shared_ptr[PointCloudGeometryHandlerXYZ[cpp.PointXYZRGBA]] PointCloudGeometryHandlerXYZ_PointXYZRGBA_Ptr_t
###

# point_cloud_handlers.h
# template <typename PointT>
# class PointCloudGeometryHandlerSurfaceNormal : public PointCloudGeometryHandler<PointT>
cdef extern from "pcl/visualization/point_cloud_handlers.h" namespace "pcl::visualization" nogil:
    cdef cppclass PointCloudGeometryHandlerSurfaceNormal[PointT]:
        PointCloudGeometryHandlerSurfaceNormal()
        # public:
        # typedef typename PointCloudGeometryHandler<PointT>::PointCloud PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # typedef typename boost::shared_ptr<PointCloudGeometryHandlerSurfaceNormal<PointT> > Ptr;
        # typedef typename boost::shared_ptr<const PointCloudGeometryHandlerSurfaceNormal<PointT> > ConstPtr;
        
        # /** \brief Constructor. */
        # PointCloudGeometryHandlerSurfaceNormal (const PointCloudConstPtr &cloud);
        
        # /** \brief Class getName method. */
        # virtual inline std::string getName () const
        
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const
        
        # /** \brief Obtain the actual point geometry for the input dataset in VTK format.
        #   * \param[out] points the resultant geometry
        # virtual void getGeometry (vtkSmartPointer<vtkPoints> &points) const;


ctypedef PointCloudGeometryHandlerSurfaceNormal[cpp.PointXYZ] PointCloudGeometryHandlerSurfaceNormal_t
ctypedef PointCloudGeometryHandlerSurfaceNormal[cpp.PointXYZI] PointCloudGeometryHandlerSurfaceNormal_PointXYZI_t
ctypedef PointCloudGeometryHandlerSurfaceNormal[cpp.PointXYZRGB] PointCloudGeometryHandlerSurfaceNormal_PointXYZRGB_t
ctypedef PointCloudGeometryHandlerSurfaceNormal[cpp.PointXYZRGBA] PointCloudGeometryHandlerSurfaceNormal_PointXYZRGBA_t
ctypedef shared_ptr[PointCloudGeometryHandlerSurfaceNormal[cpp.PointXYZ]] PointCloudGeometryHandlerSurfaceNormal_Ptr_t
ctypedef shared_ptr[PointCloudGeometryHandlerSurfaceNormal[cpp.PointXYZI]] PointCloudGeometryHandlerSurfaceNormal_PointXYZI_Ptr_t
ctypedef shared_ptr[PointCloudGeometryHandlerSurfaceNormal[cpp.PointXYZRGB]] PointCloudGeometryHandlerSurfaceNormal_PointXYZRGB_Ptr_t
ctypedef shared_ptr[PointCloudGeometryHandlerSurfaceNormal[cpp.PointXYZRGBA]] PointCloudGeometryHandlerSurfaceNormal_PointXYZRGBA_Ptr_t
###

# point_cloud_handlers.h
# template <typename PointT>
# class PointCloudGeometryHandlerCustom : public PointCloudGeometryHandler<PointT>
cdef extern from "pcl/visualization/point_cloud_handlers.h" namespace "pcl::visualization" nogil:
    cdef cppclass PointCloudGeometryHandlerCustom[PointT]:
        PointCloudGeometryHandlerCustom()
        # public:
        # typedef typename PointCloudGeometryHandler<PointT>::PointCloud PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # typedef typename boost::shared_ptr<PointCloudGeometryHandlerCustom<PointT> > Ptr;
        # typedef typename boost::shared_ptr<const PointCloudGeometryHandlerCustom<PointT> > ConstPtr;
        # /** \brief Constructor. */
        # PointCloudGeometryHandlerCustom (const PointCloudConstPtr &cloud,
        #                                  const std::string &x_field_name,
        #                                  const std::string &y_field_name,
        #                                  const std::string &z_field_name);
        
        # /** \brief Class getName method. */
        # virtual inline std::string getName () const
        
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const
        
        # /** \brief Obtain the actual point geometry for the input dataset in VTK format.
        #   * \param[out] points the resultant geometry
        # virtual void getGeometry (vtkSmartPointer<vtkPoints> &points) const;


ctypedef PointCloudGeometryHandlerCustom[cpp.PointXYZ] PointCloudGeometryHandlerCustom_t
ctypedef PointCloudGeometryHandlerCustom[cpp.PointXYZI] PointCloudGeometryHandlerCustom_PointXYZI_t
ctypedef PointCloudGeometryHandlerCustom[cpp.PointXYZRGB] PointCloudGeometryHandlerCustom_PointXYZRGB_t
ctypedef PointCloudGeometryHandlerCustom[cpp.PointXYZRGBA] PointCloudGeometryHandlerCustom_PointXYZRGBA_t
ctypedef shared_ptr[PointCloudGeometryHandlerCustom[cpp.PointXYZ]] PointCloudGeometryHandlerCustom_Ptr_t
ctypedef shared_ptr[PointCloudGeometryHandlerCustom[cpp.PointXYZI]] PointCloudGeometryHandlerCustom_PointXYZI_Ptr_t
ctypedef shared_ptr[PointCloudGeometryHandlerCustom[cpp.PointXYZRGB]] PointCloudGeometryHandlerCustom_PointXYZRGB_Ptr_t
ctypedef shared_ptr[PointCloudGeometryHandlerCustom[cpp.PointXYZRGBA]] PointCloudGeometryHandlerCustom_PointXYZRGBA_Ptr_t
###

# point_cloud_handlers.h
# template <>
# class PCL_EXPORTS PointCloudGeometryHandler<sensor_msgs::PointCloud2>
        # public:
        # typedef sensor_msgs::PointCloud2 PointCloud;
        # typedef PointCloud::Ptr PointCloudPtr;
        # typedef PointCloud::ConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<PointCloudGeometryHandler<PointCloud> > Ptr;
        # typedef boost::shared_ptr<const PointCloudGeometryHandler<PointCloud> > ConstPtr;
        
        # /** \brief Constructor. */
        # PointCloudGeometryHandler (const PointCloudConstPtr &cloud, const Eigen::Vector4f &sensor_origin = Eigen::Vector4f::Zero ())
        
        # /** \brief Abstract getName method. */
        # virtual std::string getName () const = 0;
        
        # /** \brief Abstract getFieldName method. */
        # virtual std::string getFieldName () const  = 0;
        
        # /** \brief Check if this handler is capable of handling the input data or not. */
        # inline bool isCapable () const { return (capable_); }
        
        # /** \brief Obtain the actual point geometry for the input dataset in VTK format.
        #   * \param[out] points the resultant geometry
        # virtual void getGeometry (vtkSmartPointer<vtkPoints> &points) const;
###

# point_cloud_handlers.h
# template <>
# class PCL_EXPORTS PointCloudGeometryHandlerXYZ<sensor_msgs::PointCloud2> : public PointCloudGeometryHandler<sensor_msgs::PointCloud2>
        # public:
        # typedef PointCloudGeometryHandler<sensor_msgs::PointCloud2>::PointCloud PointCloud;
        # typedef PointCloud::Ptr PointCloudPtr;
        # typedef PointCloud::ConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<PointCloudGeometryHandlerXYZ<PointCloud> > Ptr;
        # typedef boost::shared_ptr<const PointCloudGeometryHandlerXYZ<PointCloud> > ConstPtr;
        # /** \brief Constructor. */
        # PointCloudGeometryHandlerXYZ (const PointCloudConstPtr &cloud);
        
        # /** \brief Destructor. */
        # virtual ~PointCloudGeometryHandlerXYZ () {}
        
        # /** \brief Class getName method. */
        # virtual inline std::string getName () const { return ("PointCloudGeometryHandlerXYZ"); }
        
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const { return ("xyz"); }
###

# point_cloud_handlers.h
# template <>
# class PCL_EXPORTS PointCloudGeometryHandlerSurfaceNormal<sensor_msgs::PointCloud2> : public PointCloudGeometryHandler<sensor_msgs::PointCloud2>
        # public:
        # typedef PointCloudGeometryHandler<sensor_msgs::PointCloud2>::PointCloud PointCloud;
        # typedef PointCloud::Ptr PointCloudPtr;
        # typedef PointCloud::ConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<PointCloudGeometryHandlerSurfaceNormal<PointCloud> > Ptr;
        # typedef boost::shared_ptr<const PointCloudGeometryHandlerSurfaceNormal<PointCloud> > ConstPtr;
        # /** \brief Constructor. */
        # PointCloudGeometryHandlerSurfaceNormal (const PointCloudConstPtr &cloud);
        
        # /** \brief Class getName method. */
        # virtual inline std::string getName () const { return ("PointCloudGeometryHandlerSurfaceNormal"); }
        
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const { return ("normal_xyz"); }
###

# point_cloud_handlers.h
# template <>
# class PCL_EXPORTS PointCloudGeometryHandlerCustom<sensor_msgs::PointCloud2> : public PointCloudGeometryHandler<sensor_msgs::PointCloud2>
        # public:
        # typedef PointCloudGeometryHandler<sensor_msgs::PointCloud2>::PointCloud PointCloud;
        # typedef PointCloud::Ptr PointCloudPtr;
        # typedef PointCloud::ConstPtr PointCloudConstPtr;
        # /** \brief Constructor. */
        # PointCloudGeometryHandlerCustom (const PointCloudConstPtr &cloud,
        #                                  const std::string &x_field_name,
        #                                  const std::string &y_field_name,
        #                                  const std::string &z_field_name);
        # /** \brief Destructor. */
        # virtual ~PointCloudGeometryHandlerCustom () {}
        
        # /** \brief Class getName method. */
        # virtual inline std::string getName () const { return ("PointCloudGeometryHandlerCustom"); }
        
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const { return (field_name_); }


###

# point_cloud_handlers.h
# template <typename PointT>
# class PointCloudColorHandlerRandom : public PointCloudColorHandler<PointT>
cdef extern from "pcl/visualization/point_cloud_handlers.h" namespace "pcl::visualization" nogil:
    cdef cppclass PointCloudColorHandlerRandom[PointT](PointCloudColorHandler[PointT]):
        PointCloudColorHandlerRandom()
        # typedef typename PointCloudColorHandler<PointT>::PointCloud PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        
        # public:
        # typedef boost::shared_ptr<PointCloudColorHandlerRandom<PointT> > Ptr;
        # typedef boost::shared_ptr<const PointCloudColorHandlerRandom<PointT> > ConstPtr;
        
        # /** \brief Constructor. */
        # PointCloudColorHandlerRandom (const PointCloudConstPtr &cloud) :
        
        # /** \brief Abstract getName method. */
        # virtual inline std::string getName () const
        
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const
        
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        # virtual void getColor (vtkSmartPointer<vtkDataArray> &scalars) const;


ctypedef PointCloudColorHandlerRandom[cpp.PointXYZ] PointCloudColorHandlerRandom_t
ctypedef PointCloudColorHandlerRandom[cpp.PointXYZI] PointCloudColorHandlerRandom_PointXYZI_t
ctypedef PointCloudColorHandlerRandom[cpp.PointXYZRGB] PointCloudColorHandlerRandom_PointXYZRGB_t
ctypedef PointCloudColorHandlerRandom[cpp.PointXYZRGBA] PointCloudColorHandlerRandom_PointXYZRGBA_t
ctypedef shared_ptr[PointCloudColorHandlerRandom[cpp.PointXYZ]] PointCloudColorHandlerRandom_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerRandom[cpp.PointXYZI]] PointCloudColorHandlerRandom_PointXYZI_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerRandom[cpp.PointXYZRGB]] PointCloudColorHandlerRandom_PointXYZRGB_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerRandom[cpp.PointXYZRGBA]] PointCloudColorHandlerRandom_PointXYZRGBA_Ptr_t
###

# point_cloud_handlers.h
# template <typename PointT>
# class PointCloudColorHandlerRGBField : public PointCloudColorHandler<PointT>
cdef extern from "pcl/visualization/point_cloud_handlers.h" namespace "pcl::visualization" nogil:
    cdef cppclass PointCloudColorHandlerRGBField[PointT](PointCloudColorHandler[PointT]):
        # PointCloudColorHandlerRGBField ()
        # /** \brief Constructor. */
        # PointCloudColorHandlerRGBField (const PointCloudConstPtr &cloud);
        PointCloudColorHandlerRGBField (const shared_ptr[cpp.PointCloud[PointT]] &cloud)
        
        # typedef typename PointCloudColorHandler<PointT>::PointCloud PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # public:
        # typedef boost::shared_ptr<PointCloudColorHandlerRGBField<PointT> > Ptr;
        # typedef boost::shared_ptr<const PointCloudColorHandlerRGBField<PointT> > ConstPtr;
        
        # /** \brief Constructor. */
        # PointCloudColorHandlerRGBField (const PointCloudConstPtr &cloud);
        
        # /** \brief Destructor. */
        # virtual ~PointCloudColorHandlerRGBField () {}
        
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const { return ("rgb"); }
        
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        # virtual void getColor (vtkSmartPointer<vtkDataArray> &scalars) const;


ctypedef PointCloudColorHandlerRGBField[cpp.PointXYZ] PointCloudColorHandlerRGBField_t
ctypedef PointCloudColorHandlerRGBField[cpp.PointXYZI] PointCloudColorHandlerRGBField_PointXYZI_t
ctypedef PointCloudColorHandlerRGBField[cpp.PointXYZRGB] PointCloudColorHandlerRGBField_PointXYZRGB_t
ctypedef PointCloudColorHandlerRGBField[cpp.PointXYZRGBA] PointCloudColorHandlerRGBField_PointXYZRGBA_t
ctypedef shared_ptr[PointCloudColorHandlerRGBField[cpp.PointXYZ]] PointCloudColorHandlerRGBField_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerRGBField[cpp.PointXYZI]] PointCloudColorHandlerRGBField_PointXYZI_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerRGBField[cpp.PointXYZRGB]] PointCloudColorHandlerRGBField_PointXYZRGB_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerRGBField[cpp.PointXYZRGBA]] PointCloudColorHandlerRGBField_PointXYZRGBA_Ptr_t
###

# point_cloud_handlers.h
# template <typename PointT>
# class PointCloudColorHandlerHSVField : public PointCloudColorHandler<PointT>
cdef extern from "pcl/visualization/point_cloud_handlers.h" namespace "pcl::visualization" nogil:
    cdef cppclass PointCloudColorHandlerHSVField[PointT](PointCloudColorHandler[PointT]):
        # PointCloudColorHandlerHSVField ()
        # /** \brief Constructor. */
        # PointCloudColorHandlerHSVField (const PointCloudConstPtr &cloud);
        PointCloudColorHandlerHSVField (const shared_ptr[cpp.PointCloud[PointT]] &cloud)
        
        # typedef typename PointCloudColorHandler<PointT>::PointCloud PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # public:
        # typedef boost::shared_ptr<PointCloudColorHandlerHSVField<PointT> > Ptr;
        # typedef boost::shared_ptr<const PointCloudColorHandlerHSVField<PointT> > ConstPtr;
        
        
        
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const { return ("hsv"); }
        
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        #   */
        # virtual void getColor (vtkSmartPointer<vtkDataArray> &scalars) const;


ctypedef PointCloudColorHandlerHSVField[cpp.PointXYZ] PointCloudColorHandlerHSVField_t
ctypedef PointCloudColorHandlerHSVField[cpp.PointXYZI] PointCloudColorHandlerHSVField_PointXYZI_t
ctypedef PointCloudColorHandlerHSVField[cpp.PointXYZRGB] PointCloudColorHandlerHSVField_PointXYZRGB_t
ctypedef PointCloudColorHandlerHSVField[cpp.PointXYZRGBA] PointCloudColorHandlerHSVField_PointXYZRGBA_t
ctypedef shared_ptr[PointCloudColorHandlerHSVField[cpp.PointXYZ]] PointCloudColorHandlerHSVField_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerHSVField[cpp.PointXYZI]] PointCloudColorHandlerHSVField_PointXYZI_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerHSVField[cpp.PointXYZRGB]] PointCloudColorHandlerHSVField_PointXYZRGB_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerHSVField[cpp.PointXYZRGBA]] PointCloudColorHandlerHSVField_PointXYZRGBA_Ptr_t
###

# point_cloud_handlers.h
# template <typename PointT>
# class PointCloudColorHandlerGenericField : public PointCloudColorHandler<PointT>
cdef extern from "pcl/visualization/point_cloud_handlers.h" namespace "pcl::visualization" nogil:
    cdef cppclass PointCloudColorHandlerGenericField[PointT](PointCloudColorHandler[PointT]):
        PointCloudColorHandlerGenericField ()
        # /** \brief Constructor. */
        # PointCloudColorHandlerGenericField (const PointCloudConstPtr &cloud, const std::string &field_name);
        PointCloudColorHandlerGenericField (const shared_ptr[cpp.PointCloud[PointT]] &cloud, const string &field_name)
        
        # typedef typename PointCloudColorHandler<PointT>::PointCloud PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # public:
        # typedef boost::shared_ptr<PointCloudColorHandlerGenericField<PointT> > Ptr;
        # typedef boost::shared_ptr<const PointCloudColorHandlerGenericField<PointT> > ConstPtr;
        
        # /** \brief Destructor. */
        # virtual ~PointCloudColorHandlerGenericField () {}
        
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const { return (field_name_); }
        
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        # virtual void getColor (vtkSmartPointer<vtkDataArray> &scalars) const;


ctypedef PointCloudColorHandlerGenericField[cpp.PointXYZ] PointCloudColorHandlerGenericField_t
ctypedef PointCloudColorHandlerGenericField[cpp.PointXYZI] PointCloudColorHandlerGenericField_PointXYZI_t
ctypedef PointCloudColorHandlerGenericField[cpp.PointXYZRGB] PointCloudColorHandlerGenericField_PointXYZRGB_t
ctypedef PointCloudColorHandlerGenericField[cpp.PointXYZRGBA] PointCloudColorHandlerGenericField_PointXYZRGBA_t
ctypedef shared_ptr[PointCloudColorHandlerGenericField[cpp.PointXYZ]] PointCloudColorHandlerGenericField_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerGenericField[cpp.PointXYZI]] PointCloudColorHandlerGenericField_PointXYZI_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerGenericField[cpp.PointXYZRGB]] PointCloudColorHandlerGenericField_PointXYZRGB_Ptr_t
ctypedef shared_ptr[PointCloudColorHandlerGenericField[cpp.PointXYZRGBA]] PointCloudColorHandlerGenericField_PointXYZRGBA_Ptr_t
###

# point_cloud_handlers.h
# template <>
# class PCL_EXPORTS PointCloudColorHandler<sensor_msgs::PointCloud2>
        # public:
        # typedef sensor_msgs::PointCloud2 PointCloud;
        # typedef PointCloud::Ptr PointCloudPtr;
        # typedef PointCloud::ConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<PointCloudColorHandler<PointCloud> > Ptr;
        # typedef boost::shared_ptr<const PointCloudColorHandler<PointCloud> > ConstPtr;
        # /** \brief Constructor. */
        # PointCloudColorHandler (const PointCloudConstPtr &cloud) :
        # /** \brief Destructor. */
        # virtual ~PointCloudColorHandler () {}
        # /** \brief Return whether this handler is capable of handling the input data or not. */
        # inline bool
        # isCapable () const { return (capable_); }
        # /** \brief Abstract getName method. */
        # virtual std::string
        # getName () const = 0;
        # /** \brief Abstract getFieldName method. */
        # virtual std::string
        # getFieldName () const = 0;
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        # virtual void
        # getColor (vtkSmartPointer<vtkDataArray> &scalars) const = 0;


###

# template <>
# class PCL_EXPORTS PointCloudColorHandlerRandom<sensor_msgs::PointCloud2> : public PointCloudColorHandler<sensor_msgs::PointCloud2>
        # typedef PointCloudColorHandler<sensor_msgs::PointCloud2>::PointCloud PointCloud;
        # typedef PointCloud::Ptr PointCloudPtr;
        # typedef PointCloud::ConstPtr PointCloudConstPtr;
        # public:
        # typedef boost::shared_ptr<PointCloudColorHandlerRandom<PointCloud> > Ptr;
        # typedef boost::shared_ptr<const PointCloudColorHandlerRandom<PointCloud> > ConstPtr;
        # /** \brief Constructor. */
        # PointCloudColorHandlerRandom (const PointCloudConstPtr &cloud) :
        # /** \brief Get the name of the class. */
        # virtual inline std::string getName () const
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        # virtual void getColor (vtkSmartPointer<vtkDataArray> &scalars) const;
###

# template <>
# class PCL_EXPORTS PointCloudColorHandlerCustom<sensor_msgs::PointCloud2> : public PointCloudColorHandler<sensor_msgs::PointCloud2>
        # typedef PointCloudColorHandler<sensor_msgs::PointCloud2>::PointCloud PointCloud;
        # typedef PointCloud::Ptr PointCloudPtr;
        # typedef PointCloud::ConstPtr PointCloudConstPtr;
        # public:
        # /** \brief Constructor. */
        # PointCloudColorHandlerCustom (const PointCloudConstPtr &cloud, double r, double g, double b) :
        # /** \brief Get the name of the class. */
        # virtual inline std::string getName () const
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        # virtual void getColor (vtkSmartPointer<vtkDataArray> &scalars) const;
        # protected:
        # /** \brief Internal R, G, B holding the values given by the user. */
        # double r_, g_, b_;
###

# template <>
# class PCL_EXPORTS PointCloudColorHandlerRGBField<sensor_msgs::PointCloud2> : public PointCloudColorHandler<sensor_msgs::PointCloud2>
        # typedef PointCloudColorHandler<sensor_msgs::PointCloud2>::PointCloud PointCloud;
        # typedef PointCloud::Ptr PointCloudPtr;
        # typedef PointCloud::ConstPtr PointCloudConstPtr;
        # public:
        # typedef boost::shared_ptr<PointCloudColorHandlerRGBField<PointCloud> > Ptr;
        # typedef boost::shared_ptr<const PointCloudColorHandlerRGBField<PointCloud> > ConstPtr;
        # /** \brief Constructor. */
        # PointCloudColorHandlerRGBField (const PointCloudConstPtr &cloud);
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        # virtual void getColor (vtkSmartPointer<vtkDataArray> &scalars) const;
        # protected:
        # /** \brief Get the name of the class. */
        # virtual inline std::string getName () const { return ("PointCloudColorHandlerRGBField"); }
        # /** \brief Get the name of the field used. */
        # virtual std::string getFieldName () const { return ("rgb"); }
###

# template <>
# class PCL_EXPORTS PointCloudColorHandlerHSVField<sensor_msgs::PointCloud2> : public PointCloudColorHandler<sensor_msgs::PointCloud2>
        # typedef PointCloudColorHandler<sensor_msgs::PointCloud2>::PointCloud PointCloud;
        # typedef PointCloud::Ptr PointCloudPtr;
        # typedef PointCloud::ConstPtr PointCloudConstPtr;
        # public:
        # typedef boost::shared_ptr<PointCloudColorHandlerHSVField<PointCloud> > Ptr;
        # typedef boost::shared_ptr<const PointCloudColorHandlerHSVField<PointCloud> > ConstPtr;
        # /** \brief Constructor. */
        # PointCloudColorHandlerHSVField (const PointCloudConstPtr &cloud);
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        # virtual void getColor (vtkSmartPointer<vtkDataArray> &scalars) const;


###

# template <>
# class PCL_EXPORTS PointCloudColorHandlerGenericField<sensor_msgs::PointCloud2> : public PointCloudColorHandler<sensor_msgs::PointCloud2>
        # typedef PointCloudColorHandler<sensor_msgs::PointCloud2>::PointCloud PointCloud;
        # typedef PointCloud::Ptr PointCloudPtr;
        # typedef PointCloud::ConstPtr PointCloudConstPtr;
        # public:
        # typedef boost::shared_ptr<PointCloudColorHandlerGenericField<PointCloud> > Ptr;
        # typedef boost::shared_ptr<const PointCloudColorHandlerGenericField<PointCloud> > ConstPtr;
        # /** \brief Constructor. */
        # PointCloudColorHandlerGenericField (const PointCloudConstPtr &cloud, const std::string &field_name);
        
        # /** \brief Obtain the actual color for the input dataset as vtk scalars.
        #   * \param[out] scalars the output scalars containing the color for the dataset
        # virtual void getColor (vtkSmartPointer<vtkDataArray> &scalars) const;


###


# pcl_visualizer.h
# class PCL_EXPORTS PCLVisualizer
cdef extern from "pcl/visualization/pcl_visualizer.h" namespace "pcl::visualization" nogil:
    cdef cppclass PCLVisualizer:
        PCLVisualizer()
        # public:
        # brief PCL Visualizer constructor.
        # param[in] name the window name (empty by default)
        # param[in] create_interactor if true (default), create an interactor, false otherwise
        # PCLVisualizer (const std::string &name = "", const bool create_interactor = true);
        PCLVisualizer (const string name, bool create_interactor)
        
        # brief PCL Visualizer constructor.
        # param[in] argc
        # param[in] argv
        # param[in] name the window name (empty by default)
        # param[in] style interactor style (defaults to PCLVisualizerInteractorStyle)
        # param[in] create_interactor if true (default), create an interactor, false otherwise
        # PCLVisualizer (int &argc, char **argv, const std::string &name = "", PCLVisualizerInteractorStyle* style = PCLVisualizerInteractorStyle::New (), const bool create_interactor = true);
        # 
        # PCLVisualizer (int &argc, char **argv, const std::string &name = "", PCLVisualizerInteractorStyle* style = PCLVisualizerInteractorStyle::New (), const bool create_interactor = true)
        
        # brief PCL Visualizer destructor.
        # virtual ~PCLVisualizer ();
        
        # brief Enables/Disabled the underlying window mode to full screen.
        # note This might or might not work, depending on your window manager.
        # See the VTK documentation for additional details.
        # param[in] mode true for full screen, false otherwise
        # inline void setFullScreen (bool mode)
        void setFullScreen (bool mode)
        
        # brief Enables or disable the underlying window borders.
        # note This might or might not work, depending on your window manager.
        # See the VTK documentation for additional details.
        # param[in] mode true for borders, false otherwise
        # inline void setWindowBorders (bool mode)
        void setWindowBorders (bool mode)
        
        # brief Register a callback boost::function for keyboard events
        # param[in] cb a boost function that will be registered as a callback for a keyboard event
        # return a connection object that allows to disconnect the callback function.
        # boost::signals2::connection registerKeyboardCallback (boost::function<void (const pcl::visualization::KeyboardEvent&)> cb);
        
        # brief Register a callback function for keyboard events
        # param[in] callback  the function that will be registered as a callback for a keyboard event
        # param[in] cookie    user data that is passed to the callback
        # return a connection object that allows to disconnect the callback function.
        # 
        # inline boost::signals2::connection
        # registerKeyboardCallback (void (*callback) (const pcl::visualization::KeyboardEvent&, void*), void* cookie = NULL)
        
        # brief Register a callback function for keyboard events
        # param[in] callback the member function that will be registered as a callback for a keyboard event
        # param[in] instance instance to the class that implements the callback function
        # param[in] cookie   user data that is passed to the callback
        # return a connection object that allows to disconnect the callback function.
        # 
        # template<typename T> inline boost::signals2::connection
        # registerKeyboardCallback (void (T::*callback) (const pcl::visualization::KeyboardEvent&, void*), T& instance, void* cookie = NULL)
        
        # brief Register a callback function for mouse events
        # param[in] cb a boost function that will be registered as a callback for a mouse event
        # return a connection object that allows to disconnect the callback function.
        # 
        # boost::signals2::connection
        # registerMouseCallback (boost::function<void (const pcl::visualization::MouseEvent&)> cb);
        
        # brief Register a callback function for mouse events
        # param[in] callback  the function that will be registered as a callback for a mouse event
        # param[in] cookie    user data that is passed to the callback
        # return a connection object that allows to disconnect the callback function.
        # 
        # inline boost::signals2::connection
        # registerMouseCallback (void (*callback) (const pcl::visualization::MouseEvent&, void*), void* cookie = NULL)
        
        # brief Register a callback function for mouse events
        # param[in] callback  the member function that will be registered as a callback for a mouse event
        # param[in] instance  instance to the class that implements the callback function
        # param[in] cookie    user data that is passed to the callback
        # return a connection object that allows to disconnect the callback function.
        # 
        # template<typename T> inline boost::signals2::connection
        # registerMouseCallback (void (T::*callback) (const pcl::visualization::MouseEvent&, void*), T& instance, void* cookie = NULL)
        
        # brief Register a callback function for point picking events
        # param[in] cb a boost function that will be registered as a callback for a point picking event
        # return a connection object that allows to disconnect the callback function.
        # 
        # boost::signals2::connection
        # registerPointPickingCallback (boost::function<void (const pcl::visualization::PointPickingEvent&)> cb);
        
        # brief Register a callback function for point picking events
        # param[in] callback  the function that will be registered as a callback for a point picking event
        # param[in] cookie    user data that is passed to the callback
        # return a connection object that allows to disconnect the callback function.
        # 
        # inline boost::signals2::connection
        # registerPointPickingCallback (void (*callback) (const pcl::visualization::PointPickingEvent&, void*), void* cookie = NULL)
        
        # brief Register a callback function for point picking events
        # param[in] callback  the member function that will be registered as a callback for a point picking event
        # param[in] instance  instance to the class that implements the callback function
        # param[in] cookie    user data that is passed to the callback
        # return a connection object that allows to disconnect the callback function.
        # 
        # template<typename T> inline boost::signals2::connection
        # registerPointPickingCallback (void (T::*callback) (const pcl::visualization::PointPickingEvent&, void*), T& instance, void* cookie = NULL)
        
        # brief Spin method. Calls the interactor and runs an internal loop.
        void spin ()
        
        # brief Spin once method. Calls the interactor and updates the screen once.
        # param[in] time - How long (in ms) should the visualization loop be allowed to run.
        # param[in] force_redraw - if false it might return without doing anything if the
        # interactor's framerate does not require a redraw yet.
        # void spinOnce (int time = 1, bool force_redraw = false)
        void spinOnce (int time, bool force_redraw)
        
        # brief Adds 3D axes describing a coordinate system to screen at 0,0,0.
        # param[in] scale the scale of the axes (default: 1)
        # param[in] viewport the view port where the 3D axes should be added (default: all)
        # 
        # void addCoordinateSystem (double scale = 1.0, int viewport = 0);
        void addCoordinateSystem (double scale, int viewport)
        
        # brief Adds 3D axes describing a coordinate system to screen at x, y, z
        # param[in] scale the scale of the axes (default: 1)
        # param[in] x the X position of the axes
        # param[in] y the Y position of the axes
        # param[in] z the Z position of the axes
        # param[in] viewport the view port where the 3D axes should be added (default: all)
        # 
        # void addCoordinateSystem (double scale, float x, float y, float z, int viewport = 0);
        void addCoordinateSystem (double scale, float x, float y, float z, int viewport)
        
        # brief Adds 3D axes describing a coordinate system to screen at x, y, z, Roll,Pitch,Yaw
        # param[in] scale the scale of the axes (default: 1)
        # param[in] t transformation matrix
        # param[in] viewport the view port where the 3D axes should be added (default: all)
        # RPY Angles
        # Rotate the reference frame by the angle roll about axis x
        # Rotate the reference frame by the angle pitch about axis y
        # Rotate the reference frame by the angle yaw about axis z
        # Description:
        # Sets the orientation of the Prop3D.  Orientation is specified as
        # X,Y and Z rotations in that order, but they are performed as
        # RotateZ, RotateX, and finally RotateY.
        # All axies use right hand rule. x=red axis, y=green axis, z=blue axis
        # z direction is point into the screen.
        #     z
        #      \
        #       \
        #        \
        #         -----------> x
        #         |
        #         |
        #         |
        #         |
        #         |
        #         |
        #         y
        # 
        # void addCoordinateSystem (double scale, const Eigen::Affine3f& t, int viewport = 0);
        void addCoordinateSystem (double scale, const eigen3.Affine3f& t, int viewport)
        
        # brief Removes a previously added 3D axes (coordinate system)
        # param[in] viewport view port where the 3D axes should be removed from (default: all)
        # bool removeCoordinateSystem (int viewport = 0);
        bool removeCoordinateSystem (int viewport)
        
        # brief Removes a Point Cloud from screen, based on a given ID.
        # param[in] id the point cloud object id (i.e., given on \a addPointCloud)
        # param[in] viewport view port from where the Point Cloud should be removed (default: all)
        # return true if the point cloud is successfully removed and false if the point cloud is
        # not actually displayed
        # bool removePointCloud (const std::string &id = "cloud", int viewport = 0);
        bool removePointCloud (const string &id, int viewport)
        
        # brief Removes a PolygonMesh from screen, based on a given ID.
        # param[in] id the polygon object id (i.e., given on \a addPolygonMesh)
        # param[in] viewport view port from where the PolygonMesh should be removed (default: all)
        # inline bool removePolygonMesh (const std::string &id = "polygon", int viewport = 0)
        bool removePolygonMesh (const string &id, int viewport)
        
        # brief Removes an added shape from screen (line, polygon, etc.), based on a given ID
        # note This methods also removes PolygonMesh objects and PointClouds, if they match the ID
        # param[in] id the shape object id (i.e., given on \a addLine etc.)
        # param[in] viewport view port from where the Point Cloud should be removed (default: all)
        # bool removeShape (const std::string &id = "cloud", int viewport = 0);
        bool removeShape (const string &id, int viewport)
        
        # brief Removes an added 3D text from the scene, based on a given ID
        # param[in] id the 3D text id (i.e., given on \a addText3D etc.)
        # param[in] viewport view port from where the 3D text should be removed (default: all)
        # bool removeText3D (const std::string &id = "cloud", int viewport = 0);
        bool removeText3D (const string &id, int viewport)
        
        # brief Remove all point cloud data on screen from the given viewport.
        # param[in] viewport view port from where the clouds should be removed (default: all)
        # bool removeAllPointClouds (int viewport = 0);
        bool removeAllPointClouds (int viewport)
        
        # brief Remove all 3D shape data on screen from the given viewport.
        # param[in] viewport view port from where the shapes should be removed (default: all)
        # bool removeAllShapes (int viewport = 0);
        bool removeAllShapes (int viewport)
        
        # brief Set the viewport's background color.
        # param[in] r the red component of the RGB color
        # param[in] g the green component of the RGB color
        # param[in] b the blue component of the RGB color
        # param[in] viewport the view port (default: all)
        # void setBackgroundColor (const double &r, const double &g, const double &b, int viewport = 0);
        void setBackgroundColor (const double &r, const double &g, const double &b, int viewport)
        
        ### addText function
        # brief Add a text to screen
        # param[in] text the text to add
        # param[in] xpos the X position on screen where the text should be added
        # param[in] ypos the Y position on screen where the text should be added
        # param[in] id the text object id (default: equal to the "text" parameter)
        # param[in] viewport the view port (default: all)
        # bool addText (
        #          const std::string &text,
        #          int xpos, int ypos,
        #          const std::string &id = "", int viewport = 0);
        bool addText (const string &text, int xpos, int ypos, const string &id, int viewport)
        
        # brief Add a text to screen
        # param[in] text the text to add
        # param[in] xpos the X position on screen where the text should be added
        # param[in] ypos the Y position on screen where the text should be added
        # param[in] r the red color value
        # param[in] g the green color value
        # param[in] b the blue color vlaue
        # param[in] id the text object id (default: equal to the "text" parameter)
        # param[in] viewport the view port (default: all)
        # bool addText (const std::string &text, int xpos, int ypos, double r, double g, double b,
        #               const std::string &id = "", int viewport = 0);
        bool addText (const string &text, int xpos, int ypos, double r, double g, double b, const string &id, int viewport)
        # bool addText_rgb "addText" (const string &text, int xpos, int ypos, double r, double g, double b, const string &id, int viewport)
        
        # brief Add a text to screen
        # param[in] text the text to add
        # param[in] xpos the X position on screen where the text should be added
        # param[in] ypos the Y position on screen where the text should be added
        # param[in] fontsize the fontsize of the text
        # param[in] r the red color value
        # param[in] g the green color value
        # param[in] b the blue color vlaue
        # param[in] id the text object id (default: equal to the "text" parameter)
        # param[in] viewport the view port (default: all)
        # bool addText (const std::string &text, int xpos, int ypos, int fontsize, double r, double g, double b,
        #               const std::string &id = "", int viewport = 0);
        bool addText (const string &text, int xpos, int ypos, int fontsize, double r, double g, double b, const string &id, int viewport)
        # bool addText_rgb_ftsize "addText" (const string &text, int xpos, int ypos, int fontsize, double r, double g, double b, const string &id, int viewport)
        
        ### addText function
        
        ### updateText function
        # brief Update a text to screen
        # param[in] text the text to update
        # param[in] xpos the new X position on screen
        # param[in] ypos the new Y position on screen 
        # param[in] id the text object id (default: equal to the "text" parameter)
        bool updateText (const string &text, int xpos, int ypos, const string &id)
        
        # brief Update a text to screen
        # param[in] text the text to update
        # param[in] xpos the new X position on screen
        # param[in] ypos the new Y position on screen 
        # param[in] r the red color value
        # param[in] g the green color value
        # param[in] b the blue color vlaue
        # param[in] id the text object id (default: equal to the "text" parameter)
        # bool updateText (const std::string &text, 
        #                  int xpos, int ypos, double r, double g, double b,
        #                  const std::string &id = "");
        bool updateText (const string &text, int xpos, int ypos, double r, double g, double b, const string &id)
        # bool updateText_rgb "updateText" (const string &text, int xpos, int ypos, double r, double g, double b, const string &id)
        
        # brief Update a text to screen
        # param[in] text the text to update
        # param[in] xpos the new X position on screen
        # param[in] ypos the new Y position on screen 
        # param[in] fontsize the fontsize of the text
        # param[in] r the red color value
        # param[in] g the green color value
        # param[in] b the blue color vlaue
        # param[in] id the text object id (default: equal to the "text" parameter)
        # bool updateText (const std::string &text, 
        #                  int xpos, int ypos, int fontsize, double r, double g, double b,
        #                  const std::string &id = "");
        bool updateText (const string &text, int xpos, int ypos, int fontsize, double r, double g, double b, const string &id)
        # bool updateText_rgb_ftsize "updateText" (const string &text, int xpos, int ypos, int fontsize, double r, double g, double b, const string &id)
        
        ### updateText function
        
        # brief Set the pose of an existing shape. 
        # Returns false if the shape doesn't exist, true if the pose was succesfully 
        # updated.
        # param[in] id the shape or cloud object id (i.e., given on \a addLine etc.)
        # param[in] pose the new pose
        # return false if no shape or cloud with the specified ID was found
        # bool updateShapePose (const std::string &id, const Eigen::Affine3f& pose);
        bool updateShapePose (const string &id, const eigen3.Affine3f& pose)
        
        # brief Add a 3d text to the scene
        # param[in] text the text to add
        # param[in] position the world position where the text should be added
        # param[in] textScale the scale of the text to render
        # param[in] r the red color value
        # param[in] g the green color value
        # param[in] b the blue color value
        # param[in] id the text object id (default: equal to the "text" parameter)
        # param[in] viewport the view port (default: all)
        # template <typename PointT> bool
        # addText3D (const std::string &text,
        #            const PointT &position,
        #            double textScale = 1.0,
        #            double r = 1.0, double g = 1.0, double b = 1.0, const std::string &id = "", int viewport = 0);
        bool addText3D[PointT](const string &text, const PointT &position, double textScale, double r, double g, double b, const string &id, int viewport)
        
        ###
        # brief Add the estimated surface normals of a Point Cloud to screen.
        # param[in] cloud the input point cloud dataset containing XYZ data and normals
        # param[in] level display only every level'th point (default: 100)
        # param[in] scale the normal arrow scale (default: 0.02m)
        # param[in] id the point cloud object id (default: cloud)
        # param[in] viewport the view port where the Point Cloud should be added (default: all)
        # template <typename PointNT> bool
        # addPointCloudNormals (const typename pcl::PointCloud<PointNT>::ConstPtr &cloud, int level = 100, double scale = 0.02, const std::string &id = "cloud", int viewport = 0);
        bool addPointCloudNormals[PointNT](cpp.PointCloud[PointNT] cloud, int level, double scale, string id, int viewport)
        
        # brief Add the estimated surface normals of a Point Cloud to screen.
        # param[in] cloud the input point cloud dataset containing the XYZ data
        # param[in] normals the input point cloud dataset containing the normal data
        # param[in] level display only every level'th point (default: 100)
        # param[in] scale the normal arrow scale (default: 0.02m)
        # param[in] id the point cloud object id (default: cloud)
        # param[in] viewport the view port where the Point Cloud should be added (default: all)
        # template <typename PointT, typename PointNT> bool
        # addPointCloudNormals (const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
        #                       const typename pcl::PointCloud<PointNT>::ConstPtr &normals,
        #                       int level = 100, double scale = 0.02, const std::string &id = "cloud", int viewport = 0);
        bool addPointCloudNormals [PointT, PointNT] (const shared_ptr[cpp.PointCloud[PointT]] &cloud, const shared_ptr[cpp.PointCloud[PointNT]] &normals, int level, double scale, const string &id, int viewport)
        
        ### addPointCloudPrincipalCurvatures function ###
        ### PCL 1.6.0 NG (not define)
        ### PCL 1.7.2 
        # brief Add the estimated principal curvatures of a Point Cloud to screen.
        # param[in] cloud the input point cloud dataset containing the XYZ data
        # param[in] normals the input point cloud dataset containing the normal data
        # param[in] pcs the input point cloud dataset containing the principal curvatures data
        # param[in] level display only every level'th point. Default: 100
        # param[in] scale the normal arrow scale. Default: 1.0
        # param[in] id the point cloud object id. Default: "cloud"
        # param[in] viewport the view port where the Point Cloud should be added (default: all)
        # bool addPointCloudPrincipalCurvatures (
        #     const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
        #     const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
        #     const pcl::PointCloud<pcl::PrincipalCurvatures>::ConstPtr &pcs,
        #     int level = 100, double scale = 1.0,
        #     const std::string &id = "cloud", int viewport = 0);
        # bool addPointCloudPrincipalCurvatures (
        #             const shared_ptr[cpp.PointCloud[cpp.PointXYZ]] &cloud,
        #             const shared_ptr[cpp.PointCloud[cpp.Normal]] &normals,
        #             const shared_ptr[cpp.PointCloud[cpp.PrincipalCurvatures]] &pcs,
        #             int level, double scale, string &id, int viewport)
        
        ### addPointCloudPrincipalCurvatures function ###
        
        ### updatePointCloud Functions ###
        # brief Updates the XYZ data for an existing cloud object id on screen.
        # param[in] cloud the input point cloud dataset
        # param[in] id the point cloud object id to update (default: cloud)
        # return false if no cloud with the specified ID was found
        # template <typename PointT> bool updatePointCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::string &id = "cloud");
        bool updatePointCloud[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, string &id)
        
        # brief Updates the XYZ data for an existing cloud object id on screen.
        # param[in] cloud the input point cloud dataset
        # param[in] geometry_handler the geometry handler to use
        # param[in] id the point cloud object id to update (default: cloud)
        # return false if no cloud with the specified ID was found
        # template <typename PointT> bool
        # updatePointCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const PointCloudGeometryHandler<PointT> &geometry_handler, const std::string &id = "cloud");
        # bool updatePointCloud[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudGeometryHandler[PointT] &geometry_handler, string &id)
        bool updatePointCloud_GeometryHandler "updatePointCloud" [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudGeometryHandler[PointT] &geometry_handler, string &id)
        
        # brief Updates the XYZ data for an existing cloud object id on screen.
        # param[in] cloud the input point cloud dataset
        # param[in] color_handler the color handler to use
        # param[in] id the point cloud object id to update (default: cloud)
        # return false if no cloud with the specified ID was found
        # template <typename PointT> bool
        # updatePointCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const PointCloudColorHandler<PointT> &color_handler, const std::string &id = "cloud");
        # bool updatePointCloud[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudColorHandler[PointT] &color_handler, const string &id)
        bool updatePointCloud_ColorHandler "updatePointCloud" [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudColorHandler[PointT] &color_handler, const string &id)
        
        ### updatePointCloud Functions ###
        
        ### addPointCloud Functions ###
        # typedef boost::shared_ptr<const PointCloudColorHandler<PointT> > ConstPtr;
        # brief Add a Point Cloud (templated) to screen.
        # param[in] cloud the input point cloud dataset
        # param[in] id the point cloud object id (default: cloud)
        # param viewport the view port where the Point Cloud should be added (default: all)
        # template <typename PointT> bool
        # addPointCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::string &id = "cloud", int viewport = 0);
        bool addPointCloud[PointT] (const shared_ptr[const cpp.PointCloud[PointT]] &cloud, string id, int viewport)
        
        # brief Add a Point Cloud (templated) to screen.
        # param[in] cloud the input point cloud dataset
        # param[in] geometry_handler use a geometry handler object to extract the XYZ data
        # param[in] id the point cloud object id (default: cloud)
        # param[in] viewport the view port where the Point Cloud should be added (default: all)
        # template <typename PointT> bool
        # addPointCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
        #                const PointCloudGeometryHandler<PointT> &geometry_handler,
        #                const std::string &id = "cloud", int viewport = 0);
        # bool addPointCloud[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudGeometryHandler[PointT] &geometry_handler, const string &id, int viewport)
        bool addPointCloud_GeometryHandler "addPointCloud" [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudGeometryHandler[PointT] &geometry_handler, const string &id, int viewport)
        
        # \brief Add a Point Cloud (templated) to screen.
        # Because the geometry handler is given as a pointer, it will be pushed back to the list of available
        # handlers, rather than replacing the current active geometric handler. This makes it possible to
        # switch between different geometric handlers 'on-the-fly' at runtime, from the PCLVisualizer
        # interactor interface (using Alt+0..9).
        # \param[in] cloud the input point cloud dataset
        # \param[in] geometry_handler use a geometry handler object to extract the XYZ data
        # \param[in] id the point cloud object id (default: cloud)
        # \param[in] viewport the view port where the Point Cloud should be added (default: all)
        # template <typename PointT> bool
        # addPointCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const GeometryHandlerConstPtr &geometry_handler, const std::string &id = "cloud", int viewport = 0);
        # set BaseClass(use NG)
        # bool addPointCloud[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const shared_ptr[PointCloudGeometryHandler[PointT]] &geometry_handler, const string &id, int viewport)
        # set InheritanceClass
        # bool addPointCloud [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const shared_ptr[PointCloudGeometryHandlerCustom[PointT]] &geometry_handler, const string &id, int viewport)
        # bool addPointCloud [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const shared_ptr[PointCloudGeometryHandlerSurfaceNormal[PointT]] &geometry_handler, const string &id, int viewport)
        # bool addPointCloud [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const shared_ptr[PointCloudGeometryHandlerXYZ[PointT]] &geometry_handler, const string &id, int viewport)
        bool addPointCloud_GeometryHandler2 "addPointCloud" [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const shared_ptr[PointCloudGeometryHandlerXYZ[PointT]] &geometry_handler, const string &id, int viewport)
        
        # brief Add a Point Cloud (templated) to screen.
        # param[in] cloud the input point cloud dataset
        # param[in] color_handler a specific PointCloud visualizer handler for colors
        # param[in] id the point cloud object id (default: cloud)
        # param[in] viewport the view port where the Point Cloud should be added (default: all)
        # template <typename PointT> bool
        # addPointCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const PointCloudColorHandler<PointT> &color_handler, const std::string &id = "cloud", int viewport = 0);
        # set BaseClass(use NG)
        bool addPointCloud [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudColorHandlerCustom[PointT] &color_handler, const string &id, int viewport)
        # set InheritanceClass
        bool addPointCloud [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudColorHandler[PointT] color_handler, const string &id, int viewport)
        # bool addPointCloud [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudColorHandlerGenericField[PointT] color_handler, const string &id, int viewport)
        # bool addPointCloud [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudColorHandlerHSVField[PointT] color_handler, const string &id, int viewport)
        # bool addPointCloud [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudColorHandlerRandom[PointT] color_handler, const string &id, int viewport)
        # bool addPointCloud [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudColorHandlerRGBField[PointT] color_handler, const string &id, int viewport)
        bool addPointCloud_ColorHandler "addPointCloud" [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudColorHandlerCustom[PointT] &color_handler, const string &id, int viewport)
        
        # brief Add a Point Cloud (templated) to screen.
        # Because the color handler is given as a pointer, it will be pushed back to the list of available
        # handlers, rather than replacing the current active color handler. This makes it possible to
        # switch between different color handlers 'on-the-fly' at runtime, from the PCLVisualizer
        # interactor interface (using 0..9).
        # param[in] cloud the input point cloud dataset
        # param[in] color_handler a specific PointCloud visualizer handler for colors
        # param[in] id the point cloud object id (default: cloud)
        # param[in] viewport the view port where the Point Cloud should be added (default: all)
        # template <typename PointT> bool
        # addPointCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const ColorHandlerConstPtr &color_handler, const std::string &id = "cloud", int viewport = 0);
        # bool addPointCloud[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const shared_ptr[PointCloudColorHandler[PointT]] &color_handler, const string &id, int viewport)
        bool addPointCloud_ColorHandler2 "addPointCloud" [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const shared_ptr[PointCloudColorHandler[PointT]] &color_handler, const string &id, int viewport)
        
        # brief Add a Point Cloud (templated) to screen.
        # param[in] cloud the input point cloud dataset
        # param[in] color_handler a specific PointCloud visualizer handler for colors
        # param[in] geometry_handler use a geometry handler object to extract the XYZ data
        # param[in] id the point cloud object id (default: cloud)
        # param[in] viewport the view port where the Point Cloud should be added (default: all)
        # template <typename PointT> bool
        # addPointCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
        #                const PointCloudColorHandler<PointT> &color_handler,
        #                const PointCloudGeometryHandler<PointT> &geometry_handler,
        #                const std::string &id = "cloud", int viewport = 0);
        # bool addPointCloud [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudColorHandler[PointT] &color_handler, const PointCloudGeometryHandler[PointT] &geometry_handler, const string &id, int viewport)
        bool addPointCloud_ColorGeometryHandler "addPointCloud" [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudColorHandler[PointT] &color_handler, const PointCloudGeometryHandler[PointT] &geometry_handler, const string &id, int viewport)
        
        # brief Add a Point Cloud (templated) to screen.
        # Because the geometry/color handler is given as a pointer, it will be pushed back to the list of
        # available handlers, rather than replacing the current active handler. This makes it possible to
        # switch between different handlers 'on-the-fly' at runtime, from the PCLVisualizer interactor
        # interface (using [Alt+]0..9).
        # param[in] cloud the input point cloud dataset
        # param[in] geometry_handler a specific PointCloud visualizer handler for geometry
        # param[in] color_handler a specific PointCloud visualizer handler for colors
        # param[in] id the point cloud object id (default: cloud)
        # param[in] viewport the view port where the Point Cloud should be added (default: all)
        # template <typename PointT> bool
        # addPointCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
        #                const GeometryHandlerConstPtr &geometry_handler,
        #                const ColorHandlerConstPtr &color_handler,
        #                const std::string &id = "cloud", int viewport = 0);
        # bool addPointCloud[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const shared_ptr[PointCloudGeometryHandler[PointT] &geometry_handler, const shared_ptr[PointCloudColorHandler[PointT]] &color_handler, const string &id, int viewport)
        # bool addPointCloud_ColorGeometryHandler2 "addPointCloud" [PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const shared_ptr[PointCloudGeometryHandler[PointT] &geometry_handler, const shared_ptr[PointCloudColorHandler[PointT]] &color_handler, const string &id, int viewport)
        
        # brief Add a binary blob Point Cloud to screen.
        # Because the geometry/color handler is given as a pointer, it will be pushed back to the list of
        # available handlers, rather than replacing the current active handler. This makes it possible to
        # switch between different handlers 'on-the-fly' at runtime, from the PCLVisualizer interactor
        # interface (using [Alt+]0..9).
        # param[in] cloud the input point cloud dataset
        # param[in] geometry_handler a specific PointCloud visualizer handler for geometry
        # param[in] color_handler a specific PointCloud visualizer handler for colors
        # param[in] sensor_origin the origin of the cloud data in global coordinates (defaults to 0,0,0)
        # param[in] sensor_orientation the orientation of the cloud data in global coordinates (defaults to 1,0,0,0)
        # param[in] id the point cloud object id (default: cloud)
        # param[in] viewport the view port where the Point Cloud should be added (default: all)
        # pcl 1.6.0
        # bool addPointCloud (const sensor_msgs::PointCloud2::ConstPtr &cloud,
        #                const GeometryHandlerConstPtr &geometry_handler,
        #                const ColorHandlerConstPtr &color_handler,
        #                const Eigen::Vector4f& sensor_origin,
        #                const Eigen::Quaternion<float>& sensor_orientation,
        #                const std::string &id = "cloud", int viewport = 0);
        
        # brief Add a binary blob Point Cloud to screen.
        # Because the geometry/color handler is given as a pointer, it will be pushed back to the list of
        # available handlers, rather than replacing the current active handler. This makes it possible to
        # switch between different handlers 'on-the-fly' at runtime, from the PCLVisualizer interactor
        # interface (using [Alt+]0..9).
        # param[in] cloud the input point cloud dataset
        # param[in] geometry_handler a specific PointCloud visualizer handler for geometry
        # param[in] sensor_origin the origin of the cloud data in global coordinates (defaults to 0,0,0)
        # param[in] sensor_orientation the orientation of the cloud data in global coordinates (defaults to 1,0,0,0)
        # param[in] id the point cloud object id (default: cloud)
        # param[in] viewport the view port where the Point Cloud should be added (default: all)
        # pcl 1.6.0
        # bool addPointCloud (const sensor_msgs::PointCloud2::ConstPtr &cloud,
        #                const GeometryHandlerConstPtr &geometry_handler,
        #                const Eigen::Vector4f& sensor_origin,
        #                const Eigen::Quaternion<float>& sensor_orientation,
        #                const std::string &id = "cloud", int viewport = 0);
        
        # brief Add a binary blob Point Cloud to screen.
        # Because the geometry/color handler is given as a pointer, it will be pushed back to the list of
        # available handlers, rather than replacing the current active handler. This makes it possible to
        # switch between different handlers 'on-the-fly' at runtime, from the PCLVisualizer interactor
        # interface (using [Alt+]0..9).
        # param[in] cloud the input point cloud dataset
        # param[in] color_handler a specific PointCloud visualizer handler for colors
        # param[in] sensor_origin the origin of the cloud data in global coordinates (defaults to 0,0,0)
        # param[in] sensor_orientation the orientation of the cloud data in global coordinates (defaults to 1,0,0,0)
        # param[in] id the point cloud object id (default: cloud)
        # param[in] viewport the view port where the Point Cloud should be added (default: all)
        # pcl 1.6.0
        # bool addPointCloud (const sensor_msgs::PointCloud2::ConstPtr &cloud,
        #                const ColorHandlerConstPtr &color_handler,
        #                const Eigen::Vector4f& sensor_origin,
        #                const Eigen::Quaternion<float>& sensor_orientation,
        #                const std::string &id = "cloud", int viewport = 0);
        ### addPointCloud
        
        # /** \brief Add a PolygonMesh object to screen
        #   * \param[in] polymesh the polygonal mesh
        #   * \param[in] id the polygon object id (default: "polygon")
        #   * \param[in] viewport the view port where the PolygonMesh should be added (default: all)
        #   */
        # bool addPolygonMesh (const pcl::PolygonMesh &polymesh, const std::string &id = "polygon", int viewport = 0);
        bool addPolygonMesh (const cpp.PolygonMesh &polymesh, const string &id, int viewport)
        
        # /** \brief Add a PolygonMesh object to screen
        #   * \param[in] cloud the polygonal mesh point cloud
        #   * \param[in] vertices the polygonal mesh vertices
        #   * \param[in] id the polygon object id (default: "polygon")
        #   * \param[in] viewport the view port where the PolygonMesh should be added (default: all)
        #   */
        # template <typename PointT> bool
        # addPolygonMesh (const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
        #                 const std::vector<pcl::Vertices> &vertices,
        #                 const std::string &id = "polygon",
        #                 int viewport = 0);
        bool addPolygonMesh[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const vector[cpp.Vertices] &vertices, const string &id, int viewport)
        
        # /** \brief Update a PolygonMesh object on screen
        #   * \param[in] cloud the polygonal mesh point cloud
        #   * \param[in] vertices the polygonal mesh vertices
        #   * \param[in] id the polygon object id (default: "polygon")
        #   * \return false if no polygonmesh with the specified ID was found
        #   */
        # template <typename PointT> bool
        # updatePolygonMesh (const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
        #                    const std::vector<pcl::Vertices> &vertices,
        #                    const std::string &id = "polygon");
        bool updatePolygonMesh[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const vector[cpp.Vertices] &vertices, const string &id)
        
        # /** \brief Add a Polygonline from a polygonMesh object to screen
        #   * \param[in] polymesh the polygonal mesh from where the polylines will be extracted
        #   * \param[in] id the polygon object id (default: "polygon")
        #   * \param[in] viewport the view port where the PolygonMesh should be added (default: all)
        #   */
        bool addPolylineFromPolygonMesh (const cpp.PolygonMesh &polymesh, const string &id, int viewport)
        
        # /** \brief Add the specified correspondences to the display.
        #   * \param[in] source_points The source points
        #   * \param[in] target_points The target points
        #   * \param[in] correspondences The mapping from source points to target points. Each element must be an index into target_points
        #   * \param[in] id the polygon object id (default: "correspondences")
        #   * \param[in] viewport the view port where the correspondences should be added (default: all)
        #   */
        # template <typename PointT> bool
        # addCorrespondences (const typename pcl::PointCloud<PointT>::ConstPtr &source_points,
        #                     const typename pcl::PointCloud<PointT>::ConstPtr &target_points,
        #                     const std::vector<int> & correspondences,
        #                     const std::string &id = "correspondences",
        #                     int viewport = 0);
        # bool addCorrespondences[PointT](const shared_ptr[cpp.PointCloud[PointT]] &source_points, const shared_ptr[cpp.PointCloud[PointT]] &target_points, const vector[int] & correspondences, const string &id, int viewport)
        
        ### addCorrespondences
        # /** \brief Add the specified correspondences to the display.
        #   * \param[in] source_points The source points
        #   * \param[in] target_points The target points
        #   * \param[in] correspondences The mapping from source points to target points. Each element must be an index into target_points
        #   * \param[in] id the polygon object id (default: "correspondences")
        #   * \param[in] viewport the view port where the correspondences should be added (default: all)
        #   */
        # template <typename PointT> bool
        # addCorrespondences (const typename pcl::PointCloud<PointT>::ConstPtr &source_points,
        #                     const typename pcl::PointCloud<PointT>::ConstPtr &target_points,
        #                     const pcl::Correspondences &correspondences,
        #                     const std::string &id = "correspondences",
        #                     int viewport = 0);
        # bool addCorrespondences[PointT](const shared_ptr[cpp.PointCloud[PointT]] &source_points, const shared_ptr[cpp.PointCloud[PointT]] &target_points, const cpp.Correspondences &correspondences, const string &id, int viewport)
        
        # /** \brief Remove the specified correspondences from the display.
        #   * \param[in] id the polygon correspondences object id (i.e., given on \ref addCorrespondences)
        #   * \param[in] viewport view port from where the correspondences should be removed (default: all)
        #   */
        # inline void removeCorrespondences (const std::string &id = "correspondences", int viewport = 0)
        void removeCorrespondences (const string &id, int viewport)
        
        # /** \brief Get the color handler index of a rendered PointCloud based on its ID
        #   * \param[in] id the point cloud object id
        #   */
        # inline int getColorHandlerIndex (const std::string &id)
        int getColorHandlerIndex (const string &id)
        
        # /** \brief Get the geometry handler index of a rendered PointCloud based on its ID
        #   * \param[in] id the point cloud object id
        #   */
        # inline int getGeometryHandlerIndex (const std::string &id)
        int getGeometryHandlerIndex (const string &id)
        
        # /** \brief Update/set the color index of a renderered PointCloud based on its ID
        #   * \param[in] id the point cloud object id
        #   * \param[in] index the color handler index to use
        #   */
        # bool updateColorHandlerIndex (const std::string &id, int index);
        bool updateColorHandlerIndex (const string &id, int index)
        
        # /** \brief Set the rendering properties of a PointCloud (3x values - e.g., RGB)
        #   * \param[in] property the property type
        #   * \param[in] val1 the first value to be set
        #   * \param[in] val2 the second value to be set
        #   * \param[in] val3 the third value to be set
        #   * \param[in] id the point cloud object id (default: cloud)
        #   * \param[in] viewport the view port where the Point Cloud's rendering properties should be modified (default: all)
        #   */
        # bool setPointCloudRenderingProperties (int property, double val1, double val2, double val3, const std::string &id = "cloud", int viewport = 0);
        bool setPointCloudRenderingProperties (int property, double val1, double val2, double val3, string &id, int viewport)
        
        # /** \brief Set the rendering properties of a PointCloud
        #  * \param[in] property the property type
        #  * \param[in] value the value to be set
        #  * \param[in] id the point cloud object id (default: cloud)
        #  * \param[in] viewport the view port where the Point Cloud's rendering properties should be modified (default: all)
        #  */
        # bool setPointCloudRenderingProperties (int property, double value, const std::string &id = "cloud", int viewport = 0);
        bool setPointCloudRenderingProperties (int property, double value, const string id, int viewport)
        
        # /** \brief Get the rendering properties of a PointCloud
        #  * \param[in] property the property type
        #  * \param[in] value the resultant property value
        #  * \param[in] id the point cloud object id (default: cloud)
        #  */
        # bool getPointCloudRenderingProperties (int property, double &value, const std::string &id = "cloud");
        bool getPointCloudRenderingProperties (int property, double &value, const string &id)
        
        # /** \brief Set the rendering properties of a shape
        #  * \param[in] property the property type
        #  * \param[in] value the value to be set
        #  * \param[in] id the shape object id
        #  * \param[in] viewport the view port where the shape's properties should be modified (default: all)
        #  */
        # bool setShapeRenderingProperties (int property, double value, const std::string &id, int viewport = 0);
        bool setShapeRenderingProperties (int property, double value, const string &id, int viewport)
        
        # /** \brief Set the rendering properties of a shape (3x values - e.g., RGB)
        #   * \param[in] property the property type
        #   * \param[in] val1 the first value to be set
        #   * \param[in] val2 the second value to be set
        #   * \param[in] val3 the third value to be set
        #   * \param[in] id the shape object id
        #   * \param[in] viewport the view port where the shape's properties should be modified (default: all)
        #   */
        # bool setShapeRenderingProperties (int property, double val1, double val2, double val3, const std::string &id, int viewport = 0);
        bool setShapeRenderingProperties (int property, double val1, double val2, double val3, const string &id, int viewport)
        
        bool wasStopped ()
        void resetStoppedFlag ()
        void close ()
        
        # /** \brief Create a new viewport from [xmin,ymin] -> [xmax,ymax].
        #   * \param[in] xmin the minimum X coordinate for the viewport (0.0 <= 1.0)
        #   * \param[in] ymin the minimum Y coordinate for the viewport (0.0 <= 1.0)
        #   * \param[in] xmax the maximum X coordinate for the viewport (0.0 <= 1.0)
        #   * \param[in] ymax the maximum Y coordinate for the viewport (0.0 <= 1.0)
        #   * \param[in] viewport the id of the new viewport
        #   * \note If no renderer for the current window exists, one will be created, and 
        #   * the viewport will be set to 0 ('all'). In case one or multiple renderers do 
        #   * exist, the viewport ID will be set to the total number of renderers - 1.
        # void createViewPort (double xmin, double ymin, double xmax, double ymax, int &viewport);
        void createViewPort (double xmin, double ymin, double xmax, double ymax, int &viewport)
        
        # /** \brief Add a polygon (polyline) that represents the input point cloud (connects all
        #   * points in order)
        #   * \param[in] cloud the point cloud dataset representing the polygon
        #   * \param[in] r the red channel of the color that the polygon should be rendered with
        #   * \param[in] g the green channel of the color that the polygon should be rendered with
        #   * \param[in] b the blue channel of the color that the polygon should be rendered with
        #   * \param[in] id (optional) the polygon id/name (default: "polygon")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        # template <typename PointT> bool
        # addPolygon (const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
        #             double r, double g, double b, const std::string &id = "polygon", int viewport = 0);
        bool addPolygon[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, double r, double g, double b, const string &id, int viewport)
        
        # /** \brief Add a polygon (polyline) that represents the input point cloud (connects all
        #   * points in order)
        #   * \param[in] cloud the point cloud dataset representing the polygon
        #   * \param[in] id the polygon id/name (default: "polygon")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        # template <typename PointT> bool
        # addPolygon (const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
        #             const std::string &id = "polygon", int viewport = 0);
        bool addPolygon[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const string &id, int viewport)
        
        # /** \brief Add a line segment from two points
        #   * \param[in] pt1 the first (start) point on the line
        #   * \param[in] pt2 the second (end) point on the line
        #   * \param[in] id the line id/name (default: "line")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   */
        # template <typename P1, typename P2> bool
        # addLine (const P1 &pt1, const P2 &pt2, const std::string &id = "line", int viewport = 0);
        bool addLine[P1, P2](const P1 &pt1, const P2 &pt2, const string &id, int viewport)
        
        # /** \brief Add a line segment from two points
        #   * \param[in] pt1 the first (start) point on the line
        #   * \param[in] pt2 the second (end) point on the line
        #   * \param[in] r the red channel of the color that the line should be rendered with
        #   * \param[in] g the green channel of the color that the line should be rendered with
        #   * \param[in] b the blue channel of the color that the line should be rendered with
        #   * \param[in] id the line id/name (default: "line")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   */
        # template <typename P1, typename P2> bool
        # addLine (const P1 &pt1, const P2 &pt2, double r, double g, double b, const std::string &id = "line", int viewport = 0);
        bool addLine[P1, P2](const P1 &pt1, const P2 &pt2, double r, double g, double b, const string &id, int viewport)
        
        # /** \brief Add a line arrow segment between two points, and display the distance between them
        #   * \param[in] pt1 the first (start) point on the line
        #   * \param[in] pt2 the second (end) point on the line
        #   * \param[in] r the red channel of the color that the line should be rendered with
        #   * \param[in] g the green channel of the color that the line should be rendered with
        #   * \param[in] b the blue channel of the color that the line should be rendered with
        #   * \param[in] id the arrow id/name (default: "arrow")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   */
        # template <typename P1, typename P2> bool
        # addArrow (const P1 &pt1, const P2 &pt2, double r, double g, double b, const std::string &id = "arrow", int viewport = 0);
        bool addArrow[P1, P2](const P1 &pt1, const P2 &pt2, double r, double g, double b, const string &id, int viewport)
        
        # /** \brief Add a line arrow segment between two points, and display the distance between them
        #   * \param[in] pt1 the first (start) point on the line
        #   * \param[in] pt2 the second (end) point on the line
        #   * \param[in] r the red channel of the color that the line should be rendered with
        #   * \param[in] g the green channel of the color that the line should be rendered with
        #   * \param[in] b the blue channel of the color that the line should be rendered with
        #   * \param[in] display_length true if the length should be displayed on the arrow as text
        #   * \param[in] id the line id/name (default: "arrow")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        # template <typename P1, typename P2> bool
        # addArrow (const P1 &pt1, const P2 &pt2, double r, double g, double b, bool display_length, const std::string &id = "arrow", int viewport = 0);
        bool addArrow[P1, P2](const P1 &pt1, const P2 &pt2, double r, double g, double b, bool display_length, const string &id, int viewport)
        
        # /** \brief Add a sphere shape from a point and a radius
        #   * \param[in] center the center of the sphere
        #   * \param[in] radius the radius of the sphere
        #   * \param[in] id the sphere id/name (default: "sphere")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        # template <typename PointT> bool
        # addSphere (const PointT &center, double radius, const std::string &id = "sphere", int viewport = 0);
        bool addSphere[PointT](const PointT &center, double radius, const string &id, int viewport)
        
        # /** \brief Add a sphere shape from a point and a radius
        #   * \param[in] center the center of the sphere
        #   * \param[in] radius the radius of the sphere
        #   * \param[in] r the red channel of the color that the sphere should be rendered with
        #   * \param[in] g the green channel of the color that the sphere should be rendered with
        #   * \param[in] b the blue channel of the color that the sphere should be rendered with
        #   * \param[in] id the line id/name (default: "sphere")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        # template <typename PointT> bool
        # addSphere (const PointT &center, double radius, double r, double g, double b, const std::string &id = "sphere", int viewport = 0);
        bool addSphere[PointT](const PointT &center, double radius, double r, double g, double b, const string &id, int viewport)
        
        # /** \brief Update an existing sphere shape from a point and a radius
        #   * \param[in] center the center of the sphere
        #   * \param[in] radius the radius of the sphere
        #   * \param[in] r the red channel of the color that the sphere should be rendered with
        #   * \param[in] g the green channel of the color that the sphere should be rendered with
        #   * \param[in] b the blue channel of the color that the sphere should be rendered with
        #   * \param[in] id the sphere id/name (default: "sphere")
        # template <typename PointT> bool
        # updateSphere (const PointT &center, double radius, double r, double g, double b, const std::string &id = "sphere");
        bool updateSphere[PointT](const PointT &center, double radius, double r, double g, double b, const string &id)
        
        #  /** \brief Add a vtkPolydata as a mesh
        #   * \param[in] polydata vtkPolyData
        #   * \param[in] id the model id/name (default: "PolyData")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        # bool addModelFromPolyData (vtkSmartPointer<vtkPolyData> polydata, const std::string & id = "PolyData", int viewport = 0);
        # bool addModelFromPolyData (vtkSmartPointer[vtkPolyData] polydata, const string & id, int viewport)
        
        # /** \brief Add a vtkPolydata as a mesh
        #   * \param[in] polydata vtkPolyData
        #   * \param[in] transform transformation to apply
        #   * \param[in] id the model id/name (default: "PolyData")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        # bool addModelFromPolyData (vtkSmartPointer<vtkPolyData> polydata, vtkSmartPointer<vtkTransform> transform, const std::string &id = "PolyData", int viewport = 0);
        # bool addModelFromPolyData (vtkSmartPointer[vtkPolyData] polydata, vtkSmartPointer[vtkTransform] transform, const string &id, int viewport)
        
        # /** \brief Add a PLYmodel as a mesh
        #   * \param[in] filename of the ply file
        #   * \param[in] id the model id/name (default: "PLYModel")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        # bool addModelFromPLYFile (const std::string &filename, const std::string &id = "PLYModel", int viewport = 0);
        bool addModelFromPLYFile (const string &filename, const string &id, int viewport)
        
        # /** \brief Add a PLYmodel as a mesh and applies given transformation
        #   * \param[in] filename of the ply file
        #   * \param[in] transform transformation to apply
        #   * \param[in] id the model id/name (default: "PLYModel")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        # bool addModelFromPLYFile (const std::string &filename, vtkSmartPointer<vtkTransform> transform, const std::string &id = "PLYModel", int viewport = 0);
        # bool addModelFromPLYFile (const string &filename, vtkSmartPointer[vtkTransform] transform, const string &id, int viewport)
        
        # /** \brief Add a cylinder from a set of given model coefficients
        #   * \param[in] coefficients the model coefficients (point_on_axis, axis_direction, radius)
        #   * \param[in] id the cylinder id/name (default: "cylinder")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   * \code
        #   * // The following are given (or computed using sample consensus techniques)
        #   * // See SampleConsensusModelCylinder for more information.
        #   * // Eigen::Vector3f pt_on_axis, axis_direction;
        #   * // float radius;
        #   * pcl::ModelCoefficients cylinder_coeff;
        #   * cylinder_coeff.values.resize (7);    // We need 7 values
        #   * cylinder_coeff.values[0] = pt_on_axis.x ();
        #   * cylinder_coeff.values[1] = pt_on_axis.y ();
        #   * cylinder_coeff.values[2] = pt_on_axis.z ();
        #   * cylinder_coeff.values[3] = axis_direction.x ();
        #   * cylinder_coeff.values[4] = axis_direction.y ();
        #   * cylinder_coeff.values[5] = axis_direction.z ();
        #   * cylinder_coeff.values[6] = radius;
        #   * addCylinder (cylinder_coeff);
        #   * \endcode
        #   */
        # bool addCylinder (const pcl::ModelCoefficients &coefficients, const std::string &id = "cylinder", int viewport = 0);
        bool addCylinder (const cpp.ModelCoefficients &coefficients, const string &id, int viewport)
        
        # /** \brief Add a sphere from a set of given model coefficients
        #   * \param[in] coefficients the model coefficients (sphere center, radius)
        #   * \param[in] id the sphere id/name (default: "sphere")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   * \code
        #   * // The following are given (or computed using sample consensus techniques)
        #   * // See SampleConsensusModelSphere for more information
        #   * // Eigen::Vector3f sphere_center;
        #   * // float radius;
        #   * pcl::ModelCoefficients sphere_coeff;
        #   * sphere_coeff.values.resize (4);    // We need 4 values
        #   * sphere_coeff.values[0] = sphere_center.x ();
        #   * sphere_coeff.values[1] = sphere_center.y ();
        #   * sphere_coeff.values[2] = sphere_center.z ();
        #   * sphere_coeff.values[3] = radius;
        #   * addSphere (sphere_coeff);
        #   * \endcode
        #   */
        # bool addSphere (const pcl::ModelCoefficients &coefficients, const std::string &id = "sphere", int viewport = 0);
        bool addSphere (const cpp.ModelCoefficients &coefficients, const string &id, int viewport)
        
        # /** \brief Add a line from a set of given model coefficients
        #   * \param[in] coefficients the model coefficients (point_on_line, direction)
        #   * \param[in] id the line id/name (default: "line")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   * \code
        #   * // The following are given (or computed using sample consensus techniques)
        #   * // See SampleConsensusModelLine for more information
        #   * // Eigen::Vector3f point_on_line, line_direction;
        #   * pcl::ModelCoefficients line_coeff;
        #   * line_coeff.values.resize (6);    // We need 6 values
        #   * line_coeff.values[0] = point_on_line.x ();
        #   * line_coeff.values[1] = point_on_line.y ();
        #   * line_coeff.values[2] = point_on_line.z ();
        #   * line_coeff.values[3] = line_direction.x ();
        #   * line_coeff.values[4] = line_direction.y ();
        #   * line_coeff.values[5] = line_direction.z ();
        #   * addLine (line_coeff);
        #   * \endcode
        #   */
        # bool addLine (const pcl::ModelCoefficients &coefficients, const std::string &id = "line", int viewport = 0);
        bool addLine (const cpp.ModelCoefficients &coefficients, const string &id, int viewport)
        
        # /** \brief Add a plane from a set of given model coefficients
        #   * \param[in] coefficients the model coefficients (a, b, c, d with ax+by+cz+d=0)
        #   * \param[in] id the plane id/name (default: "plane")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   * \code
        #   * // The following are given (or computed using sample consensus techniques)
        #   * // See SampleConsensusModelPlane for more information
        #   * // Eigen::Vector4f plane_parameters;
        #   * pcl::ModelCoefficients plane_coeff;
        #   * plane_coeff.values.resize (4);    // We need 4 values
        #   * plane_coeff.values[0] = plane_parameters.x ();
        #   * plane_coeff.values[1] = plane_parameters.y ();
        #   * plane_coeff.values[2] = plane_parameters.z ();
        #   * plane_coeff.values[3] = plane_parameters.w ();
        #   * addPlane (plane_coeff);
        #   * \endcode
        #   */
        # bool addPlane (const pcl::ModelCoefficients &coefficients, const std::string &id = "plane", int viewport = 0);
        bool addPlane (const cpp.ModelCoefficients &coefficients, const string &id, int viewport)
        
        # /** \brief Add a circle from a set of given model coefficients
        #   * \param[in] coefficients the model coefficients (x, y, radius)
        #   * \param[in] id the circle id/name (default: "circle")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   * \code
        #   * // The following are given (or computed using sample consensus techniques)
        #   * // See SampleConsensusModelCircle2D for more information
        #   * // float x, y, radius;
        #   * pcl::ModelCoefficients circle_coeff;
        #   * circle_coeff.values.resize (3);    // We need 3 values
        #   * circle_coeff.values[0] = x;
        #   * circle_coeff.values[1] = y;
        #   * circle_coeff.values[2] = radius;
        #   * vtkSmartPointer<vtkDataSet> data = pcl::visualization::create2DCircle (circle_coeff, z);
        #   * \endcode
        #    */
        # bool addCircle (const pcl::ModelCoefficients &coefficients, const std::string &id = "circle", int viewport = 0);
        bool addCircle (const cpp.ModelCoefficients &coefficients, const string &id, int viewport)
        
        # /** \brief Add a cone from a set of given model coefficients
        #   * \param[in] coefficients the model coefficients (point_on_axis, axis_direction, radiu)
        #   * \param[in] id the cone id/name (default: "cone")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   */
        # bool addCone (const pcl::ModelCoefficients &coefficients, const std::string &id = "cone", int viewport = 0);
        bool addCone (const cpp.ModelCoefficients &coefficients, const string &id, int viewport)
        
        # /** \brief Add a cube from a set of given model coefficients
        #   * \param[in] coefficients the model coefficients (Tx, Ty, Tz, Qx, Qy, Qz, Qw, width, height, depth)
        #   * \param[in] id the cube id/name (default: "cube")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   */
        # bool addCube (const pcl::ModelCoefficients &coefficients, const std::string &id = "cube", int viewport = 0);
        bool addCube (const cpp.ModelCoefficients &coefficients, const string &id, int viewport)
        
        # /** \brief Add a cube from a set of given model coefficients
        #   * \param[in] translation a translation to apply to the cube from 0,0,0
        #   * \param[in] rotation a quaternion-based rotation to apply to the cube
        #   * \param[in] width the cube's width
        #   * \param[in] height the cube's height
        #   * \param[in] depth the cube's depth
        #   * \param[in] id the cube id/name (default: "cube")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   */
        # bool addCube (const Eigen::Vector3f &translation, const Eigen::Quaternionf &rotation, double width, double height, double depth, const std::string &id = "cube", int viewport = 0);
        bool addCube (const eigen3.Vector3f &translation, const eigen3.Quaternionf &rotation, double width, double height, double depth, const string &id, int viewport)
        
        # /** \brief Add a cube from a set of bounding points
        #   * \param[in] x_min is the minimum x value of the box
        #   * \param[in] x_max is the maximum x value of the box
        #   * \param[in] y_min is the minimum y value of the box 
        #   * \param[in] y_max is the maximum y value of the box
        #   * \param[in] z_min is the minimum z value of the box
        #   * \param[in] z_max is the maximum z value of the box
        #   * \param[in] r the red color value (default: 1.0)
        #   * \param[in] g the green color value (default: 1.0)
        #   * \param[in] b the blue color vlaue (default: 1.0)
        #   * \param[in] id the cube id/name (default: "cube")
        #   * \param[in] viewport (optional) the id of the new viewport (default: 0)
        #   */
        # bool
        # addCube (double x_min, double x_max,
        #          double y_min, double y_max,
        #          double z_min, double z_max,
        #          double r = 1.0, double g = 1.0, double b = 1.0,
        #          const std::string &id = "cube",
        #          int viewport = 0);
        bool addCube (double x_min, double x_max, double y_min, double y_max, double z_min, double z_max, double r, double g, double b, const string &id, int viewport)
        
        # /** \brief Changes the visual representation for all actors to surface representation. */
        # void setRepresentationToSurfaceForAllActors ();
        void setRepresentationToSurfaceForAllActors ()
                
        # /** \brief Changes the visual representation for all actors to points representation. */
        # void setRepresentationToPointsForAllActors ();
        void setRepresentationToPointsForAllActors ()
        
        # /** \brief Changes the visual representation for all actors to wireframe representation. */
        # void setRepresentationToWireframeForAllActors ();
        void setRepresentationToWireframeForAllActors ()
        
        # /** \brief Renders a virtual scene as seen from the camera viewpoint and returns the rendered point cloud.
        #   * ATT: This method will only render the scene if only on viewport exists. Otherwise, returns an empty
        #   * point cloud and exits immediately.
        #   * \param[in] xres is the size of the window (X) used to render the scene
        #   * \param[in] yres is the size of the window (Y) used to render the scene
        #   * \param[in] cloud is the rendered point cloud
        #   */
        # void renderView (int xres, int yres, pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud);
        void renderView (int xres, int yres, shared_ptr[cpp.PointCloud[cpp.PointXYZ]] &cloud)
        
        # /** \brief The purpose of this method is to render a CAD model added to the visualizer from different viewpoints
        #   * in order to simulate partial views of model. The viewpoint locations are the vertices of a tesselated sphere
        #   * build from an icosaheadron. The tesselation paremeter controls how many times the triangles of the original
        #   * icosahedron are divided to approximate the sphere and thus the number of partial view generated for a model,
        #   * with a tesselation_level of 0, 12 views are generated if use_vertices=true and 20 views if use_vertices=false
        #   * \param[in] xres the size of the window (X) used to render the partial view of the object
        #   * \param[in] yres the size of the window (Y) used to render the partial view of the object
        #   * \param[in] cloud is a vector of pointcloud with XYZ information that represent the model as seen from the respective viewpoints.
        #   * \param[out] poses represent the transformation from object coordinates to camera coordinates for the respective viewpoint.
        #   * \param[out] enthropies are values between 0 and 1 representing which percentage of the model is seen from the respective viewpoint.
        #   * \param[in] tesselation_level represents the number of subdivisions applied to the triangles of original icosahedron.
        #   * \param[in] view_angle field of view of the virtual camera. Default: 45
        #   * \param[in] radius_sphere the tesselated sphere radius. Default: 1
        #   * \param[in] use_vertices if true, use the vertices of tesselated icosahedron (12,42,...) or if false, use the faces of tesselated
        #   * icosahedron (20,80,...). Default: true
        #   */
        # void renderViewTesselatedSphere (
        #     int xres, int yres,
        #     std::vector<pcl::PointCloud<pcl::PointXYZ>,Eigen::aligned_allocator< pcl::PointCloud<pcl::PointXYZ> > > & cloud,
        #     std::vector<Eigen::Matrix4f,Eigen::aligned_allocator< Eigen::Matrix4f > > & poses, std::vector<float> & enthropies, int tesselation_level,
        #     float view_angle = 45, float radius_sphere = 1, bool use_vertices = true);
        # void renderViewTesselatedSphere (
        #       int xres, int yres,vector[cpp.PointCloud[cpp.PointXYZ]], eigen3.aligned_allocator[cpp.PointCloud[cpp.PointXYZ]]] &cloud,
        #       vector[eigen3.Matrix4f, eigen3.aligned_allocator[eigen3.Matrix4f]] &poses, vector[float] &enthropies, int tesselation_level,
        #       float view_angl, float radius_sphere, bool use_vertices)
        
        # /** \brief Camera view, window position and size. */
        # Camera camera_;
        
        # /** \brief Initialize camera parameters with some default values. */
        # void initCameraParameters ();
        void initCameraParameters()
        
        # /** \brief Search for camera parameters at the command line and set them internally.
        #   * \param[in] argc
        #   * \param[in] argv
        #   */
        # bool getCameraParameters (int argc, char **argv);
        
        # /** \brief Checks whether the camera parameters were manually loaded from file.*/
        # bool cameraParamsSet () const;
        bool cameraParamsSet ()
        
        # /** \brief Update camera parameters and render. */
        # void updateCamera ();
        void updateCamera ()
        
        # /** \brief Reset camera parameters and render. */
        # void resetCamera ();
        void resetCamera ()
        
        # /** \brief Reset the camera direction from {0, 0, 0} to the center_{x, y, z} of a given dataset.
        #   * \param[in] id the point cloud object id (default: cloud)
        #   */
        # void resetCameraViewpoint (const std::string &id = "cloud");
        void resetCameraViewpoint (const string &id)
        
        # /** \brief sets the camera pose given by position, viewpoint and up vector
        #   * \param posX the x co-ordinate of the camera location
        #   * \param posY the y co-ordinate of the camera location
        #   * \param posZ the z co-ordinate of the camera location
        #   * \param viewX the x component of the view upoint of the camera
        #   * \param viewY the y component of the view point of the camera
        #   * \param viewZ the z component of the view point of the camera
        #   * \param upX the x component of the view up direction of the camera
        #   * \param upY the y component of the view up direction of the camera
        #   * \param upZ the y component of the view up direction of the camera
        #   * \param viewport the viewport to modify camera of, if 0, modifies all cameras
        # void setCameraPose (double posX, double posY, double posZ, double viewX, double viewY, double viewZ, double upX, double upY, double upZ, int viewport = 0);
        void setCameraPose (double posX, double posY, double posZ, double viewX, double viewY, double viewZ, double upX, double upY, double upZ, int viewport)
        
        # /** \brief Set the camera location and viewup according to the given arguments
        #   * \param[in] posX the x co-ordinate of the camera location
        #   * \param[in] posY the y co-ordinate of the camera location
        #   * \param[in] posZ the z co-ordinate of the camera location
        #   * \param[in] viewX the x component of the view up direction of the camera
        #   * \param[in] viewY the y component of the view up direction of the camera
        #   * \param[in] viewZ the z component of the view up direction of the camera
        #   * \param[in] viewport the viewport to modify camera of, if 0, modifies all cameras
        # void setCameraPosition (double posX,double posY, double posZ, double viewX, double viewY, double viewZ, int viewport = 0);
        void setCameraPosition (double posX,double posY, double posZ, double viewX, double viewY, double viewZ, int viewport)
        
        # /** \brief Get the current camera parameters. */
        # void getCameras (std::vector<Camera>& cameras);
        # void getCameras (vector[Camera]& cameras)
        
        # /** \brief Get the current viewing pose. */
        # Eigen::Affine3f getViewerPose ();
        eigen3.Affine3f getViewerPose ()
        
        # /** \brief Save the current rendered image to disk, as a PNG screenshot.
        #   * \param[in] file the name of the PNG file
        # void saveScreenshot (const std::string &file);
        void saveScreenshot (const string &file)
        
        # /** \brief Return a pointer to the underlying VTK Render Window used. */
        # vtkSmartPointer<vtkRenderWindow> getRenderWindow ()
        
        # /** \brief Create the internal Interactor object. */
        # void createInteractor ();
        void createInteractor ()
        
        # /** \brief Set up our unique PCL interactor style for a given vtkRenderWindowInteractor object
        #   * attached to a given vtkRenderWindow
        #   * \param[in,out] iren the vtkRenderWindowInteractor object to set up
        #   * \param[in,out] win a vtkRenderWindow object that the interactor is attached to
        # void setupInteractor (vtkRenderWindowInteractor *iren, vtkRenderWindow *win);
        
        # /** \brief Get a pointer to the current interactor style used. */
        # inline vtkSmartPointer<PCLVisualizerInteractorStyle> getInteractorStyle ()


# ctypedef PCLVisualizer PCLVisualizer_t
ctypedef shared_ptr[PCLVisualizer] PCLVisualizerPtr_t
###

# cloud_viewer.h
cdef extern from "pcl/visualization/cloud_viewer.h" namespace "pcl::visualization" nogil:
    cdef cppclass CloudViewer:
        # CloudViewer ()
        CloudViewer (string& window_name)
        # public:
        # /** \brief Show a cloud, with an optional key for multiple clouds.
        # * \param[in] cloud RGB point cloud
        # * \param[in] cloudname a key for the point cloud, use the same name if you would like to overwrite the existing cloud.
        # void showCloud (const ColorCloud::ConstPtr &cloud, const std::string& cloudname = "cloud");
        void showCloud (cpp.PointCloud_PointXYZRGB_Ptr_t cloud, const string cloudname)
        
        # /** \brief Show a cloud, with an optional key for multiple clouds.
        #  * \param[in] cloud RGBA point cloud
        #  * \param[in] cloudname a key for the point cloud, use the same name if you would like to overwrite the existing cloud.
        # void showCloud (const ColorACloud::ConstPtr &cloud, const std::string& cloudname = "cloud");
        void showCloud (cpp.PointCloud_PointXYZRGBA_Ptr_t cloud, const string cloudname)
        
        # /** \brief Show a cloud, with an optional key for multiple clouds.
        #   * \param[in] cloud XYZI point cloud
        #   * \param[in] cloudname a key for the point cloud, use the same name if you would like to overwrite the existing cloud.
        void showCloud (cpp.PointCloud_PointXYZI_Ptr_t cloud, string cloudname)
        
        # /** \brief Show a cloud, with an optional key for multiple clouds.
        #   * \param[in] cloud XYZ point cloud
        #   * \param[in] cloudname a key for the point cloud, use the same name if you would like to overwrite the existing cloud.
        void showCloud (cpp.PointCloudPtr_t cloud, string cloudname)
        
        # /** \brief Check if the gui was quit, you should quit also
        #  * \param millis_to_wait This will request to "spin" for the number of milliseconds, before exiting.
        #  * \return true if the user signaled the gui to stop
        bool wasStopped (int millis_to_wait)
        
        # /** Visualization callable function, may be used for running things on the UI thread.
        # ctypedef boost::function1<void, pcl::visualization::PCLVisualizer&> VizCallable;
        
        # /** \brief Run a callbable object on the UI thread. Will persist until removed
        #  * @param x Use boost::ref(x) for a function object that you would like to not copy
        #  * \param key The key for the callable -- use the same key to overwrite.
        # void runOnVisualizationThread (VizCallable x, const std::string& key = "callable");
        
        # /** \brief Run a callbable object on the UI thread. This will run once and be removed
        #  * @param x Use boost::ref(x) for a function object that you would like to not copy
        # void runOnVisualizationThreadOnce (VizCallable x);
        
        # /** \brief Remove a previously added callable object, NOP if it doesn't exist.
        #  * @param key the key that was registered with the callable object.
        # void removeVisualizationCallable (string& key = "callable")
        
        # /** \brief Register a callback function for keyboard events
        #   * \param[in] callback  the function that will be registered as a callback for a keyboard event
        #   * \param[in] cookie    user data that is passed to the callback
        #   * \return              connection object that allows to disconnect the callback function.
        # inline boost::signals2::connection registerKeyboardCallback (void (*callback) (const pcl::visualization::KeyboardEvent&, void*), void* cookie = NULL)
        
        # /** \brief Register a callback function for keyboard events
        #   * \param[in] callback  the member function that will be registered as a callback for a keyboard event
        #   * \param[in] instance  instance to the class that implements the callback function
        #   * \param[in] cookie    user data that is passed to the callback
        #   * \return              connection object that allows to disconnect the callback function.
        # template<typename T> inline boost::signals2::connection registerKeyboardCallback (void (T::*callback) (const pcl::visualization::KeyboardEvent&, void*), T& instance, void* cookie = NULL)
        
        # /** \brief Register a callback function for mouse events
        #   * \param[in] callback  the function that will be registered as a callback for a mouse event
        #   * \param[in] cookie    user data that is passed to the callback
        #   * \return              connection object that allows to disconnect the callback function.
        # inline boost::signals2::connection registerMouseCallback (void (*callback) (const pcl::visualization::MouseEvent&, void*), void* cookie = NULL)
        
        # /** \brief Register a callback function for mouse events
        #   * \param[in] callback  the member function that will be registered as a callback for a mouse event
        #   * \param[in] instance  instance to the class that implements the callback function
        #   * \param[in] cookie    user data that is passed to the callback
        #   * \return              connection object that allows to disconnect the callback function.
        # template<typename T> inline boost::signals2::connection registerMouseCallback (void (T::*callback) (const pcl::visualization::MouseEvent&, void*), T& instance, void* cookie = NULL)
        
        # /** \brief Register a callback function for point picking events
        #   * \param[in] callback  the function that will be registered as a callback for a point picking event
        #   * \param[in] cookie    user data that is passed to the callback
        #   * \return              connection object that allows to disconnect the callback function.
        # inline boost::signals2::connection  registerPointPickingCallback (void (*callback) (const pcl::visualization::PointPickingEvent&, void*), void* cookie = NULL)
        
        # /** \brief Register a callback function for point picking events
        #   * \param[in] callback  the member function that will be registered as a callback for a point picking event
        #   * \param[in] instance  instance to the class that implements the callback function
        #   * \param[in] cookie    user data that is passed to the callback
        #   * \return              connection object that allows to disconnect the callback function.
        #   */
        # template<typename T> inline boost::signals2::connection  registerPointPickingCallback (void (T::*callback) (const pcl::visualization::PointPickingEvent&, void*), T& instance, void* cookie = NULL)


# ctypedef CloudViewer CloudViewer_t
ctypedef shared_ptr[CloudViewer] CloudViewerPtr_t
# ctypedef boost::function1<void, pcl::visualization::PCLVisualizer&> VizCallable;
# ctypedef function1[void, PCLVisualizer] VizCallable;
###

# histogram_visualizer.h
cdef extern from "pcl/visualization/histogram_visualizer.h" namespace "pcl::visualization":
    cdef cppclass PCLHistogramVisualizer:
        PCLHistogramVisualizer ()
        
        # brief Spin once method. Calls the interactor and updates the screen once. 
        # void spinOnce (int time = 1, bool force_redraw = false)
        void spinOnce ()
        # void spinOnce (int time, bool force_redraw)
        
        # brief Spin method. Calls the interactor and runs an internal loop. */
        void spin ()
        
        # brief Set the viewport's background color.
        # param[in] r the red component of the RGB color
        # param[in] g the green component of the RGB color
        # param[in] b the blue component of the RGB color
        # param[in] viewport the view port (default: all)
        # void setBackgroundColor (const double &r, const double &g, const double &b, int viewport = 0)
        void setBackgroundColor (const double &r, const double &g, const double &b, int viewport)
        
        # brief Add a histogram feature to screen as a separate window, from a cloud containing a single histogram.
        # param[in] cloud the PointCloud dataset containing the histogram
        # param[in] hsize the length of the histogram
        # param[in] id the point cloud object id (default: cloud)
        # param[in] win_width the width of the window
        # param[in] win_height the height of the window
        # template <typename PointT> bool 
        # addFeatureHistogram (const pcl::PointCloud<PointT> &cloud, int hsize, const std::string &id = "cloud", int win_width = 640, int win_height = 200);
        bool addFeatureHistogram[PointT](const cpp.PointCloud[PointT] &cloud, int hsize, string cloudname, int win_width, int win_height)
        
        # brief Add a histogram feature to screen as a separate window from a cloud containing a single histogram.
        # param[in] cloud the PointCloud dataset containing the histogram
        # param[in] field_name the field name containing the histogram
        # param[in] id the point cloud object id (default: cloud)
        # param[in] win_width the width of the window
        # param[in] win_height the height of the window
        # bool addFeatureHistogram (const sensor_msgs::PointCloud2 &cloud,  const std::string &field_name, const std::string &id = "cloud", int win_width = 640, int win_height = 200);
        
        # /** \brief Add a histogram feature to screen as a separate window.
        #   * \param[in] cloud the PointCloud dataset containing the histogram
        #   * \param[in] field_name the field name containing the histogram
        #   * \param[in] index the point index to extract the histogram from
        #   * \param[in] id the point cloud object id (default: cloud)
        #   * \param[in] win_width the width of the window
        #   * \param[in] win_height the height of the window 
        # template <typename PointT> bool 
        # addFeatureHistogram (const pcl::PointCloud<PointT> &cloud, 
        #                      const std::string &field_name, 
        #                      const int index,
        #                      const std::string &id = "cloud", int win_width = 640, int win_height = 200);
        # Override before addFeatureHistogram Function
        # bool addFeatureHistogram[PointT](const cpp.PointCloud[PointT] &cloud, string field_name, int index, string id, int win_width, int win_height)
        
        # /** \brief Add a histogram feature to screen as a separate window.
        #   * \param[in] cloud the PointCloud dataset containing the histogram
        #   * \param[in] field_name the field name containing the histogram
        #   * \param[in] index the point index to extract the histogram from
        #   * \param[in] id the point cloud object id (default: cloud)
        #   * \param[in] win_width the width of the window
        #   * \param[in] win_height the height of the window
        # bool 
        # addFeatureHistogram (const sensor_msgs::PointCloud2 &cloud, 
        #                      const std::string &field_name, 
        #                      const int index,
        #                      const std::string &id = "cloud", int win_width = 640, int win_height = 200);
        
        # /** \brief Update a histogram feature that is already on screen, with a cloud containing a single histogram.
        #   * \param[in] cloud the PointCloud dataset containing the histogram
        #   * \param[in] hsize the length of the histogram
        #   * \param[in] id the point cloud object id (default: cloud)
        # template <typename PointT> bool updateFeatureHistogram (const pcl::PointCloud<PointT> &cloud, int hsize, const std::string &id = "cloud");
        bool updateFeatureHistogram[PointT](const cpp.PointCloud[PointT] &cloud, int hsize, const string &id)
        
        # /** \brief Update a histogram feature that is already on screen, with a cloud containing a single histogram.
        #   * \param[in] cloud the PointCloud dataset containing the histogram
        #   * \param[in] field_name the field name containing the histogram
        #   * \param[in] id the point cloud object id (default: cloud)
        # bool updateFeatureHistogram (const sensor_msgs::PointCloud2 &cloud, const std::string &field_name, const std::string &id = "cloud");
        
        # /** \brief Update a histogram feature that is already on screen, with a cloud containing a single histogram.
        #   * \param[in] cloud the PointCloud dataset containing the histogram
        #   * \param[in] field_name the field name containing the histogram
        #   * \param[in] index the point index to extract the histogram from
        #   * \param[in] id the point cloud object id (default: cloud)
        # template <typename PointT> bool 
        # updateFeatureHistogram (const pcl::PointCloud<PointT> &cloud, const std::string &field_name, const int index, const std::string &id = "cloud");
        bool updateFeatureHistogram[PointT](const cpp.PointCloud[PointT]  &cloud, const string &field_name, const int index, const string &id)
        
        # /** \brief Update a histogram feature that is already on screen, with a cloud containing a single histogram.
        #   * \param[in] cloud the PointCloud dataset containing the histogram
        #   * \param[in] field_name the field name containing the histogram
        #   * \param[in] index the point index to extract the histogram from
        #   * \param[in] id the point cloud object id (default: cloud)
        # bool updateFeatureHistogram (const sensor_msgs::PointCloud2 &cloud, const std::string &field_name, const int index, const std::string &id = "cloud");
        
        # /** \brief Set the Y range to minp-maxp for all histograms.
        #    * \param[in] minp the minimum Y range
        #    * \param[in] maxp the maximum Y range
        # void setGlobalYRange (float minp, float maxp);
        void setGlobalYRange (float minp, float maxp)
        
        # /** \brief Update all window positions on screen so that they fit. */
        # void updateWindowPositions ();
        void updateWindowPositions ()
        
        # #if ((VTK_MAJOR_VERSION) == 5 && (VTK_MINOR_VERSION <= 4))
        # /** \brief Returns true when the user tried to close the window */
        # bool wasStopped ();
        # /** \brief Set the stopped flag back to false */
        # void resetStoppedFlag ();
        # #endif

# ctypedef CloudViewer CloudViewer_t
ctypedef shared_ptr[PCLHistogramVisualizer] PCLHistogramVisualizerPtr_t
###

# image_viewer.h
# class PCL_EXPORTS ImageViewer
cdef extern from "pcl/visualization/image_viewer.h" namespace "pcl::visualization" nogil:
    cdef cppclass ImageViewer:
        ImageViewer()
        ImageViewer(const string& window_title)
        # ImageViewer()
        # ImageViewer (const std::string& window_title = "");
        
        # public:
        # /** \brief Show a monochrome 2D image on screen.
        #   * \param[in] data the input data representing the image
        #   * \param[in] width the width of the image
        #   * \param[in] height the height of the image
        #   * \param[in] layer_id the name of the layer (default: "image")
        #   * \param[in] opacity the opacity of the layer (default: 1.0)
        #   */
        # void  showMonoImage (const unsigned char* data, unsigned width, unsigned height, const std::string &layer_id = "mono_image", double opacity = 1.0);
        void showMonoImage (const unsigned char* data, unsigned width, unsigned height,const string &layer_id, double opacity)
        
        # brief Add a monochrome 2D image layer, but do not render it (use spin/spinOnce to update).
        # param[in] data the input data representing the image
        # param[in] width the width of the image
        # param[in] height the height of the image
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void addMonoImage (const unsigned char* data, unsigned width, unsigned height, const std::string &layer_id = "mono_image", double opacity = 1.0)
        void addMonoImage (const unsigned char* data, unsigned width, unsigned height, const string &layer_id, double opacity)
        
        # brief Show a 2D RGB image on screen.
        # param[in] data the input data representing the image
        # param[in] width the width of the image
        # param[in] height the height of the image
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void showRGBImage (const unsigned char* data, unsigned width, unsigned height, 
        #               const std::string &layer_id = "rgb_image", double opacity = 1.0);
        void showRGBImage (const unsigned char* data, unsigned width, unsigned height, const string &layer_id, double opacity)
        
        # brief Add an RGB 2D image layer, but do not render it (use spin/spinOnce to update).
        # param[in] data the input data representing the image
        # param[in] width the width of the image
        # param[in] height the height of the image
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void addRGBImage (const unsigned char* data, unsigned width, unsigned height, 
        #              const std::string &layer_id = "rgb_image", double opacity = 1.0);
        void addRGBImage (const unsigned char* data, unsigned width, unsigned height, const string &layer_id, double opacity)
        
        # brief Show a 2D image on screen, obtained from the RGB channel of a point cloud.
        # param[in] data the input data representing the RGB point cloud 
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # template <typename T> inline void 
        # showRGBImage (const typename pcl::PointCloud<T>::ConstPtr &cloud,
        #               const std::string &layer_id = "rgb_image", double opacity = 1.0)
        # void showRGBImage (const shared_ptr[cpp.PointCloud[PointT]] &cloud, const string &layer_id, double opacity)
        
        # brief Add an RGB 2D image layer, but do not render it (use spin/spinOnce to update).
        # param[in] data the input data representing the RGB point cloud 
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # template <typename T> inline void 
        # addRGBImage (const typename pcl::PointCloud<T>::ConstPtr &cloud, const std::string &layer_id = "rgb_image", double opacity = 1.0)
        # void addRGBImage[T](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const string &layer_id, double opacity)
        
        # brief Show a 2D image on screen, obtained from the RGB channel of a point cloud.
        # param[in] data the input data representing the RGB point cloud 
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # template <typename T> void 
        # showRGBImage (const pcl::PointCloud<T> &cloud, const std::string &layer_id = "rgb_image", double opacity = 1.0);
        # void showRGBImage[T](const cpp.PointCloud[T] &cloud, const string &layer_id, double opacity)
        
        # brief Add an RGB 2D image layer, but do not render it (use spin/spinOnce to update).
        # param[in] data the input data representing the RGB point cloud 
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # template <typename T> void 
        # addRGBImage (const pcl::PointCloud<T> &cloud, const std::string &layer_id = "rgb_image", double opacity = 1.0);
        # void addRGBImage (const cpp.PointCloud[T] &cloud, const string &layer_id, double opacity)
        
        # brief Show a 2D image (float) on screen.
        # param[in] data the input data representing the image in float format
        # param[in] width the width of the image
        # param[in] height the height of the image
        # param[in] min_value filter all values in the image to be larger than this minimum value
        # param[in] max_value filter all values in the image to be smaller than this maximum value
        # param[in] grayscale show data as grayscale (true) or not (false). Default: false
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void showFloatImage (const float* data, unsigned int width, unsigned int height, 
        #                 float min_value = std::numeric_limits<float>::min (), 
        #                 float max_value = std::numeric_limits<float>::max (), bool grayscale = false,
        #                 const std::string &layer_id = "float_image", double opacity = 1.0);
        void showFloatImage (
                        const float* data,
                        unsigned int width,
                        unsigned int height, 
                        float min_value, 
                        float max_value,
                        bool grayscale,
                        const string &layer_id,
                        double opacity)
        
        # brief Add a float 2D image layer, but do not render it (use spin/spinOnce to update).
        # param[in] data the input data representing the image in float format
        # param[in] width the width of the image
        # param[in] height the height of the image
        # param[in] min_value filter all values in the image to be larger than this minimum value
        # param[in] max_value filter all values in the image to be smaller than this maximum value
        # param[in] grayscale show data as grayscale (true) or not (false). Default: false
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void addFloatImage (const float* data, unsigned int width, unsigned int height, 
        #                float min_value = std::numeric_limits<float>::min (), 
        #                float max_value = std::numeric_limits<float>::max (), bool grayscale = false,
        #                const std::string &layer_id = "float_image", double opacity = 1.0);
        void addFloatImage (
                        const float* data,
                        unsigned int width,
                        unsigned int height,
                        float min_value, 
                        float max_value,
                        bool grayscale,
                        const string &layer_id,
                        double opacity)
        
        
        # brief Show a 2D image (unsigned short) on screen.
        # param[in] short_image the input data representing the image in unsigned short format
        # param[in] width the width of the image
        # param[in] height the height of the image
        # param[in] min_value filter all values in the image to be larger than this minimum value
        # param[in] max_value filter all values in the image to be smaller than this maximum value
        # param[in] grayscale show data as grayscale (true) or not (false). Default: false
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void
        # showShortImage (const unsigned short* short_image, unsigned int width, unsigned int height, 
        #                 unsigned short min_value = std::numeric_limits<unsigned short>::min (), 
        #                 unsigned short max_value = std::numeric_limits<unsigned short>::max (), bool grayscale = false,
        #                 const std::string &layer_id = "short_image", double opacity = 1.0);
        # void showShortImage (
        #                       const unsigned short* short_image, 
        #                       unsigned int width, 
        #                       unsigned int height, 
        #                       unsigned short min_value, 
        #                       unsigned short max_value, 
        #                       bool grayscale = false,
        #                       const string &layer_id,
        #                       double opacity)
        
        # brief Add a short 2D image layer, but do not render it (use spin/spinOnce to update).
        # param[in] short_image the input data representing the image in unsigned short format
        # param[in] width the width of the image
        # param[in] height the height of the image
        # param[in] min_value filter all values in the image to be larger than this minimum value
        # param[in] max_value filter all values in the image to be smaller than this maximum value
        # param[in] grayscale show data as grayscale (true) or not (false). Default: false
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void
        # addShortImage (const unsigned short* short_image, unsigned int width, unsigned int height, 
        #                unsigned short min_value = std::numeric_limits<unsigned short>::min (), 
        #                unsigned short max_value = std::numeric_limits<unsigned short>::max (), bool grayscale = false,
        #                const std::string &layer_id = "short_image", double opacity = 1.0);
        void addShortImage (
                            const unsigned short* short_image, 
                            unsigned int width, unsigned int height, 
                            unsigned short min_value, unsigned short max_value,
                            bool grayscale,
                            const string &layer_id, double opacity)
        
        # brief Show a 2D image on screen representing angle data.
        # param[in] data the input data representing the image
        # param[in] width the width of the image
        # param[in] height the height of the image
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void showAngleImage (const float* data, unsigned width, unsigned height, const std::string &layer_id = "angle_image", double opacity = 1.0);
        void showAngleImage (const float* data, unsigned width, unsigned height, const string &layer_id, double opacity)
        
        # brief Add an angle 2D image layer, but do not render it (use spin/spinOnce to update).
        # param[in] data the input data representing the image
        # param[in] width the width of the image
        # param[in] height the height of the image
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void addAngleImage (const float* data, unsigned width, unsigned height, const std::string &layer_id = "angle_image", double opacity = 1.0);
        void addAngleImage (const float* data, unsigned width, unsigned height, const string &layer_id, double opacity)
        
        # brief Show a 2D image on screen representing half angle data.
        # param[in] data the input data representing the image
        # param[in] width the width of the image
        # param[in] height the height of the image
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void showHalfAngleImage (const float* data, unsigned width, unsigned height, const std::string &layer_id = "half_angle_image", double opacity = 1.0);
        void showHalfAngleImage (const float* data, unsigned width, unsigned height, const string &layer_id, double opacity)
        
        # brief Add a half angle 2D image layer, but do not render it (use spin/spinOnce to update).
        # param[in] data the input data representing the image
        # param[in] width the width of the image
        # param[in] height the height of the image
        # param[in] layer_id the name of the layer (default: "image")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void addHalfAngleImage (const float* data, unsigned width, unsigned height,
        #                         const std::string &layer_id = "half_angle_image", double opacity = 1.0);
        void addHalfAngleImage (const float* data, unsigned width, unsigned height, const string &layer_id, double opacity)
        
        # brief Sets the pixel at coordinates(u,v) to color while setting the neighborhood to another
        # param[in] u the u/x coordinate of the pixel
        # param[in] v the v/y coordinate of the pixel
        # param[in] fg_color the pixel color
        # param[in] bg_color the neighborhood color
        # param[in] radius the circle radius around the pixel
        # param[in] layer_id the name of the layer (default: "points")
        # param[in] opacity the opacity of the layer (default: 1.0)
        # void markPoint (size_t u, size_t v, Vector3ub fg_color, Vector3ub bg_color = red_color, double radius = 3.0,
        #                 const std::string &layer_id = "points", double opacity = 1.0);
        # Vector3ub Unknown
        # void markPoint (size_t u, size_t v, Vector3ub fg_color, Vector3ub bg_color, double radius, const string &layer_id, double opacity)
        
        # brief Set the window title name
        # param[in] name the window title
        # void setWindowTitle (const std::string& name)
        void setWindowTitle (const string& name)
        
        # brief Spin method. Calls the interactor and runs an internal loop. */
        # void spin ();
        void spin ()
        
        # brief Spin once method. Calls the interactor and updates the screen once. 
        # param[in] time - How long (in ms) should the visualization loop be allowed to run.
        # param[in] force_redraw - if false it might return without doing anything if the 
        # interactor's framerate does not require a redraw yet.
        # void spinOnce (int time = 1, bool force_redraw = true);
        void spinOnce (int time, bool force_redraw)
        
        # brief Register a callback function for keyboard events
        # param[in] callback  the function that will be registered as a callback for a keyboard event
        # param[in] cookie    user data that is passed to the callback
        # return a connection object that allows to disconnect the callback function.
        # boost::signals2::connection 
        # registerKeyboardCallback (void (*callback) (const pcl::visualization::KeyboardEvent&, void*), 
        #                           void* cookie = NULL)
        
        # brief Register a callback function for keyboard events
        # param[in] callback  the member function that will be registered as a callback for a keyboard event
        # param[in] instance  instance to the class that implements the callback function
        # param[in] cookie    user data that is passed to the callback
        # return a connection object that allows to disconnect the callback function.
        # template<typename T> boost::signals2::connection 
        # registerKeyboardCallback (void (T::*callback) (const pcl::visualization::KeyboardEvent&, void*), 
        #                           T& instance, void* cookie = NULL)
        
        # brief Register a callback boost::function for keyboard events
        # param[in] cb the boost function that will be registered as a callback for a keyboard event
        # return a connection object that allows to disconnect the callback function.
        # boost::signals2::connection 
        # registerKeyboardCallback (boost::function<void (const pcl::visualization::KeyboardEvent&)> cb);
        
        # brief Register a callback boost::function for mouse events
        # param[in] callback  the function that will be registered as a callback for a mouse event
        # param[in] cookie    user data that is passed to the callback
        # return a connection object that allows to disconnect the callback function.
        # boost::signals2::connection 
        # registerMouseCallback (void (*callback) (const pcl::visualization::MouseEvent&, void*), 
        #                        void* cookie = NULL)
        
        # brief Register a callback function for mouse events
        # param[in] callback  the member function that will be registered as a callback for a mouse event
        # param[in] instance  instance to the class that implements the callback function
        # param[in] cookie    user data that is passed to the callback
        # return a connection object that allows to disconnect the callback function.
        # template<typename T> boost::signals2::connection 
        # registerMouseCallback(void (T::*callback) (const pcl::visualization::MouseEvent&, void*), 
        #                        T& instance, void* cookie = NULL)
        # boost::signals2::connection registerMouseCallback[T](void (T::*callback) (const pcl::visualization::MouseEvent&, void*),  T& instance, void* cookie = NULL)
        
        # brief Register a callback function for mouse events
        # param[in] cb the boost function that will be registered as a callback for a mouse event
        # return a connection object that allows to disconnect the callback function.
        # boost::signals2::connection 
        # registerMouseCallback (boost::function<void (const pcl::visualization::MouseEvent&)> cb);
        
        # brief Set the position in screen coordinates.
        # param[in] x where to move the window to (X)
        # param[in] y where to move the window to (Y)
        # void setPosition (int x, int y)
        void setPosition (int x, int y)
        
        # brief Set the window size in screen coordinates.
        # param[in] xw window size in horizontal (pixels)
        # param[in] yw window size in vertical (pixels)
        # void setSize (int xw, int yw)
        void setSize (int xw, int yw)
        
        # brief Returns true when the user tried to close the window
        # bool wasStopped () const
        bool wasStopped ()
        
        # brief Add a circle shape from a point and a radius
        # param[in] x the x coordinate of the circle center
        # param[in] y the y coordinate of the circle center
        # param[in] radius the radius of the circle
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # bool
        # addCircle (unsigned int x, unsigned int y, double radius, const std::string &layer_id = "circles", double opacity = 1.0);
        bool addCircle (unsigned int x, unsigned int y, double radius, const string &layer_id, double opacity)
        
        # brief Add a circle shape from a point and a radius
        # param[in] x the x coordinate of the circle center
        # param[in] y the y coordinate of the circle center
        # param[in] radius the radius of the circle
        # param[in] r the red channel of the color that the sphere should be rendered with (0.0 -> 1.0)
        # param[in] g the green channel of the color that the sphere should be rendered with (0.0 -> 1.0)
        # param[in] b the blue channel of the color that the sphere should be rendered with (0.0 -> 1.0)
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # bool
        # addCircle (unsigned int x, unsigned int y, double radius, 
        #            double r, double g, double b,
        #            const std::string &layer_id = "circles", double opacity = 1.0);
        bool addCircle (unsigned int x, unsigned int y, double radius, double r, double g, double b, const string &layer_id, double opacity)
        
        # brief Add a 2D box and color its edges with a given color
        # param[in] min_pt the X,Y min coordinate
        # param[in] max_pt the X,Y max coordinate
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # bool
        # addRectangle (const pcl::PointXY &min_pt, const pcl::PointXY &max_pt,
        #               const std::string &layer_id = "rectangles", double opacity = 1.0);
        # bool addRectangle (const pcl::PointXY &min_pt, const pcl::PointXY &max_pt, const string &layer_id, double opacity)
        
        # brief Add a 2D box and color its edges with a given color
        # param[in] min_pt the X,Y min coordinate
        # param[in] max_pt the X,Y max coordinate
        # param[in] r the red channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] g the green channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] b the blue channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # bool
        # addRectangle (const pcl::PointXY &min_pt, const pcl::PointXY &max_pt,
        #               double r, double g, double b,
        #               const std::string &layer_id = "rectangles", double opacity = 1.0);
        # bool addRectangle (const pcl::PointXY &min_pt, const pcl::PointXY &max_pt, double r, double g, double b, const string &layer_id, double opacity)
        
        # brief Add a 2D box and color its edges with a given color
        # param[in] x_min the X min coordinate
        # param[in] x_max the X max coordinate
        # param[in] y_min the Y min coordinate
        # param[in] y_max the Y max coordinate 
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # bool
        # addRectangle (unsigned int x_min, unsigned int x_max, unsigned int y_min, unsigned int y_max,  
        #               const std::string &layer_id = "rectangles", double opacity = 1.0);
        # bool addRectangle (unsigned int x_min, unsigned int x_max, unsigned int y_min, unsigned int y_max, const string &layer_id, double opacity)
        
        # brief Add a 2D box and color its edges with a given color
        # param[in] x_min the X min coordinate
        # param[in] x_max the X max coordinate
        # param[in] y_min the Y min coordinate
        # param[in] y_max the Y max coordinate 
        # param[in] r the red channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] g the green channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] b the blue channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # bool
        # addRectangle (unsigned int x_min, unsigned int x_max, unsigned int y_min, unsigned int y_max,  
        #               double r, double g, double b,
        #               const std::string &layer_id = "rectangles", double opacity = 1.0);
        # bool addRectangle (unsigned int x_min, unsigned int x_max, unsigned int y_min, unsigned int y_max, double r, double g, double b, const string &layer_id, double opacity)
        
        # brief Add a 2D box and color its edges with a given color
        # param[in] image the organized point cloud dataset containing the image data
        # param[in] min_pt the X,Y min coordinate
        # param[in] max_pt the X,Y max coordinate
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # template <typename T> bool
        # addRectangle (const typename pcl::PointCloud<T>::ConstPtr &image, 
        #               const T &min_pt, const T &max_pt,
        #               const std::string &layer_id = "rectangles", double opacity = 1.0);
        # bool addRectangle (const shared_ptr[cpp.PointCloud[T]] &image, const T &min_pt, const T &max_pt, const string &layer_id, double opacity)
        
        # brief Add a 2D box and color its edges with a given color
        # param[in] image the organized point cloud dataset containing the image data
        # param[in] min_pt the X,Y min coordinate
        # param[in] max_pt the X,Y max coordinate
        # param[in] r the red channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] g the green channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] b the blue channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # template <typename T> bool
        # addRectangle (const typename pcl::PointCloud<T>::ConstPtr &image, 
        #               const T &min_pt, const T &max_pt,
        #               double r, double g, double b,
        #               const std::string &layer_id = "rectangles", double opacity = 1.0);
        # bool addRectangle (const shared_ptr[cpp.PointCloud[T]] &image, const T &min_pt, const T &max_pt, double r, double g, double b, const string &layer_id, double opacity)
        
        # brief Add a 2D box that contains a given image mask and color its edges
        # param[in] image the organized point cloud dataset containing the image data
        # param[in] mask the point data representing the mask that we want to draw
        # param[in] r the red channel of the color that the mask should be rendered with 
        # param[in] g the green channel of the color that the mask should be rendered with
        # param[in] b the blue channel of the color that the mask should be rendered with
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn.
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # template <typename T> bool
        # addRectangle (const typename pcl::PointCloud<T>::ConstPtr &image, const pcl::PointCloud<T> &mask, 
        #               double r, double g, double b, 
        #               const std::string &layer_id = "rectangles", double opacity = 1.0);
        # bool addRectangle (
        #                     const cpp.PointCloud[T] &image,
        #                     const cpp.PointCloud[T] &mask, 
        #                     double r, double g, double b, 
        #                     const string &layer_id, double opacity)
        
        # brief Add a 2D box that contains a given image mask and color its edges in red
        # param[in] image the organized point cloud dataset containing the image data
        # param[in] mask the point data representing the mask that we want to draw
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn.
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # template <typename T> bool
        # addRectangle (const typename pcl::PointCloud<T>::ConstPtr &image, const pcl::PointCloud<T> &mask, 
        #               const std::string &layer_id = "image_mask", double opacity = 1.0);
        # bool addRectangle (const shared_ptr[cpp.PointCloud[T]] &image, const shared_ptr[cpp.PointCloud[T]] &mask, const string &layer_id, double opacity)
        
        # brief Add a 2D box and fill it in with a given color
        # param[in] x_min the X min coordinate
        # param[in] x_max the X max coordinate
        # param[in] y_min the Y min coordinate
        # param[in] y_max the Y max coordinate 
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 0.5)
        # bool
        # addFilledRectangle (unsigned int x_min, unsigned int x_max, unsigned int y_min, unsigned int y_max,  
        #                     const std::string &layer_id = "boxes", double opacity = 0.5);
        bool addFilledRectangle (unsigned int x_min, unsigned int x_max, unsigned int y_min, unsigned int y_max, const string &layer_id, double opacity)
        
        # brief Add a 2D box and fill it in with a given color
        # param[in] x_min the X min coordinate
        # param[in] x_max the X max coordinate
        # param[in] y_min the Y min coordinate
        # param[in] y_max the Y max coordinate 
        # param[in] r the red channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] g the green channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] b the blue channel of the color that the box should be rendered with (0.0 -> 1.0)
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 0.5)
        # bool addFilledRectangle (unsigned int x_min, unsigned int x_max, unsigned int y_min, unsigned int y_max,  
        #                     double r, double g, double b,
        #                     const std::string &layer_id = "boxes", double opacity = 0.5);
        bool addFilledRectangle (
                                    unsigned int x_min, unsigned int x_max,
                                    unsigned int y_min, unsigned int y_max,  
                                    double r, double g, double b,
                                    const string &layer_id, double opacity)
        
        # brief Add a 2D line with a given color
        # param[in] x_min the X min coordinate
        # param[in] y_min the Y min coordinate
        # param[in] x_max the X max coordinate
        # param[in] y_max the Y max coordinate 
        # param[in] r the red channel of the color that the line should be rendered with (0.0 -> 1.0)
        # param[in] g the green channel of the color that the line should be rendered with (0.0 -> 1.0)
        # param[in] b the blue channel of the color that the line should be rendered with (0.0 -> 1.0)
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # bool
        # addLine (unsigned int x_min, unsigned int y_min, unsigned int x_max, unsigned int y_max,
        #          double r, double g, double b, 
        #          const std::string &layer_id = "line", double opacity = 1.0);
        bool addLine (
                      unsigned int x_min, unsigned int y_min,
                      unsigned int x_max, unsigned int y_max,
                      double r, double g, double b, 
                      const string &layer_id, double opacity)
        
        # brief Add a 2D line with a given color
        # param[in] x_min the X min coordinate
        # param[in] y_min the Y min coordinate
        # param[in] x_max the X max coordinate
        # param[in] y_max the Y max coordinate 
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn. 
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # bool
        # addLine (unsigned int x_min, unsigned int y_min, unsigned int x_max, unsigned int y_max,
        #          const std::string &layer_id = "line", double opacity = 1.0);
        bool addLine (
                        unsigned int x_min, unsigned int y_min,
                        unsigned int x_max, unsigned int y_max,
                        const string &layer_id, double opacity)
        
        # brief Add a generic 2D mask to an image 
        # param[in] image the organized point cloud dataset containing the image data
        # param[in] mask the point data representing the mask that we want to draw
        # param[in] r the red channel of the color that the mask should be rendered with 
        # param[in] g the green channel of the color that the mask should be rendered with
        # param[in] b the blue channel of the color that the mask should be rendered with
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn.
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 0.5)
        # template <typename T> bool
        # addMask (const typename pcl::PointCloud<T>::ConstPtr &image, const pcl::PointCloud<T> &mask, 
        #          double r, double g, double b, 
        #          const std::string &layer_id = "image_mask", double opacity = 0.5);
        # addMask (const shared_ptr[cpp.PointCloud[T]] &image, const shared_ptr[cpp.PointCloud[T]] &mask, double r, double g, double b, const string &layer_id, double opacity)
        
        # brief Add a generic 2D mask to an image (colored in red)
        # param[in] image the organized point cloud dataset containing the image data
        # param[in] mask the point data representing the mask that we want to draw
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn.
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 0.5)
        # template <typename T> bool
        # addMask (const typename pcl::PointCloud<T>::ConstPtr &image, const pcl::PointCloud<T> &mask, 
        #          const std::string &layer_id = "image_mask", double opacity = 0.5);
        # bool addMask (const shared_ptr[cpp.PointCloud[T]] &image, const shared_ptr[cpp.PointCloud[T]] &mask, const string &layer_id, double opacity)
        
        # brief Add a generic 2D planar polygon to an image 
        # param[in] image the organized point cloud dataset containing the image data
        # param[in] polygon the point data representing the polygon that we want to draw. 
        # A line will be drawn from each point to the next in the dataset.
        # param[in] r the red channel of the color that the polygon should be rendered with 
        # param[in] g the green channel of the color that the polygon should be rendered with
        # param[in] b the blue channel of the color that the polygon should be rendered with
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn.
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # template <typename T> bool
        # addPlanarPolygon (const typename pcl::PointCloud<T>::ConstPtr &image, const pcl::PlanarPolygon<T> &polygon, 
        #                   double r, double g, double b, 
        #                   const std::string &layer_id = "planar_polygon", double opacity = 1.0);
        # bool addPlanarPolygon (const shared_ptr[cpp.PointCloud[T]] &image, const cpp.PlanarPolygon[T] &polygon, double r, double g, double b, const string &layer_id, double opacity)
        
        # brief Add a generic 2D planar polygon to an image 
        # param[in] image the organized point cloud dataset containing the image data
        # param[in] polygon the point data representing the polygon that we want to draw. 
        # A line will be drawn from each point to the next in the dataset.
        # param[in] layer_id the 2D layer ID where we want the extra information to be drawn.
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 1.0)
        # 
        # template <typename T> bool
        # addPlanarPolygon (const typename pcl::PointCloud<T>::ConstPtr &image, const pcl::PlanarPolygon<T> &polygon, 
        #                   const std::string &layer_id = "planar_polygon", double opacity = 1.0);
        # bool addPlanarPolygon (const shared_ptr[cpp.PointCloud[T]] &image, const cpp.PlanarPolygon[T] &polygon, const string &layer_id, double opacity)
        
        # brief Add a new 2D rendering layer to the viewer. 
        # param[in] layer_id the name of the layer
        # param[in] width the width of the layer
        # param[in] height the height of the layer
        # param[in] opacity the opacity of the layer: 0 for invisible, 1 for opaque. (default: 0.5)
        # bool addLayer (const std::string &layer_id, int width, int height, double opacity = 0.5);
        bool addLayer (const string &layer_id, int width, int height, double opacity)
        
        # brief Remove a 2D layer given by its ID.
        # param[in] layer_id the name of the layer
        # void removeLayer (const std::string &layer_id);
        void removeLayer (const string &layer_id)


###

# interactor.h
# namespace pcl
# namespace visualization
#     /** \brief The PCLVisualizer interactor */
# #ifdef _WIN32
#     class PCL_EXPORTS PCLVisualizerInteractor : public vtkWin32RenderWindowInteractor
# #elif defined VTK_USE_CARBON
#     class PCLVisualizerInteractor : public vtkCarbonRenderWindowInteractor
# #elif defined VTK_USE_COCOA
#     class PCLVisualizerInteractor : public vtkCocoaRenderWindowInteractor
# #else
#     class PCLVisualizerInteractor : public vtkXRenderWindowInteractor
# #endif
        # public:
        # static PCLVisualizerInteractor *New ();
        # 
        # void stopLoop ();
        # 
        # bool stopped;
        # int timer_id_;
        # 
        # #ifdef _WIN32
        # int BreakLoopFlag;                // if true quit the GetMessage loop
        # virtual void Start ();                         // Redefine the vtkWin32RenderWindowInteractor::Start method...
        # vtkGetMacro (BreakLoopFlag, int);
        # void SetBreakLoopFlag (int);           // Change the value of BreakLoopFlag
        # void BreakLoopFlagOff ();              // set BreakLoopFlag to 0
        # void BreakLoopFlagOn ();               // set BreakLoopFlag to 1 (quit)
        # #endif


###

# interactor_style.h
# namespace pcl
# namespace visualization
# /** \brief A list of potential keyboard modifiers for \ref PCLVisualizerInteractorStyle.
#   * Defaults to Alt. 
#   */ 
# enum InteractorKeyboardModifier
# {
#   INTERACTOR_KB_MOD_ALT,
#   INTERACTOR_KB_MOD_CTRL,
#   INTERACTOR_KB_MOD_SHIFT
# };

# interactor_style.h
# namespace pcl
# namespace visualization
# /** \brief PCLVisualizerInteractorStyle defines an unique, custom VTK
#   * based interactory style for PCL Visualizer applications. Besides
#   * defining the rendering style, we also create a list of custom actions
#   * that are triggered on different keys being pressed:
#   *
#   * -        p, P   : switch to a point-based representation
#   * -        w, W   : switch to a wireframe-based representation (where available)
#   * -        s, S   : switch to a surface-based representation (where available)
#   * -        j, J   : take a .PNG snapshot of the current window view
#   * -        c, C   : display current camera/window parameters
#   * -        f, F   : fly to point mode
#   * -        e, E   : exit the interactor\
#   * -        q, Q   : stop and call VTK's TerminateApp
#   * -       + / -   : increment/decrement overall point size
#   * -        g, G   : display scale grid (on/off)
#   * -        u, U   : display lookup table (on/off)
#   * -  r, R [+ ALT] : reset camera [to viewpoint = {0, 0, 0} -> center_{x, y, z}]
#   * -  ALT + s, S   : turn stereo mode on/off
#   * -  ALT + f, F   : switch between maximized window mode and original size
#   * -        l, L           : list all available geometric and color handlers for the current actor map
#   * -  ALT + 0..9 [+ CTRL]  : switch between different geometric handlers (where available)
#   * -        0..9 [+ CTRL]  : switch between different color handlers (where available)
#   * - 
#   * -  SHIFT + left click   : select a point
#   *
#   * \author Radu B. Rusu
#   * \ingroup visualization
#   */
# class PCL_EXPORTS PCLVisualizerInteractorStyle : public vtkInteractorStyleTrackballCamera
        # typedef boost::shared_ptr<CloudActorMap> CloudActorMapPtr;
        # public:
        # static PCLVisualizerInteractorStyle *New ();
        # 
        # /** \brief Empty constructor. */
        # PCLVisualizerInteractorStyle () : 
        #   init_ (), rens_ (), actors_ (), win_height_ (), win_width_ (), win_pos_x_ (), win_pos_y_ (),
        #   max_win_height_ (), max_win_width_ (), grid_enabled_ (), grid_actor_ (), lut_enabled_ (),
        #   lut_actor_ (), snapshot_writer_ (), wif_ (), mouse_signal_ (), keyboard_signal_ (),
        #   point_picking_signal_ (), stereo_anaglyph_mask_default_ (), mouse_callback_ (), modifier_ ()
        # {}
        # 
        # // this macro defines Superclass, the isA functionality and the safe downcast method
        # vtkTypeMacro (PCLVisualizerInteractorStyle, vtkInteractorStyleTrackballCamera);
        # 
        # /** \brief Initialization routine. Must be called before anything else. */
        # virtual void Initialize ();
        # 
        # /** \brief Pass a pointer to the actor map
        #   * \param[in] actors the actor map that will be used with this style
        #   */
        # inline void setCloudActorMap (const CloudActorMapPtr &actors) { actors_ = actors; }
        # 
        # /** \brief Get the cloud actor map pointer. */
        # inline CloudActorMapPtr getCloudActorMap () { return (actors_); }
        # 
        # /** \brief Pass a set of renderers to the interactor style. 
        #   * \param[in] rens the vtkRendererCollection to use
        #   */
        # void setRendererCollection (vtkSmartPointer<vtkRendererCollection> &rens) { rens_ = rens; }
        # 
        # /** \brief Register a callback function for mouse events
        #   * \param[in] cb a boost function that will be registered as a callback for a mouse event
        #   * \return a connection object that allows to disconnect the callback function.
        #   */
        # boost::signals2::connection registerMouseCallback (boost::function<void (const pcl::visualization::MouseEvent&)> cb);
        # 
        # /** \brief Register a callback boost::function for keyboard events
        #   * \param[in] cb a boost function that will be registered as a callback for a keyboard event
        #   * \return a connection object that allows to disconnect the callback function.
        #   */
        # boost::signals2::connection registerKeyboardCallback (boost::function<void (const pcl::visualization::KeyboardEvent&)> cb);
        # 
        # /** \brief Register a callback function for point picking events
        #   * \param[in] cb a boost function that will be registered as a callback for a point picking event
        #   * \return a connection object that allows to disconnect the callback function.
        #   */
        # boost::signals2::connection registerPointPickingCallback (boost::function<void (const pcl::visualization::PointPickingEvent&)> cb);
        # 
        # /** \brief Save the current rendered image to disk, as a PNG screenshot.
        #   * \param[in] file the name of the PNG file
        #   */
        # void saveScreenshot (const std::string &file);
        # 
        # /** \brief Change the default keyboard modified from ALT to a different special key.
        #   * Allowed values are:
        #   * - INTERACTOR_KB_MOD_ALT
        #   * - INTERACTOR_KB_MOD_CTRL
        #   * - INTERACTOR_KB_MOD_SHIFT
        #   * \param[in] modifier the new keyboard modifier
        #   */
        # inline void setKeyboardModifier (const InteractorKeyboardModifier &modifier)


###

# interactor_style.h
# namespace pcl
# namespace visualization
# /** \brief PCL histogram visualizer interactory style class.
#   * \author Radu B. Rusu
#   */
# class PCLHistogramVisualizerInteractorStyle : public vtkInteractorStyleTrackballCamera
        # public:
        # static PCLHistogramVisualizerInteractorStyle *New ();
        # 
        # /** \brief Empty constructor. */
        # PCLHistogramVisualizerInteractorStyle () : wins_ (), init_ (false) {}
        # 
        # /** \brief Initialization routine. Must be called before anything else. */
        # void Initialize ();
        # 
        # /** \brief Pass a map of render/window/interactors to the interactor style. 
        #   * \param[in] wins the RenWinInteract map to use
        #   */
        # void setRenWinInteractMap (const RenWinInteractMap &wins) { wins_ = wins; }


###

# keyboard_event.h
# namespace pcl
# namespace visualization
# /** /brief Class representing key hit/release events */
# class KeyboardEvent
        # public:
        # /** \brief bit patter for the ALT key*/
        # static const unsigned int Alt   = 1;
        # /** \brief bit patter for the Control key*/
        # static const unsigned int Ctrl  = 2;
        # /** \brief bit patter for the Shift key*/
        # static const unsigned int Shift = 4;
        # 
        # /** \brief Constructor
        #   * \param[in] action    true for key was pressed, false for released
        #   * \param[in] key_sym   the key-name that caused the action
        #   * \param[in] key       the key code that caused the action
        #   * \param[in] alt       whether the alt key was pressed at the time where this event was triggered
        #   * \param[in] ctrl      whether the ctrl was pressed at the time where this event was triggered
        #   * \param[in] shift     whether the shift was pressed at the time where this event was triggered
        #   */
        # inline KeyboardEvent (bool action, const std::string& key_sym, unsigned char key, bool alt, bool ctrl, bool shift);
        # 
        # /**
        #   * \return   whether the alt key was pressed at the time where this event was triggered
        #   */
        # inline bool isAltPressed () const;
        # 
        # /**
        #   * \return whether the ctrl was pressed at the time where this event was triggered
        #   */
        # inline bool isCtrlPressed () const;
        # 
        # /**
        #   * \return whether the shift was pressed at the time where this event was triggered
        #   */
        # inline bool isShiftPressed () const;
        # 
        # /**
        #   * \return the ASCII Code of the key that caused the event. If 0, then it was a special key, like ALT, F1, F2,... PgUp etc. Then the name of the key is in the keysym field.
        #   */
        # inline unsigned char getKeyCode () const;
        # 
        # /**
        #   * \return name of the key that caused the event
        #   */
        # inline const std::string& getKeySym () const;
        # 
        # /**
        #   * \return true if a key-press caused the event, false otherwise
        #   */
        # inline bool keyDown () const;
        # 
        # /**
        #   * \return true if a key-release caused the event, false otherwise
        #   */
        # inline bool keyUp () const;

    # KeyboardEvent::KeyboardEvent (bool action, const std::string& key_sym, unsigned char key, bool alt, bool ctrl, bool shift)
    #   : action_ (action)
    #   , modifiers_ (0)
    #   , key_code_(key)
    #   , key_sym_ (key_sym)
    # 
    # bool KeyboardEvent::isAltPressed () const
    # bool KeyboardEvent::isCtrlPressed () const
    # bool KeyboardEvent::isShiftPressed () const
    # unsigned char KeyboardEvent::getKeyCode () const
    # const std::string& KeyboardEvent::getKeySym () const
    # bool KeyboardEvent::keyDown () const
    # bool KeyboardEvent::keyUp () const


###

# mouse_event.h
# namespace pcl
# namespace visualization
# class MouseEvent
        # public:
        # typedef enum
        # {
        #   MouseMove = 1,
        #       MouseButtonPress,
        #       MouseButtonRelease,
        #       MouseScrollDown,
        #       MouseScrollUp,
        #       MouseDblClick
        # } Type;
        # 
        # typedef enum
        # {
        #       NoButton      = 0,
        #       LeftButton,
        #       MiddleButton,
        #       RightButton,
        #       VScroll /*other buttons, scroll wheels etc. may follow*/
        # } MouseButton;
        # 
        # /** Constructor.
        #   * \param[in] type   event type
        #   * \param[in] button The Button that causes the event
        #   * \param[in] x      x position of mouse pointer at that time where event got fired
        #   * \param[in] y      y position of mouse pointer at that time where event got fired
        #   * \param[in] alt    whether the ALT key was pressed at that time where event got fired
        #   * \param[in] ctrl   whether the CTRL key was pressed at that time where event got fired
        #   * \param[in] shift  whether the Shift key was pressed at that time where event got fired
        #   */
        # inline MouseEvent (const Type& type, const MouseButton& button, unsigned int x, unsigned int y, bool alt, bool ctrl, bool shift);
        # 
        # /**
        #   * \return type of mouse event
        #   */
        # inline const Type& getType () const;
        # 
        # /**
        #   * \brief Sets the mouse event type
        #   */
        # inline void setType (const Type& type);
        # 
        # /**
        #   * \return the Button that caused the action
        #   */
        # inline const MouseButton& getButton () const;
        # 
        # /** \brief Set the button that caused the event */
        # inline void setButton (const MouseButton& button);
        # 
        # /**
        #   * \return the x position of the mouse pointer at that time where the event got fired
        #   */
        # inline unsigned int getX () const;
        # 
        # /**
        #   * \return the y position of the mouse pointer at that time where the event got fired
        #   */
        # inline unsigned int getY () const;
        # 
        # /**
        #   * \return returns the keyboard modifiers state at that time where the event got fired
        #   */
        # inline unsigned int getKeyboardModifiers () const;
        # 

    # MouseEvent::MouseEvent (const Type& type, const MouseButton& button, unsigned x, unsigned y,  bool alt, bool ctrl, bool shift)
    # : type_ (type)
    # , button_ (button)
    # , pointer_x_ (x)
    # , pointer_y_ (y)
    # , key_state_ (0)
    # 
    # const MouseEvent::Type& MouseEvent::getType () const
    # void MouseEvent::setType (const Type& type)
    # const MouseEvent::MouseButton& MouseEvent::getButton () const
    # void MouseEvent::setButton (const MouseButton& button)
    # unsigned int MouseEvent::getX () const
    # unsigned int MouseEvent::getY () const
    # unsigned int MouseEvent::getKeyboardModifiers () const


###

# point_picking_event.h
# class PCL_EXPORTS PointPickingCallback : public vtkCommand
        # public:
        # static PointPickingCallback *New () 
        # PointPickingCallback () : x_ (0), y_ (0), z_ (0), idx_ (-1), pick_first_ (false) {}
        # 
        # virtual void Execute (vtkObject *caller, unsigned long eventid, void*);
        # 
        # int performSinglePick (vtkRenderWindowInteractor *iren);
        # 
        # int performSinglePick (vtkRenderWindowInteractor *iren, float &x, float &y, float &z);
###

# class PCL_EXPORTS PointPickingEvent
        # public:
        # PointPickingEvent (int idx) : idx_ (idx), idx2_ (-1), x_ (), y_ (), z_ (), x2_ (), y2_ (), z2_ () {}
        # PointPickingEvent (int idx, float x, float y, float z) : idx_ (idx), idx2_ (-1), x_ (x), y_ (y), z_ (z), x2_ (), y2_ (), z2_ () {}
        # 
        # PointPickingEvent (int idx1, int idx2, float x1, float y1, float z1, float x2, float y2, float z2) :
        
        # /** \brief Obtain the ID of a point that the user just clicked on. */
        # inline int getPointIndex () const
        
        # /** \brief Obtain the XYZ point coordinates of a point that the user just clicked on.
        #   * \param[out] x the x coordinate of the point that got selected by the user
        #   * \param[out] y the y coordinate of the point that got selected by the user
        #   * \param[out] z the z coordinate of the point that got selected by the user
        #   */
        # inline void getPoint (float &x, float &y, float &z) const
        
        # /** \brief For situations when multiple points are selected in a sequence, return the point coordinates.
        #   * \param[out] x1 the x coordinate of the first point that got selected by the user
        #   * \param[out] y1 the y coordinate of the first point that got selected by the user
        #   * \param[out] z1 the z coordinate of the firts point that got selected by the user
        #   * \param[out] x2 the x coordinate of the second point that got selected by the user
        #   * \param[out] y2 the y coordinate of the second point that got selected by the user
        #   * \param[out] z2 the z coordinate of the second point that got selected by the user
        #   * \return true, if two points are available and have been clicked by the user, false otherwise
        # inline bool getPoints (float &x1, float &y1, float &z1, float &x2, float &y2, float &z2) const
###

# range_image_visualizer.h
# class PCL_EXPORTS RangeImageVisualizer : public ImageViewer
cdef extern from "pcl/visualization/range_image_visualizer.h" namespace "pcl::visualization" nogil:
    cdef cppclass RangeImageVisualizer(ImageViewer):
        RangeImageVisualizer()
        RangeImageVisualizer (const string name)
        # public:
        # =====CONSTRUCTOR & DESTRUCTOR=====
        # //! Constructor
        # RangeImageVisualizer (const std::string& name="Range Image");
        # //! Destructor
        # ~RangeImageVisualizer ();
        
        # =====PUBLIC STATIC METHODS=====
        # Get a widget visualizing the given range image.
        # You are responsible for deleting it after usage!
        # static RangeImageVisualizer* getRangeImageWidget (
        #                                   const pcl::RangeImage& range_image, float min_value,
        #                                   float max_value, bool grayscale, const std::string& name="Range image");
        # RangeImageVisualizer* getRangeImageWidget (pcl.RangeImage& range_image, float min_value, float max_value, bool grayscale, const string& name)
        
        # Visualize the given range image and the detected borders in it.
        # Borders on the obstacles are marked green, borders on the background are marked bright blue.
        # void visualizeBorders (const pcl::RangeImage& range_image, float min_value, float max_value, bool grayscale,
        #                        const pcl::PointCloud<pcl::BorderDescription>& border_descriptions);
        # void visualizeBorders (const pcl.RangeImage& range_image, float min_value, float max_value, bool grayscale, const cpp.PointCloud[cpp.BorderDescription] &border_descriptions)
        
        # /** Same as above, but returning a new widget. You are responsible for deleting it after usage!
        # static RangeImageVisualizer* getRangeImageBordersWidget (const pcl::RangeImage& range_image, float min_value,
        #               float max_value, bool grayscale, const pcl::PointCloud<pcl::BorderDescription>& border_descriptions,
        #               const std::string& name="Range image with borders");
        # RangeImageVisualizer* getRangeImageBordersWidget (
        #                 const pcl.RangeImage& range_image, 
        #                 float min_value,
        #                 float max_value,
        #                 bool grayscale, 
        #                 const cpp.PointCloud[cpp.BorderDescription] &border_descriptions,
        #                 const string& name)
        
        # Get a widget visualizing the given angle image (assuming values in (-PI, PI]).
        # -PI and PI will return the same color
        # You are responsible for deleting it after usage!
        # static RangeImageVisualizer* getAnglesWidget (const pcl::RangeImage& range_image, float* angles_image, const std::string& name);
        RangeImageVisualizer* getAnglesWidget (const RangeImage& range_image, float* angles_image, const string& name)
        
        # Get a widget visualizing the given angle image (assuming values in (-PI/2, PI/2]).
        # -PI/2 and PI/2 will return the same color
        # You are responsible for deleting it after usage!
        # RangeImageVisualizer* getHalfAnglesWidget (const pcl.RangeImage& range_image, float* angles_image, const string& name)
        RangeImageVisualizer* getHalfAnglesWidget (const RangeImage& range_image, float* angles_image, const string& name)
        
        # /** Get a widget visualizing the interest values and extracted interest points.
        #  * The interest points will be marked green.
        #  *  You are responsible for deleting it after usage! */
        # static RangeImageVisualizer* getInterestPointsWidget (const pcl::RangeImage& range_image, const float* interest_image, float min_value, float max_value,
        #                                                       const pcl::PointCloud<pcl::InterestPoint>& interest_points, const std::string& name);
        RangeImageVisualizer* getInterestPointsWidget (const RangeImage& range_image, const float* interest_image, float min_value, float max_value, const cpp.PointCloud[cpp.InterestPoint] &interest_points, const string& name)
        
        # // =====PUBLIC METHODS=====
        # //! Visualize a range image
        # /* void  */
        # /* setRangeImage (const pcl::RangeImage& range_image,  */
        # /*                float min_value = -std::numeric_limits<float>::infinity (),  */
        # /*                float max_value =  std::numeric_limits<float>::infinity (),  */
        # /*                bool grayscale  = false); */
        
        # void showRangeImage (const pcl::RangeImage& range_image, 
        #                       float min_value = -std::numeric_limits<float>::infinity (), 
        #                       float max_value =  std::numeric_limits<float>::infinity (), 
        #                       bool grayscale  = false);
        void showRangeImage (const RangeImage range_image,  float min_value, float max_value, bool grayscale)


###

# registration_visualizer.h
# template<typename PointSource, typename PointTarget>
# class RegistrationVisualizer
cdef extern from "pcl/visualization/registration_visualizer.h" namespace "pcl::visualization" nogil:
    cdef cppclass RegistrationVisualizer[Source, Target]:
        RegistrationVisualizer ()
        
        # public:
        # /** \brief Set the registration algorithm whose intermediate steps will be rendered.
        # * The method creates the local callback function pcl::RegistrationVisualizer::update_visualizer_ and
        # * binds it to the local biffers update function pcl::RegistrationVisualizer::updateIntermediateCloud().
        # * The local callback function pcl::RegistrationVisualizer::update_visualizer_ is then linked to
        # * the pcl::Registration::update_visualizer_ callback function.
        # * \param registration represents the registration method whose intermediate steps will be rendered.
        # bool setRegistration (pcl::Registration<PointSource, PointTarget> &registration)
        # bool setRegistration (pcl.Registration[Source, Target] &registration)
        
        # /** \brief Start the viewer thread
        # void startDisplay ();
        void startDisplay ()
        
        # /** \brief Stop the viewer thread
        # void stopDisplay ();
        void stopDisplay ()
        
        # /** \brief Updates visualizer local buffers cloud_intermediate, cloud_intermediate_indices, cloud_target_indices with
        # * the newest registration intermediate results.
        # * \param cloud_src represents the initial source point cloud
        # * \param indices_src represents the incices of the intermediate source points used for the estimation of rigid transformation
        # * \param cloud_tgt represents the target point cloud
        # * \param indices_tgt represents the incices of the target points used for the estimation of rigid transformation
        # void updateIntermediateCloud (const pcl::PointCloud<PointSource> &cloud_src, const std::vector<int> &indices_src, const pcl::PointCloud<PointTarget> &cloud_tgt, const std::vector<int> &indices_tgt);
        void updateIntermediateCloud (const cpp.PointCloud[Source] &cloud_src, const vector[int] &indices_src,
                                      const cpp.PointCloud[Target] &cloud_tgt, const vector[int] &indices_tgt)
        
        # /** \brief Set maximum number of corresponcence lines whch will be rendered. */
        # inline void setMaximumDisplayedCorrespondences (const int maximum_displayed_correspondences)
        void setMaximumDisplayedCorrespondences (const int maximum_displayed_correspondences)
        
        # /** \brief Return maximum number of corresponcence lines which are rendered. */
        # inline size_t getMaximumDisplayedCorrespondences()
        size_t getMaximumDisplayedCorrespondences()


###

# vtk.h
# header file include define
###

# window.h
# class PCL_EXPORTS Window
cdef extern from "pcl/visualization/window.h" namespace "pcl::visualization" nogil:
    cdef cppclass Window:
        Window ()
        # public:
        # Window (const std::string& window_name = "");
        # Window (const Window &src);
        # Window& operator = (const Window &src);
        # virtual ~Window ();
        
        # /** \brief Spin method. Calls the interactor and runs an internal loop. */
        # void spin ()
        
        # /** \brief Spin once method. Calls the interactor and updates the screen once.
        #   *  \param time - How long (in ms) should the visualization loop be allowed to run.
        #   *  \param force_redraw - if false it might return without doing anything if the
        #   *  interactor's framerate does not require a redraw yet.
        # void spinOnce (int time = 1, bool force_redraw = false);
        
        # /** \brief Returns true when the user tried to close the window */
        # bool wasStopped () const
        
        # /**
        #   * @brief registering a callback function for keyboard events
        #   * @param callback  the function that will be registered as a callback for a keyboard event
        #   * @param cookie    user data that is passed to the callback
        #   * @return          connection object that allows to disconnect the callback function.
        # boost::signals2::connection registerKeyboardCallback (void (*callback) (const pcl::visualization::KeyboardEvent&, void*), void* cookie = NULL)
        
        # /**
        #   * @brief registering a callback function for keyboard events
        #   * @param callback  the member function that will be registered as a callback for a keyboard event
        #   * @param instance  instance to the class that implements the callback function
        #   * @param cookie    user data that is passed to the callback
        #   * @return          connection object that allows to disconnect the callback function.
        # template<typename T> boost::signals2::connection
        # registerKeyboardCallback (void (T::*callback) (const pcl::visualization::KeyboardEvent&, void*), T& instance, void* cookie = NULL)
        
        # /**
        #   * @brief
        #   * @param callback  the function that will be registered as a callback for a mouse event
        #   * @param cookie    user data that is passed to the callback
        #   * @return          connection object that allows to disconnect the callback function.
        # boost::signals2::connection
        # registerMouseCallback (void (*callback) (const pcl::visualization::MouseEvent&, void*), void* cookie = NULL)
        
        # /**
        #   * @brief registering a callback function for mouse events
        #   * @param callback  the member function that will be registered as a callback for a mouse event
        #   * @param instance  instance to the class that implements the callback function
        #   * @param cookie    user data that is passed to the callback
        #   * @return          connection object that allows to disconnect the callback function.
        # template<typename T> boost::signals2::connection
        # registerMouseCallback (void (T::*callback) (const pcl::visualization::MouseEvent&, void*), T& instance, void* cookie = NULL)


###

###############################################################################
# Enum
###############################################################################

# common.h
cdef extern from "pcl/visualization/common/common.h" namespace "pcl::visualization":
    cdef enum FrustumCull:
        PCL_INSIDE_FRUSTUM
        PCL_INTERSECT_FRUSTUM
        PCL_OUTSIDE_FRUSTUM

cdef extern from "pcl/visualization/common/common.h" namespace "pcl::visualization":
    cdef enum RenderingProperties:
        PCL_VISUALIZER_POINT_SIZE
        PCL_VISUALIZER_OPACITY
        PCL_VISUALIZER_LINE_WIDTH
        PCL_VISUALIZER_FONT_SIZE
        PCL_VISUALIZER_COLOR
        PCL_VISUALIZER_REPRESENTATION
        PCL_VISUALIZER_IMMEDIATE_RENDERING
        # PCL_VISUALIZER_SHADING

cdef extern from "pcl/visualization/common/common.h" namespace "pcl::visualization":
    cdef enum RenderingRepresentationProperties:
        PCL_VISUALIZER_REPRESENTATION_POINTS
        PCL_VISUALIZER_REPRESENTATION_WIREFRAME
        PCL_VISUALIZER_REPRESENTATION_SURFACE

cdef extern from "pcl/visualization/common/common.h" namespace "pcl::visualization":
    cdef enum ShadingRepresentationProperties:
        PCL_VISUALIZER_SHADING_FLAT
        PCL_VISUALIZER_SHADING_GOURAUD
        PCL_VISUALIZER_SHADING_PHONG

###############################################################################
# Activation
###############################################################################
