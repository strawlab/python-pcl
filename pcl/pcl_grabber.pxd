# -*- coding: utf-8 -*-

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair

# main
cimport pcl_defs as cpp

from boost_shared_ptr cimport shared_ptr
# from boost_function cimport function
# from boost_signal2_connection cimport connection
# bind
from _bind_defs cimport connection
from _bind_defs cimport arg
from _bind_defs cimport function
from _bind_defs cimport callback_t

###############################################################################
# Types
###############################################################################

### base class ###

# grabber.h
# namespace pcl
# /** \brief Grabber interface for PCL 1.x device drivers
#  * \author Suat Gedikli <gedikli@willowgarage.com>
#  * \ingroup io
#  */
# class PCL_EXPORTS Grabber
cdef extern from "pcl/io/grabber.h" namespace "pcl":
    cdef cppclass Grabber:
        Grabber ()
        # public:
        # /** \brief registers a callback function/method to a signal with the corresponding signature
        #   * \param[in] callback: the callback function/method
        #   * \return Connection object, that can be used to disconnect the callback method from the signal again.
        # template<typename T> boost::signals2::connection registerCallback (const boost::function<T>& callback);
        # connection registerCallback[T](function[T]& callback)
        connection registerCallback[T](function[T] callback)
        
        # /** \brief indicates whether a signal with given parameter-type exists or not
        #   * \return true if signal exists, false otherwise
        # template<typename T> bool providesCallback () const;
        bool providesCallback[T]()
        
        # 
        # /** \brief For devices that are streaming, the streams are started by calling this method.
        #   *        Trigger-based devices, just trigger the device once for each call of start.
        # virtual void start () = 0;
        # 
        # /** \brief For devices that are streaming, the streams are stopped.
        #   *        This method has no effect for triggered devices.
        # virtual void stop () = 0;
        # 
        # /** \brief returns the name of the concrete subclass.
        #   * \return the name of the concrete driver.
        # virtual std::string getName () const = 0;
        # 
        # /** \brief Indicates whether the grabber is streaming or not. This value is not defined for triggered devices.
        #   * \return true if grabber is running / streaming. False otherwise.
        # virtual bool isRunning () const = 0;
        # 
        # /** \brief returns fps. 0 if trigger based. */
        # virtual float getFramesPerSecond () const = 0;


# template<typename T> boost::signals2::signal<T>* Grabber::find_signal () const
# template<typename T> void Grabber::disconnect_all_slots ()
# template<typename T> void Grabber::block_signal ()
# template<typename T> void Grabber::unblock_signal ()
# void Grabber::block_signals ()
# void Grabber::unblock_signals ()
# template<typename T> int Grabber::num_slots () const
# template<typename T> boost::signals2::signal<T>* Grabber::createSignal ()
# template<typename T> boost::signals2::connection Grabber::registerCallback (const boost::function<T> & callback)
# template<typename T> bool Grabber::providesCallback () const


###

# oni_grabber.h
# namespace pcl
#     struct PointXYZ;
#     struct PointXYZRGB;
#     struct PointXYZRGBA;
#     struct PointXYZI;
# template <typename T> class PointCloud;
# /** \brief A simple ONI grabber.
#  * \author Suat Gedikli
# class PCL_EXPORTS ONIGrabber : public Grabber
# cdef extern from "pcl/io/oni_grabber.h" namespace "pcl":
#     cdef cppclass ONIGrabber(Grabber):
#         ONIGrabber (string file_name, bool repeat, bool stream)
        # public:
        # //define callback signature typedefs
        # typedef void (sig_cb_openni_image) (const boost::shared_ptr<openni_wrapper::Image>&);
        # typedef void (sig_cb_openni_depth_image) (const boost::shared_ptr<openni_wrapper::DepthImage>&);
        # typedef void (sig_cb_openni_ir_image) (const boost::shared_ptr<openni_wrapper::IRImage>&);
        # typedef void (sig_cb_openni_image_depth_image) (const boost::shared_ptr<openni_wrapper::Image>&, const boost::shared_ptr<openni_wrapper::DepthImage>&, float constant) ;
        # typedef void (sig_cb_openni_ir_depth_image) (const boost::shared_ptr<openni_wrapper::IRImage>&, const boost::shared_ptr<openni_wrapper::DepthImage>&, float constant) ;
        # typedef void (sig_cb_openni_point_cloud) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&);
        # typedef void (sig_cb_openni_point_cloud_rgb) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGB> >&);
        # typedef void (sig_cb_openni_point_cloud_rgba) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA> >&);
        # typedef void (sig_cb_openni_point_cloud_i) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&);
        # 
        # /** \brief For devices that are streaming, the streams are started by calling this method.
        #   * Trigger-based devices, just trigger the device once for each call of start.
        # void start ()
        # 
        # /** \brief For devices that are streaming, the streams are stopped.
        #  *        This method has no effect for triggered devices.
        #  */
        # void stop ()
        #
        # /** \brief returns the name of the concrete subclass.
        #  * \return the name of the concrete driver.
        #  */
        # string getName ()
        #
        # /** \brief Indicates whether the grabber is streaming or not. This value is not defined for triggered devices.
        #  * \return true if grabber is running / streaming. False otherwise.
        #  */
        # bool isRunning ()
        # 
        # /** \brief returns the frames pre second. 0 if it is trigger based. */
        # float getFramesPerSecond ()
        # 
        # protected:
        # /** \brief internal OpenNI (openni_wrapper) callback that handles image streams */
        # void imageCallback (boost::shared_ptr<openni_wrapper::Image> image, void* cookie);
        # /** \brief internal OpenNI (openni_wrapper) callback that handles depth streams */
        # void depthCallback (boost::shared_ptr<openni_wrapper::DepthImage> depth_image, void* cookie);
        # /** \brief internal OpenNI (openni_wrapper) callback that handles IR streams */
        # void irCallback (boost::shared_ptr<openni_wrapper::IRImage> ir_image, void* cookie);
        # /** \brief internal callback that handles synchronized image + depth streams */
        # void imageDepthImageCallback (const boost::shared_ptr<openni_wrapper::Image> &image,
        #                         const boost::shared_ptr<openni_wrapper::DepthImage> &depth_image);
        # /** \brief internal callback that handles synchronized IR + depth streams */
        # void irDepthImageCallback (const boost::shared_ptr<openni_wrapper::IRImage> &image,
        #                      const boost::shared_ptr<openni_wrapper::DepthImage> &depth_image);
        # /** \brief internal method to assemble a point cloud object */
        # boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > convertToXYZPointCloud (const boost::shared_ptr<openni_wrapper::DepthImage> &depth) const;
        # /** \brief internal method to assemble a point cloud object */
        # boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> >
        # convertToXYZRGBPointCloud (const boost::shared_ptr<openni_wrapper::Image> &image,
        #                           const boost::shared_ptr<openni_wrapper::DepthImage> &depth_image) const;
        # 
        # /** \brief internal method to assemble a point cloud object */
        # boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA> >
        # convertToXYZRGBAPointCloud (const boost::shared_ptr<openni_wrapper::Image> &image,
        #                            const boost::shared_ptr<openni_wrapper::DepthImage> &depth_image) const;
        # 
        # /** \brief internal method to assemble a point cloud object */
        # boost::shared_ptr<pcl::PointCloud<pcl::PointXYZI> >
        # convertToXYZIPointCloud (const boost::shared_ptr<openni_wrapper::IRImage> &image,
        #                         const boost::shared_ptr<openni_wrapper::DepthImage> &depth_image) const;
        # 
        # /** \brief synchronizer object to synchronize image and depth streams*/
        # Synchronizer<boost::shared_ptr<openni_wrapper::Image>, boost::shared_ptr<openni_wrapper::DepthImage> > rgb_sync_;
        # 
        # /** \brief synchronizer object to synchronize IR and depth streams*/
        # Synchronizer<boost::shared_ptr<openni_wrapper::IRImage>, boost::shared_ptr<openni_wrapper::DepthImage> > ir_sync_;
        # 
        # /** \brief the actual openni device*/
        # boost::shared_ptr<openni_wrapper::DeviceONI> device_;
        # std::string rgb_frame_id_;
        # std::string depth_frame_id_;
        # bool running_;
        # unsigned image_width_;
        # unsigned image_height_;
        # unsigned depth_width_;
        # unsigned depth_height_;
        # openni_wrapper::OpenNIDevice::CallbackHandle depth_callback_handle;
        # openni_wrapper::OpenNIDevice::CallbackHandle image_callback_handle;
        # openni_wrapper::OpenNIDevice::CallbackHandle ir_callback_handle;
        # boost::signals2::signal<sig_cb_openni_image >*            image_signal_;
        # boost::signals2::signal<sig_cb_openni_depth_image >*      depth_image_signal_;
        # boost::signals2::signal<sig_cb_openni_ir_image >*         ir_image_signal_;
        # boost::signals2::signal<sig_cb_openni_image_depth_image>* image_depth_image_signal_;
        # boost::signals2::signal<sig_cb_openni_ir_depth_image>*    ir_depth_image_signal_;
        # boost::signals2::signal<sig_cb_openni_point_cloud >*      point_cloud_signal_;
        # boost::signals2::signal<sig_cb_openni_point_cloud_i >*    point_cloud_i_signal_;
        # boost::signals2::signal<sig_cb_openni_point_cloud_rgb >*  point_cloud_rgb_signal_;
        # boost::signals2::signal<sig_cb_openni_point_cloud_rgba >*  point_cloud_rgba_signal_;
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW


###

# openni_grabber.h
# namespace pcl
#   struct PointXYZ;
#   struct PointXYZRGB;
#   struct PointXYZRGBA;
#   struct PointXYZI;
#   template <typename T> class PointCloud;
# 
# /** \brief Grabber for OpenNI devices (i.e., Primesense PSDK, Microsoft Kinect, Asus XTion Pro/Live)
#   * \author Nico Blodow <blodow@cs.tum.edu>, Suat Gedikli <gedikli@willowgarage.com>
#   * \ingroup io
#   */
# class PCL_EXPORTS OpenNIGrabber : public Grabber
cdef extern from "pcl/io/openni_grabber.h" namespace "pcl":
    cdef cppclass OpenNIGrabber(Grabber):
        # OpenNIGrabber ()
        OpenNIGrabber (string device_id, Mode2 depth_mode, Mode2 image_mode)
        # OpenNIGrabber (const std::string& device_id = "",
        #                const Mode& depth_mode = OpenNI_Default_Mode,
        #                const Mode& image_mode = OpenNI_Default_Mode);
        # public:
        # //define callback signature typedefs
        # typedef void (sig_cb_openni_image) (const boost::shared_ptr<openni_wrapper::Image>&);
        # typedef void (sig_cb_openni_depth_image) (const boost::shared_ptr<openni_wrapper::DepthImage>&);
        # typedef void (sig_cb_openni_ir_image) (const boost::shared_ptr<openni_wrapper::IRImage>&);
        # typedef void (sig_cb_openni_image_depth_image) (const boost::shared_ptr<openni_wrapper::Image>&, const boost::shared_ptr<openni_wrapper::DepthImage>&, float constant) ;
        # typedef void (sig_cb_openni_ir_depth_image) (const boost::shared_ptr<openni_wrapper::IRImage>&, const boost::shared_ptr<openni_wrapper::DepthImage>&, float constant) ;
        # typedef void (sig_cb_openni_point_cloud) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&);
        # typedef void (sig_cb_openni_point_cloud_rgb) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGB> >&);
        # typedef void (sig_cb_openni_point_cloud_rgba) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA> >&);
        # typedef void (sig_cb_openni_point_cloud_i) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&);
        # typedef void (sig_cb_openni_point_cloud_eigen) (const boost::shared_ptr<const pcl::PointCloud<Eigen::MatrixXf> >&);
        # public:
        # /** \brief Start the data acquisition. */
        void start ()
        
        # /** \brief Stop the data acquisition. */
        void stop ()
        
        # /** \brief Check if the data acquisition is still running. */
        bool isRunning ()
        
        string getName ()
        
        # /** \brief Obtain the number of frames per second (FPS). */
        float getFramesPerSecond () const
        
        # /** \brief Get a boost shared pointer to the \ref OpenNIDevice object. */
        # inline shared_ptr[openni_wrapper::OpenNIDevice] getDevice () const;
        # /** \brief Obtain a list of the available depth modes that this device supports. */
        # vector[pair[int, XnMapOutputMode] ] getAvailableDepthModes ()
        # /** \brief Obtain a list of the available image modes that this device supports. */
        # vector[pair[int, XnMapOutputMode] ] getAvailableImageModes ()
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
# 
# cdef extern from "pcl/io/openni_grabber.h" namespace "pcl":
#   cdef boost::shared_ptr<openni_wrapper::OpenNIDevice>
# cdef extern from "pcl/io/openni_grabber.h" namespace "pcl":
#   cdef OpenNIGrabber::getDevice () const
###

# pcd_grabber.h
# namespace pcl
# /** \brief Base class for PCD file grabber.
#   * \ingroup io
#   */
# class PCL_EXPORTS PCDGrabberBase : public Grabber
cdef extern from "pcl/io/pcd_grabber.h" namespace "pcl":
    cdef cppclass PCDGrabberBase(Grabber):
        PCDGrabberBase ()
        # public:
        # /** \brief Constructor taking just one PCD file or one TAR file containing multiple PCD files.
        #   * \param[in] pcd_file path to the PCD file
        #   * \param[in] frames_per_second frames per second. If 0, start() functions like a trigger, publishing the next PCD in the list.
        #   * \param[in] repeat whether to play PCD file in an endless loop or not.
        #   */
        # PCDGrabberBase (const std::string& pcd_file, float frames_per_second, bool repeat);
        # 
        # /** \brief Constructor taking a list of paths to PCD files, that are played in the order they appear in the list.
        #   * \param[in] pcd_files vector of paths to PCD files.
        #   * \param[in] frames_per_second frames per second. If 0, start() functions like a trigger, publishing the next PCD in the list.
        #   * \param[in] repeat whether to play PCD file in an endless loop or not.
        #   */
        # PCDGrabberBase (const std::vector<std::string>& pcd_files, float frames_per_second, bool repeat);
        # 
        # /** \brief Copy constructor.
        #   * \param[in] src the PCD Grabber base object to copy into this
        #   */
        # PCDGrabberBase (const PCDGrabberBase &src) : impl_ ()
        # /** \brief Copy operator.
        #   * \param[in] src the PCD Grabber base object to copy into this
        #   */
        # PCDGrabberBase&
        # operator = (const PCDGrabberBase &src)
        # {
        #   impl_ = src.impl_;
        #   return (*this);
        # }
        # 
        # /** \brief Virtual destructor. */
        # virtual ~PCDGrabberBase () throw ();
        # 
        # /** \brief Starts playing the list of PCD files if frames_per_second is > 0. Otherwise it works as a trigger: publishes only the next PCD file in the list. */
        # virtual void 
        # start ();
        # 
        # /** \brief Stops playing the list of PCD files if frames_per_second is > 0. Otherwise the method has no effect. */
        # virtual void 
        # stop ();
        # 
        # /** \brief Triggers a callback with new data */
        # virtual void 
        # trigger ();
        # 
        # /** \brief whether the grabber is started (publishing) or not.
        #   * \return true only if publishing.
        #   */
        # virtual bool 
        # isRunning () const;
        # 
        # /** \return The name of the grabber */
        # virtual std::string 
        # getName () const;
        # 
        # /** \brief Rewinds to the first PCD file in the list.*/
        # virtual void 
        # rewind ();
        # 
        # /** \brief Returns the frames_per_second. 0 if grabber is trigger-based */
        # virtual float 
        # getFramesPerSecond () const;
        # 
        # /** \brief Returns whether the repeat flag is on */
        # bool 
        # isRepeatOn () const;


###

# template <typename PointT> class PCDGrabber : public PCDGrabberBase
cdef extern from "pcl/io/pcd_grabber.h" namespace "pcl":
    cdef cppclass PCDGrabber[T](PCDGrabberBase):
        PCDGrabber ()
        # PCDGrabber (const std::string& pcd_path, float frames_per_second = 0, bool repeat = false);
        # PCDGrabber (const std::vector<std::string>& pcd_files, float frames_per_second = 0, bool repeat = false);
        # protected:
        # virtual void publish (const sensor_msgs::PointCloud2& blob, const Eigen::Vector4f& origin, const Eigen::Quaternionf& orientation) const;
        # boost::signals2::signal<void (const boost::shared_ptr<const pcl::PointCloud<PointT> >&)>* signal_;
        # #ifdef HAVE_OPENNI
        #   boost::signals2::signal<void (const boost::shared_ptr<openni_wrapper::DepthImage>&)>*     depth_image_signal_;
        # # #endif
# 
#   ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#   template<typename PointT>
#   PCDGrabber<PointT>::PCDGrabber (const std::string& pcd_path, float frames_per_second, bool repeat)
#   : PCDGrabberBase (pcd_path, frames_per_second, repeat)
#   {
#     signal_ = createSignal<void (const boost::shared_ptr<const pcl::PointCloud<PointT> >&)>();
# #ifdef HAVE_OPENNI
#     depth_image_signal_ = createSignal <void (const boost::shared_ptr<openni_wrapper::DepthImage>&)> ();
# #endif
#   }
# 
#   ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#   template<typename PointT>
#   PCDGrabber<PointT>::PCDGrabber (const std::vector<std::string>& pcd_files, float frames_per_second, bool repeat)
#     : PCDGrabberBase (pcd_files, frames_per_second, repeat), signal_ ()
#   {
#     signal_ = createSignal<void (const boost::shared_ptr<const pcl::PointCloud<PointT> >&)>();
# #ifdef HAVE_OPENNI
#     depth_image_signal_ = createSignal <void (const boost::shared_ptr<openni_wrapper::DepthImage>&)> ();
# #endif
#   }
# 
#   ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#   template<typename PointT> void 
#   PCDGrabber<PointT>::publish (const sensor_msgs::PointCloud2& blob, const Eigen::Vector4f& origin, const Eigen::Quaternionf& orientation) const
#   {
#     typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT> ());
#     pcl::fromROSMsg (blob, *cloud);
#     cloud->sensor_origin_ = origin;
#     cloud->sensor_orientation_ = orientation;
# 
#     signal_->operator () (cloud);
# 
# #ifdef HAVE_OPENNI
#     // If dataset is not organized, return
#     if (!cloud->isOrganized ())
#       return;
# 
#     boost::shared_ptr<xn::DepthMetaData> depth_meta_data (new xn::DepthMetaData);
#     depth_meta_data->AllocateData (cloud->width, cloud->height);
#     XnDepthPixel* depth_map = depth_meta_data->WritableData ();
#     uint32_t k = 0;
#     for (uint32_t i = 0; i < cloud->height; ++i)
#       for (uint32_t j = 0; j < cloud->width; ++j)
#       {
#         depth_map[k] = static_cast<XnDepthPixel> ((*cloud)[k].z * 1000);
#         ++k;
#       }
# 
#     boost::shared_ptr<openni_wrapper::DepthImage> depth_image (new openni_wrapper::DepthImage (depth_meta_data, 0.075f, 525, 0, 0));
#     if (depth_image_signal_->num_slots() > 0)
#       depth_image_signal_->operator()(depth_image);
# #endif
###

###############################################################################
# Enum
###############################################################################

# cdef extern from "pcl/io/openni_grabber.h" namespace "pcl":
#     cdef cppclass OpenNIGrabber(Grabber):
#         # public:
#         # typedef enum
#         #   OpenNI_Default_Mode = 0, // This can depend on the device. For now all devices (PSDK, Xtion, Kinect) its VGA@30Hz
#         #   OpenNI_SXGA_15Hz = 1,    // Only supported by the Kinect
#         #   OpenNI_VGA_30Hz = 2,     // Supported by PSDK, Xtion and Kinect
#         #   OpenNI_VGA_25Hz = 3,     // Supportged by PSDK and Xtion
#         #   OpenNI_QVGA_25Hz = 4,    // Supported by PSDK and Xtion
#         #   OpenNI_QVGA_30Hz = 5,    // Supported by PSDK, Xtion and Kinect
#         #   OpenNI_QVGA_60Hz = 6,    // Supported by PSDK and Xtion
#         #   OpenNI_QQVGA_25Hz = 7,   // Not supported -> using software downsampling (only for integer scale factor and only NN)
#         #   OpenNI_QQVGA_30Hz = 8,   // Not supported -> using software downsampling (only for integer scale factor and only NN)
#         #   OpenNI_QQVGA_60Hz = 9    // Not supported -> using software downsampling (only for integer scale factor and only NN)
#         # } Mode;
cdef extern from "pcl/io/openni_grabber.h" namespace "pcl":
    ctypedef enum Mode2 "pcl::OpenNIGrabber":
        Grabber_OpenNI_Default_Mode "pcl::OpenNIGrabber::OpenNI_Default_Mode"   # = 0, // This can depend on the device. For now all devices (PSDK, Xtion, Kinect) its VGA@30Hz
        Grabber_OpenNI_SXGA_15Hz "pcl::OpenNIGrabber::OpenNI_SXGA_15Hz"         # = 1, // Only supported by the Kinect
        Grabber_OpenNI_VGA_30Hz "pcl::OpenNIGrabber::OpenNI_VGA_30Hz"           # = 2, // Supported by PSDK, Xtion and Kinect
        Grabber_OpenNI_VGA_25Hz "pcl::OpenNIGrabber::OpenNI_VGA_25Hz"           # = 3, // Supportged by PSDK and Xtion
        Grabber_OpenNI_QVGA_25Hz "pcl::OpenNIGrabber::OpenNI_QVGA_25Hz"         # = 4, // Supported by PSDK and Xtion
        Grabber_OpenNI_QVGA_30Hz "pcl::OpenNIGrabber::OpenNI_QVGA_30Hz"         # = 5, // Supported by PSDK, Xtion and Kinect
        Grabber_OpenNI_QVGA_60Hz "pcl::OpenNIGrabber::OpenNI_QVGA_60Hz"         # = 6, // Supported by PSDK and Xtion
        Grabber_OpenNI_QQVGA_25Hz "pcl::OpenNIGrabber::OpenNI_QQVGA_25Hz"       # = 7, // Not supported -> using software downsampling (only for integer scale factor and only NN)
        Grabber_OpenNI_QQVGA_30Hz "pcl::OpenNIGrabber::OpenNI_QQVGA_30Hz"       # = 8, // Not supported -> using software downsampling (only for integer scale factor and only NN)
        Grabber_OpenNI_QQVGA_60Hz "pcl::OpenNIGrabber::OpenNI_QQVGA_60Hz"       # = 9  // Not supported -> using software downsampling (only for integer scale factor and only NN)


