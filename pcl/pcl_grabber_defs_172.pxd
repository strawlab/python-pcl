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

# Syncronization contains mutex
# oni_grabber.h defined Syncronization protected member
# http://stackoverflow.com/questions/9284352/boost-mutex-strange-error-with-private-member
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

# Syncronization contains mutex
# oni_grabber.h defined Syncronization protected member
# openni_grabber.h
# namespace pcl
#   struct PointXYZ;
#   struct PointXYZRGB;
#   struct PointXYZRGBA;
#   struct PointXYZI;
#   template <typename T> class PointCloud;
###
# openni_grabber.h
# /** \brief Grabber for OpenNI devices (i.e., Primesense PSDK, Microsoft Kinect, Asus XTion Pro/Live)
#   * \author Nico Blodow <blodow@cs.tum.edu>, Suat Gedikli <gedikli@willowgarage.com>
#   * \ingroup io
#   */
# class PCL_EXPORTS OpenNIGrabber : public Grabber
# cdef extern from "pcl/io/openni_grabber.h" namespace "pcl":
#     cdef cppclass OpenNIGrabber(Grabber):
        # OpenNIGrabber ()
        # OpenNIGrabber (string device_id, Mode2 depth_mode, Mode2 image_mode)
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
        # void start ()
        # 
        # /** \brief Stop the data acquisition. */
        # void stop ()
        # 
        # /** \brief Check if the data acquisition is still running. */
        # bool isRunning ()
        # 
        # string getName ()
        # 
        # /** \brief Obtain the number of frames per second (FPS). */
        # float getFramesPerSecond () const
        # 
        # /** \brief Get a boost shared pointer to the \ref OpenNIDevice object. */
        # inline shared_ptr[openni_wrapper::OpenNIDevice] getDevice () const;
        # /** \brief Obtain a list of the available depth modes that this device supports. */
        # vector[pair[int, XnMapOutputMode] ] getAvailableDepthModes ()
        # /** \brief Obtain a list of the available image modes that this device supports. */
        # vector[pair[int, XnMapOutputMode] ] getAvailableImageModes ()
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW


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
        # PCDGrabberBase& operator = (const PCDGrabberBase &src)
        # 
        # /** \brief Virtual destructor. */
        # virtual ~PCDGrabberBase () throw ();
        # 
        # /** \brief Starts playing the list of PCD files if frames_per_second is > 0. Otherwise it works as a trigger: publishes only the next PCD file in the list. */
        # virtual void start ();
        # 
        # /** \brief Stops playing the list of PCD files if frames_per_second is > 0. Otherwise the method has no effect. */
        # virtual void stop ();
        # 
        # /** \brief Triggers a callback with new data */
        # virtual void trigger ();
        # 
        # /** \brief whether the grabber is started (publishing) or not.
        #   * \return true only if publishing.
        #   */
        # virtual bool isRunning () const;
        # 
        # /** \return The name of the grabber */
        # virtual std::string getName () const;
        # 
        # /** \brief Rewinds to the first PCD file in the list.*/
        # virtual void rewind ();
        # 
        # /** \brief Returns the frames_per_second. 0 if grabber is trigger-based */
        # virtual float getFramesPerSecond () const;
        # 
        # /** \brief Returns whether the repeat flag is on */
        # bool isRepeatOn () const;


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


###

# template<typename PointT>
# PCDGrabber<PointT>::PCDGrabber (const std::string& pcd_path, float frames_per_second, bool repeat)
# : PCDGrabberBase (pcd_path, frames_per_second, repeat)
###

# template<typename PointT>
# PCDGrabber<PointT>::PCDGrabber (const std::vector<std::string>& pcd_files, float frames_per_second, bool repeat)
#   : PCDGrabberBase (pcd_files, frames_per_second, repeat), signal_ ()
###

# template<typename PointT> void 
# PCDGrabber<PointT>::publish (const sensor_msgs::PointCloud2& blob, const Eigen::Vector4f& origin, const Eigen::Quaternionf& orientation) const
###

# file_grabber.h
# namespace pcl
# /** \brief FileGrabber provides a container-style interface for grabbers which operate on fixed-size input
#   * \author Stephen Miller
#   * \ingroup io
#   */
# template <typename PointT>
# class PCL_EXPORTS FileGrabber
# cdef extern from "pcl/io/file_grabber.h" namespace "pcl":
#     cdef cppclass FileGrabber:
#         FileGrabber()
        # public:
        # /** \brief Empty destructor */
        # virtual ~FileGrabber () {}
        # 
        # /** \brief operator[] Returns the idx-th cloud in the dataset, without bounds checking.
        #  *  Note that in the future, this could easily be modified to do caching
        #  *  \param[in] idx The frame to load
        # */
        # virtual const boost::shared_ptr< const pcl::PointCloud<PointT> > operator[] (size_t idx) const = 0;
        # 
        # /** \brief size Returns the number of clouds currently loaded by the grabber */
        # virtual size_t size () const = 0;
        # 
        # /** \brief at Returns the idx-th cloud in the dataset, with bounds checking
        #  *  \param[in] idx The frame to load
        # */
        # virtual const boost::shared_ptr< const pcl::PointCloud<PointT> > at (size_t idx) const


###

# fotonic_grabber.h
# namespace pcl
# 
# struct PointXYZ;
# struct PointXYZRGB;
# struct PointXYZRGBA;
# struct PointXYZI;
# template <typename T> class PointCloud;
# 
# fotonic_grabber.h
# namespace pcl
# /** \brief Grabber for Fotonic devices
#   * \author Stefan Holzer <holzers@in.tum.de>
#   * \ingroup io
#   */
# class PCL_EXPORTS FotonicGrabber : public Grabber
# cdef extern from "pcl/io/fotonic_grabber.h" namespace "pcl":
#     cdef cppclass FotonicGrabber(Grabber):
#         FotonicGrabber()
        # public:
        # typedef enum
        # {
        #   Fotonic_Default_Mode = 0, // This can depend on the device. For now all devices (PSDK, Xtion, Kinect) its VGA@30Hz
        # } Mode;
        # 
        # //define callback signature typedefs
        # typedef void (sig_cb_fotonic_point_cloud) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&);
        # typedef void (sig_cb_fotonic_point_cloud_rgb) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGB> >&);
        # typedef void (sig_cb_fotonic_point_cloud_rgba) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA> >&);
        # typedef void (sig_cb_fotonic_point_cloud_i) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&);
        # 
        # public:
        # /** \brief Constructor
        #   * \param[in] device_id ID of the device, which might be a serial number, bus@address or the index of the device.
        #   * \param[in] depth_mode the mode of the depth stream
        #   * \param[in] image_mode the mode of the image stream
        #   */
        # FotonicGrabber (const FZ_DEVICE_INFO& device_info,
        #                 const Mode& depth_mode = Fotonic_Default_Mode,
        #                 const Mode& image_mode = Fotonic_Default_Mode);
        # 
        # /** \brief virtual Destructor inherited from the Grabber interface. It never throws. */
        # virtual ~FotonicGrabber () throw ();
        # 
        # /** \brief Initializes the Fotonic API. */
        # static void initAPI ();
        # 
        # /** \brief Exits the Fotonic API. */
        # static void exitAPI ();
        # 
        # /** \brief Searches for available devices. */
        # static std::vector<FZ_DEVICE_INFO> enumDevices ();
        # 
        # /** \brief Start the data acquisition. */
        # virtual void start ();
        # 
        # /** \brief Stop the data acquisition. */
        # virtual void stop ();
        # 
        # /** \brief Check if the data acquisition is still running. */
        # virtual bool isRunning () const;
        # 
        # virtual std::string getName () const;
        # 
        # /** \brief Obtain the number of frames per second (FPS). */
        # virtual float getFramesPerSecond () const;
        # 
        # protected:
        # /** \brief On initialization processing. */
        # void onInit (const FZ_DEVICE_INFO& device_info, const Mode& depth_mode, const Mode& image_mode);
        # /** \brief Sets up an OpenNI device. */
        # void setupDevice (const FZ_DEVICE_INFO& device_info, const Mode& depth_mode, const Mode& image_mode);
        # /** \brief Continously asks for data from the device and publishes it if available. */
        # void processGrabbing ();
        # boost::signals2::signal<sig_cb_fotonic_point_cloud>* point_cloud_signal_;
        # //boost::signals2::signal<sig_cb_fotonic_point_cloud_i>* point_cloud_i_signal_;
        # boost::signals2::signal<sig_cb_fotonic_point_cloud_rgb>* point_cloud_rgb_signal_;
        # boost::signals2::signal<sig_cb_fotonic_point_cloud_rgba>* point_cloud_rgba_signal_;
        # 
        # protected:
        # bool running_;
        # FZ_Device_Handle_t * fotonic_device_handle_;
        # boost::thread grabber_thread_;
        # 
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW


###

# hdl_grabber.h
# #define HDL_Grabber_toRadians(x) ((x) * M_PI / 180.0)
# 
# hdl_grabber.h
# namespace pcl
# /** \brief Grabber for the Velodyne High-Definition-Laser (HDL)
#  * \author Keven Ring <keven@mitre.org>
#  * \ingroup io
#  */
# class PCL_EXPORTS HDLGrabber : public Grabber
# cdef extern from "pcl/io/hdl_grabber.h" namespace "pcl":
#     cdef cppclass HDLGrabber(Grabber):
#         HDLGrabber()
        # public:
        # /** \brief Signal used for a single sector
        #  *         Represents 1 corrected packet from the HDL Velodyne
        #  */
        # typedef void (sig_cb_velodyne_hdl_scan_point_cloud_xyz) (
        #     const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&,
        #     float, float);
        # /** \brief Signal used for a single sector
        #  *         Represents 1 corrected packet from the HDL Velodyne.  Each laser has a different RGB
        #  */
        # typedef void (sig_cb_velodyne_hdl_scan_point_cloud_xyzrgb) (
        #     const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA> >&,
        #     float, float);
        # /** \brief Signal used for a single sector
        #  *         Represents 1 corrected packet from the HDL Velodyne with the returned intensity.
        #  */
        # typedef void (sig_cb_velodyne_hdl_scan_point_cloud_xyzi) (
        #     const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&,
        #     float startAngle, float);
        # /** \brief Signal used for a 360 degree sweep
        #  *         Represents multiple corrected packets from the HDL Velodyne
        #  *         This signal is sent when the Velodyne passes angle "0"
        #  */
        # typedef void (sig_cb_velodyne_hdl_sweep_point_cloud_xyz) (
        #     const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&);
        # /** \brief Signal used for a 360 degree sweep
        #  *         Represents multiple corrected packets from the HDL Velodyne with the returned intensity
        #  *         This signal is sent when the Velodyne passes angle "0"
        #  */
        # typedef void (sig_cb_velodyne_hdl_sweep_point_cloud_xyzi) (
        #     const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&);
        # /** \brief Signal used for a 360 degree sweep
        #  *         Represents multiple corrected packets from the HDL Velodyne
        #  *         This signal is sent when the Velodyne passes angle "0".  Each laser has a different RGB
        #  */
        # typedef void (sig_cb_velodyne_hdl_sweep_point_cloud_xyzrgb) (
        #     const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA> >&);
        # 
        # /** \brief Constructor taking an optional path to an HDL corrections file.  The Grabber will listen on the default IP/port for data packets [192.168.3.255/2368]
        #  * \param[in] correctionsFile Path to a file which contains the correction parameters for the HDL.  This parameter is mandatory for the HDL-64, optional for the HDL-32
        #  * \param[in] pcapFile Path to a file which contains previously captured data packets.  This parameter is optional
        #  */
        # HDLGrabber (const std::string& correctionsFile = "", const std::string& pcapFile = "");
        # 
        # /** \brief Constructor taking a pecified IP/port and an optional path to an HDL corrections file.
        #  * \param[in] ipAddress IP Address that should be used to listen for HDL packets
        #  * \param[in] port UDP Port that should be used to listen for HDL packets
        #  * \param[in] correctionsFile Path to a file which contains the correction parameters for the HDL.  This field is mandatory for the HDL-64, optional for the HDL-32
        #  */
        # HDLGrabber (const boost::asio::ip::address& ipAddress, const unsigned short port, const std::string& correctionsFile = "");
        # 
        # /** \brief virtual Destructor inherited from the Grabber interface. It never throws. */
        # virtual ~HDLGrabber () throw ();
        # 
        # /** \brief Starts processing the Velodyne packets, either from the network or PCAP file. */
        # virtual void start ();
        # 
        # /** \brief Stops processing the Velodyne packets, either from the network or PCAP file */
        # virtual void stop ();
        # 
        # /** \brief Obtains the name of this I/O Grabber
        #  *  \return The name of the grabber
        #  */
        # virtual std::string getName () const;
        # 
        # /** \brief Check if the grabber is still running.
        #  *  \return TRUE if the grabber is running, FALSE otherwise
        #  */
        # virtual bool isRunning () const;
        # 
        # /** \brief Returns the number of frames per second.
        #  */
        # virtual float getFramesPerSecond () const;
        # 
        # /** \brief Allows one to filter packets based on the SOURCE IP address and PORT
        #  *         This can be used, for instance, if multiple HDL LIDARs are on the same network
        #  */
        # void filterPackets (const boost::asio::ip::address& ipAddress, const unsigned short port = 443);
        # 
        # /** \brief Allows one to customize the colors used for each of the lasers.
        #  */
        # void setLaserColorRGB (const pcl::RGB& color, unsigned int laserNumber);
        # 
        # /** \brief Any returns from the HDL with a distance less than this are discarded.
        #  *         This value is in meters
        #  *         Default: 0.0
        #  */
        # void setMinimumDistanceThreshold(float & minThreshold);
        # 
        # /** \brief Any returns from the HDL with a distance greater than this are discarded.
        #  *         This value is in meters
        #  *         Default: 10000.0
        #  */
        # void setMaximumDistanceThreshold(float & maxThreshold);
        # 
        # /** \brief Returns the current minimum distance threshold, in meters
        #  */
        # float getMinimumDistanceThreshold();
        # 
        # /** \brief Returns the current maximum distance threshold, in meters
        #  */
        # float getMaximumDistanceThreshold();
        # 
        # protected:
        # static const int HDL_DATA_PORT = 2368;
        # static const int HDL_NUM_ROT_ANGLES = 36001;
        # static const int HDL_LASER_PER_FIRING = 32;
        # static const int HDL_MAX_NUM_LASERS = 64;
        # static const int HDL_FIRING_PER_PKT = 12;
        # static const boost::asio::ip::address HDL_DEFAULT_NETWORK_ADDRESS;
        # 
        # enum HDLBlock
        # {
        #   BLOCK_0_TO_31 = 0xeeff, BLOCK_32_TO_63 = 0xddff
        # };
        # 
        # #pragma pack(push, 1)
        # typedef struct HDLLaserReturn
        # {
        #     unsigned short distance;
        #     unsigned char intensity;
        # } HDLLaserReturn;
        # #pragma pack(pop)
        # 
        # struct HDLFiringData
        # {
        #     unsigned short blockIdentifier;
        #     unsigned short rotationalPosition;
        #     HDLLaserReturn laserReturns[HDL_LASER_PER_FIRING];
        # };
        # 
        # struct HDLDataPacket
        # {
        #     HDLFiringData firingData[HDL_FIRING_PER_PKT];
        #     unsigned int gpsTimestamp;
        #     unsigned char blank1;
        #     unsigned char blank2;
        # };
        # 
        # struct HDLLaserCorrection
        # {
        #     double azimuthCorrection;
        #     double verticalCorrection;
        #     double distanceCorrection;
        #     double verticalOffsetCorrection;
        #     double horizontalOffsetCorrection;
        #     double sinVertCorrection;
        #     double cosVertCorrection;
        #     double sinVertOffsetCorrection;
        #     double cosVertOffsetCorrection;
        # };
        # 


###

# image_grabber.h
# namespace pcl
# /** \brief Base class for Image file grabber.
#  * \ingroup io
#  */
# class PCL_EXPORTS ImageGrabberBase : public Grabber
# cdef extern from "pcl/io/image_grabber.h" namespace "pcl":
#     cdef cppclass ImageGrabberBase(Grabber):
#         ImageGrabberBase()
        # public:
        # /** \brief Constructor taking a folder of depth+[rgb] images.
        # * \param[in] directory Directory which contains an ordered set of images corresponding to an [RGB]D video, stored as TIFF, PNG, JPG, or PPM files. The naming convention is: frame_[timestamp]_["depth"/"rgb"].[extension]
        # * \param[in] frames_per_second frames per second. If 0, start() functions like a trigger, publishing the next PCD in the list.
        # * \param[in] repeat whether to play PCD file in an endless loop or not.
        # * \param pclzf_mode
        # */
        # ImageGrabberBase (const std::string& directory, float frames_per_second, bool repeat, bool pclzf_mode);
        # 
        # ImageGrabberBase (const std::string& depth_directory, const std::string& rgb_directory, float frames_per_second, bool repeat);
        # /** \brief Constructor taking a list of paths to PCD files, that are played in the order they appear in the list.
        # * \param[in] depth_image_files Path to the depth image files files.
        # * \param[in] frames_per_second frames per second. If 0, start() functions like a trigger, publishing the next PCD in the list.
        # * \param[in] repeat whether to play PCD file in an endless loop or not.
        # */
        # ImageGrabberBase (const std::vector<std::string>& depth_image_files, float frames_per_second, bool repeat);
        # 
        # /** \brief Copy constructor.
        # * \param[in] src the Image Grabber base object to copy into this
        # */
        # ImageGrabberBase (const ImageGrabberBase &src) : Grabber (), impl_ ()
        # 
        # /** \brief Copy operator.
        #  * \param[in] src the Image Grabber base object to copy into this
        #  */
        # ImageGrabberBase& operator = (const ImageGrabberBase &src)
        # 
        # /** \brief Virtual destructor. */
        # virtual ~ImageGrabberBase () throw ();
        # 
        # /** \brief Starts playing the list of PCD files if frames_per_second is > 0. Otherwise it works as a trigger: publishes only the next PCD file in the list. */
        # virtual void start ();
        # 
        # /** \brief Stops playing the list of PCD files if frames_per_second is > 0. Otherwise the method has no effect. */
        # virtual void  stop ();
        # 
        # /** \brief Triggers a callback with new data */
        # virtual void trigger ();
        # 
        # /** \brief whether the grabber is started (publishing) or not.
        #  * \return true only if publishing.
        #  */
        # virtual bool isRunning () const;
        # 
        # /** \return The name of the grabber */
        # virtual std::string getName () const;
        # 
        # /** \brief Rewinds to the first PCD file in the list.*/
        # virtual void rewind ();
        # 
        # /** \brief Returns the frames_per_second. 0 if grabber is trigger-based */
        # virtual float getFramesPerSecond () const;
        # 
        # /** \brief Returns whether the repeat flag is on */
        # bool isRepeatOn () const;
        # 
        # /** \brief Returns if the last frame is reached */
        # bool atLastFrame () const;
        # 
        # /** \brief Returns the filename of the current indexed file */
        # std::string getCurrentDepthFileName () const;
        # 
        # /** \brief Returns the filename of the previous indexed file 
        #  *  SDM: adding this back in, but is this useful, or confusing? */
        # std::string getPrevDepthFileName () const;
        # 
        # /** \brief Get the depth filename at a particular index */
        # std::string getDepthFileNameAtIndex (size_t idx) const;
        # 
        # /** \brief Query only the timestamp of an index, if it exists */
        # bool getTimestampAtIndex (size_t idx, pcl::uint64_t &timestamp) const;
        # 
        # /** \brief Manually set RGB image files.
        # * \param[in] rgb_image_files A vector of [tiff/png/jpg/ppm] files to use as input. There must be a 1-to-1 correspondence between these and the depth images you set
        # */
        # void setRGBImageFiles (const std::vector<std::string>& rgb_image_files);
        # 
        # /** \brief Define custom focal length and center pixel. This will override ANY other setting of parameters for the duration of the grabber's life, whether by factory defaults or explicitly read from a frame_[timestamp].xml file. 
        # *  \param[in] focal_length_x Horizontal focal length (fx)
        # *  \param[in] focal_length_y Vertical focal length (fy)
        # *  \param[in] principal_point_x Horizontal coordinates of the principal point (cx)
        # *  \param[in] principal_point_y Vertical coordinates of the principal point (cy)
        # */
        # virtual void
        # setCameraIntrinsics (const double focal_length_x, 
        #                    const double focal_length_y, 
        #                    const double principal_point_x, 
        #                    const double principal_point_y);
        # 
        # /** \brief Get the current focal length and center pixel. If the intrinsics have been manually set with setCameraIntrinsics, this will return those values. Else, if start () has been called and the grabber has found a frame_[timestamp].xml file, this will return the most recent values read. Else, returns factory defaults.
        #  *  \param[out] focal_length_x Horizontal focal length (fx)
        #  *  \param[out] focal_length_y Vertical focal length (fy)
        #  *  \param[out] principal_point_x Horizontal coordinates of the principal point (cx)
        #  *  \param[out] principal_point_y Vertical coordinates of the principal point (cy)
        #  */
        # virtual void
        # getCameraIntrinsics (double &focal_length_x, 
        #                    double &focal_length_y, 
        #                    double &principal_point_x, 
        #                    double &principal_point_y) const;
        # 
        # /** \brief Define the units the depth data is stored in.
        # *  Defaults to mm (0.001), meaning a brightness of 1000 corresponds to 1 m*/
        # void setDepthImageUnits (float units);
        # 
        # /** \brief Set the number of threads, if we wish to use OpenMP for quicker cloud population.
        # *  Note that for a standard (< 4 core) machine this is unlikely to yield a drastic speedup.*/
        # void setNumberOfThreads (unsigned int nr_threads = 0);
        # 
        # protected:
        # /** \brief Convenience function to see how many frames this consists of
        # */
        # size_t numFrames () const;
        # 
        # /** \brief Gets the cloud in ROS form at location idx */
        # bool getCloudAt (size_t idx, pcl::PCLPointCloud2 &blob, Eigen::Vector4f &origin, Eigen::Quaternionf &orientation) const;


###

# template <typename T> class PointCloud;
# image_grabber.h
# namespace pcl
# template <typename PointT> class ImageGrabber : public ImageGrabberBase, public FileGrabber<PointT>
# cdef extern from "pcl/io/image_grabber.h" namespace "pcl":
#     cdef cppclass ImageGrabber(ImageGrabberBase, FileGrabber[T]):
#         ImageGrabber()
        # public:
        # ImageGrabber (const std::string& dir, 
        #             float frames_per_second = 0, 
        #             bool repeat = false, 
        #             bool pclzf_mode = false);
        # 
        # ImageGrabber (const std::string& depth_dir, 
        #             const std::string& rgb_dir, 
        #             float frames_per_second = 0, 
        #             bool repeat = false);
        # 
        # ImageGrabber (const std::vector<std::string>& depth_image_files, 
        #             float frames_per_second = 0, 
        #             bool repeat = false);
        # 
        # /** \brief Empty destructor */
        # virtual ~ImageGrabber () throw () {}
        # 
        # // Inherited from FileGrabber
        # const boost::shared_ptr< const pcl::PointCloud<PointT> > operator[] (size_t idx) const;
        # 
        # // Inherited from FileGrabber
        # size_t size () const;
        # 
        # protected:
        # virtual void publish (const pcl::PCLPointCloud2& blob,
        #        const Eigen::Vector4f& origin, 
        #        const Eigen::Quaternionf& orientation) const;
        # boost::signals2::signal<void (const boost::shared_ptr<const pcl::PointCloud<PointT> >&)>* signal_;


###

# image_grabber.h
# namespace pcl
# template<typename PointT>
# ImageGrabber<PointT>::ImageGrabber (const std::string& dir, 
#                                     float frames_per_second, 
#                                     bool repeat, 
#                                     bool pclzf_mode)
#   : ImageGrabberBase (dir, frames_per_second, repeat, pclzf_mode)
###

# image_grabber.h
# namespace pcl
# template<typename PointT>
# ImageGrabber<PointT>::ImageGrabber (const std::string& depth_dir, 
#                                     const std::string& rgb_dir, 
#                                     float frames_per_second, 
#                                     bool repeat)
#   : ImageGrabberBase (depth_dir, rgb_dir, frames_per_second, repeat)
###

# image_grabber.h
# namespace pcl
# template<typename PointT>
# ImageGrabber<PointT>::ImageGrabber (const std::vector<std::string>& depth_image_files, 
#                                     float frames_per_second, 
#                                     bool repeat)
#   : ImageGrabberBase (depth_image_files, frames_per_second, repeat), signal_ ()
###

# image_grabber.h
# namespace pcl
# template<typename PointT> const boost::shared_ptr< const pcl::PointCloud<PointT> >
# ImageGrabber<PointT>::operator[] (size_t idx) const
#     pcl::PCLPointCloud2 blob;
#     Eigen::Vector4f origin;
#     Eigen::Quaternionf orientation;
#     getCloudAt (idx, blob, origin, orientation);
#     typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT> ());
#     pcl::fromPCLPointCloud2 (blob, *cloud);
#     cloud->sensor_origin_ = origin;
#     cloud->sensor_orientation_ = orientation;
#     return (cloud);
###
 
# image_grabber.h
# namespace pcl
# template <typename PointT> size_t ImageGrabber<PointT>::size () const
###

# image_grabber.h
# namespace pcl
# template<typename PointT> void
# ImageGrabber<PointT>::publish (const pcl::PCLPointCloud2& blob, const Eigen::Vector4f& origin, const Eigen::Quaternionf& orientation) const
###

# openni2_grabber.h
# namespace pcl
# 
# struct PointXYZ;
# struct PointXYZRGB;
# struct PointXYZRGBA;
# struct PointXYZI;
# template <typename T> class PointCloud;
#
# openni2_grabber.h
# namespace pcl
# namespace io
# /** \brief Grabber for OpenNI 2 devices (i.e., Primesense PSDK, Microsoft Kinect, Asus XTion Pro/Live)
#   * \ingroup io
#   */
# class PCL_EXPORTS OpenNI2Grabber : public Grabber
# cdef extern from "pcl/io/openni2_grabber.h" namespace "pcl::io":
#     cdef cppclass OpenNI2Grabber(Grabber):
#         OpenNI2Grabber()
        # public:
        # typedef boost::shared_ptr<OpenNI2Grabber> Ptr;
        # typedef boost::shared_ptr<const OpenNI2Grabber> ConstPtr;
        # 
        # // Templated images
        # typedef pcl::io::DepthImage DepthImage;
        # typedef pcl::io::IRImage IRImage;
        # typedef pcl::io::Image Image;
        # 
        # /** \brief Basic camera parameters placeholder. */
        # struct CameraParameters
        #     /** fx */
        #     double focal_length_x;
        #     /** fy */
        #     double focal_length_y;
        #     /** cx */
        #     double principal_point_x;
        #     /** cy */
        #     double principal_point_y;
        #   
        #     CameraParameters (double initValue)
        #       : focal_length_x (initValue), focal_length_y (initValue),
        #       principal_point_x (initValue),  principal_point_y (initValue)
        #     {}
        #   
        #     CameraParameters (double fx, double fy, double cx, double cy)
        #       : focal_length_x (fx), focal_length_y (fy), principal_point_x (cx), principal_point_y (cy)
        #     { }
        ###
        # 
        # typedef enum
        # {
        #     OpenNI_Default_Mode = 0, // This can depend on the device. For now all devices (PSDK, Xtion, Kinect) its VGA@30Hz
        #     OpenNI_SXGA_15Hz = 1,    // Only supported by the Kinect
        #     OpenNI_VGA_30Hz = 2,     // Supported by PSDK, Xtion and Kinect
        #     OpenNI_VGA_25Hz = 3,     // Supportged by PSDK and Xtion
        #     OpenNI_QVGA_25Hz = 4,    // Supported by PSDK and Xtion
        #     OpenNI_QVGA_30Hz = 5,    // Supported by PSDK, Xtion and Kinect
        #     OpenNI_QVGA_60Hz = 6,    // Supported by PSDK and Xtion
        #     OpenNI_QQVGA_25Hz = 7,   // Not supported -> using software downsampling (only for integer scale factor and only NN)
        #     OpenNI_QQVGA_30Hz = 8,   // Not supported -> using software downsampling (only for integer scale factor and only NN)
        #     OpenNI_QQVGA_60Hz = 9    // Not supported -> using software downsampling (only for integer scale factor and only NN)
        # } Mode;
        # 
        # //define callback signature typedefs
        # typedef void (sig_cb_openni_image) (const boost::shared_ptr<Image>&);
        # typedef void (sig_cb_openni_depth_image) (const boost::shared_ptr<DepthImage>&);
        # typedef void (sig_cb_openni_ir_image) (const boost::shared_ptr<IRImage>&);
        # typedef void (sig_cb_openni_image_depth_image) (const boost::shared_ptr<Image>&, const boost::shared_ptr<DepthImage>&, float reciprocalFocalLength) ;
        # typedef void (sig_cb_openni_ir_depth_image) (const boost::shared_ptr<IRImage>&, const boost::shared_ptr<DepthImage>&, float reciprocalFocalLength) ;
        # typedef void (sig_cb_openni_point_cloud) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&);
        # typedef void (sig_cb_openni_point_cloud_rgb) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGB> >&);
        # typedef void (sig_cb_openni_point_cloud_rgba) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA> >&);
        # typedef void (sig_cb_openni_point_cloud_i) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&);
        # 
        # public:
        # /** \brief Constructor
        #   * \param[in] device_id ID of the device, which might be a serial number, bus@address or the index of the device.
        #   * \param[in] depth_mode the mode of the depth stream
        #   * \param[in] image_mode the mode of the image stream
        # */
        # OpenNI2Grabber (const std::string& device_id = "",
        #   const Mode& depth_mode = OpenNI_Default_Mode,
        #     const Mode& image_mode = OpenNI_Default_Mode);
        # 
        # /** \brief virtual Destructor inherited from the Grabber interface. It never throws. */
        # virtual ~OpenNI2Grabber () throw ();
        # 
        # /** \brief Start the data acquisition. */
        # virtual void start ();
        # 
        # /** \brief Stop the data acquisition. */
        # virtual void stop ();
        # 
        # /** \brief Check if the data acquisition is still running. */
        # virtual bool isRunning () const;
        # 
        # virtual std::string getName () const;
        # 
        # /** \brief Obtain the number of frames per second (FPS). */
        # virtual float getFramesPerSecond () const;
        # 
        # /** \brief Get a boost shared pointer to the \ref OpenNIDevice object. */
        # inline boost::shared_ptr<pcl::io::openni2::OpenNI2Device> getDevice () const;
        # 
        # /** \brief Obtain a list of the available depth modes that this device supports. */
        # std::vector<std::pair<int, pcl::io::openni2::OpenNI2VideoMode> > getAvailableDepthModes () const;
        # 
        # /** \brief Obtain a list of the available image modes that this device supports. */
        # std::vector<std::pair<int, pcl::io::openni2::OpenNI2VideoMode> > getAvailableImageModes () const;
        # 
        # /** \brief Set the RGB camera parameters (fx, fy, cx, cy)
        #   * \param[in] rgb_focal_length_x the RGB focal length (fx)
        #   * \param[in] rgb_focal_length_y the RGB focal length (fy)
        #   * \param[in] rgb_principal_point_x the RGB principal point (cx)
        #   * \param[in] rgb_principal_point_y the RGB principal point (cy)
        #   * Setting the parameters to non-finite values (e.g., NaN, Inf) invalidates them
        #   * and the grabber will use the default values from the camera instead.
        # */
        # inline void
        # setRGBCameraIntrinsics (const double rgb_focal_length_x,
        #     const double rgb_focal_length_y,
        #     const double rgb_principal_point_x,
        #     const double rgb_principal_point_y)
        # 
        # /** \brief Get the RGB camera parameters (fx, fy, cx, cy)
        #   * \param[out] rgb_focal_length_x the RGB focal length (fx)
        #   * \param[out] rgb_focal_length_y the RGB focal length (fy)
        #   * \param[out] rgb_principal_point_x the RGB principal point (cx)
        #   * \param[out] rgb_principal_point_y the RGB principal point (cy)
        #   */
        # inline void
        # getRGBCameraIntrinsics (double &rgb_focal_length_x,
        #     double &rgb_focal_length_y,
        #     double &rgb_principal_point_x,
        #     double &rgb_principal_point_y) const
        # 
        # /** \brief Set the RGB image focal length (fx = fy).
        #   * \param[in] rgb_focal_length the RGB focal length (assumes fx = fy)
        #   * Setting the parameter to a non-finite value (e.g., NaN, Inf) invalidates it
        #   * and the grabber will use the default values from the camera instead.
        #   * These parameters will be used for XYZRGBA clouds.
        #   */
        # inline void
        # setRGBFocalLength (const double rgb_focal_length)
        # 
        # /** \brief Set the RGB image focal length
        #   * \param[in] rgb_focal_length_x the RGB focal length (fx)
        #   * \param[in] rgb_focal_ulength_y the RGB focal length (fy)
        #   * Setting the parameters to non-finite values (e.g., NaN, Inf) invalidates them
        #   * and the grabber will use the default values from the camera instead.
        #   * These parameters will be used for XYZRGBA clouds.
        #   */
        # inline void
        # setRGBFocalLength (const double rgb_focal_length_x, const double rgb_focal_length_y)
        # 
        # /** \brief Return the RGB focal length parameters (fx, fy)
        #   * \param[out] rgb_focal_length_x the RGB focal length (fx)
        #   * \param[out] rgb_focal_length_y the RGB focal length (fy)
        #   */
        # inline void
        # getRGBFocalLength (double &rgb_focal_length_x, double &rgb_focal_length_y) const
        # 
        # /** \brief Set the Depth camera parameters (fx, fy, cx, cy)
        #   * \param[in] depth_focal_length_x the Depth focal length (fx)
        #   * \param[in] depth_focal_length_y the Depth focal length (fy)
        #   * \param[in] depth_principal_point_x the Depth principal point (cx)
        #   * \param[in] depth_principal_point_y the Depth principal point (cy)
        #   * Setting the parameters to non-finite values (e.g., NaN, Inf) invalidates them
        #   * and the grabber will use the default values from the camera instead.
        #   */
        # inline void
        # setDepthCameraIntrinsics (const double depth_focal_length_x,
        #     const double depth_focal_length_y,
        #     const double depth_principal_point_x,
        #     const double depth_principal_point_y)
        # 
        # /** \brief Get the Depth camera parameters (fx, fy, cx, cy)
        #   * \param[out] depth_focal_length_x the Depth focal length (fx)
        #   * \param[out] depth_focal_length_y the Depth focal length (fy)
        #   * \param[out] depth_principal_point_x the Depth principal point (cx)
        #   * \param[out] depth_principal_point_y the Depth principal point (cy)
        #   */
        # inline void
        # getDepthCameraIntrinsics (double &depth_focal_length_x,
        #     double &depth_focal_length_y,
        #     double &depth_principal_point_x,
        #     double &depth_principal_point_y) const
        # 
        # /** \brief Set the Depth image focal length (fx = fy).
        #   * \param[in] depth_focal_length the Depth focal length (assumes fx = fy)
        #   * Setting the parameter to a non-finite value (e.g., NaN, Inf) invalidates it
        #   * and the grabber will use the default values from the camera instead.
        #   */
        # inline void
        # setDepthFocalLength (const double depth_focal_length)
        # 
        # /** \brief Set the Depth image focal length
        #   * \param[in] depth_focal_length_x the Depth focal length (fx)
        #   * \param[in] depth_focal_length_y the Depth focal length (fy)
        #   * Setting the parameter to non-finite values (e.g., NaN, Inf) invalidates them
        #   * and the grabber will use the default values from the camera instead.
        #   */
        # inline void
        # setDepthFocalLength (const double depth_focal_length_x, const double depth_focal_length_y)
        # 
        # /** \brief Return the Depth focal length parameters (fx, fy)
        #   * \param[out] depth_focal_length_x the Depth focal length (fx)
        #   * \param[out] depth_focal_length_y the Depth focal length (fy)
        #   */
        # inline void
        # getDepthFocalLength (double &depth_focal_length_x, double &depth_focal_length_y) const
        # 
        # protected:
        # /** \brief Sets up an OpenNI device. */
        # void setupDevice (const std::string& device_id, const Mode& depth_mode, const Mode& image_mode);
        # 
        # /** \brief Update mode maps. */
        # void updateModeMaps ();
        # 
        # /** \brief Start synchronization. */
        # void startSynchronization ();
        # 
        # /** \brief Stop synchronization. */
        # void stopSynchronization ();
        # 
        # // TODO: rename to mapMode2OniMode
        # /** \brief Map config modes. */
        # bool mapMode2XnMode (int mode, pcl::io::openni2::OpenNI2VideoMode& videoMode) const;
        # 
        # // callback methods
        # /** \brief RGB image callback. */
        # virtual void imageCallback (pcl::io::openni2::Image::Ptr image, void* cookie);
        # 
        # /** \brief Depth image callback. */
        # virtual void depthCallback (pcl::io::openni2::DepthImage::Ptr depth_image, void* cookie);
        # 
        # /** \brief IR image callback. */
        # virtual void irCallback (pcl::io::openni2::IRImage::Ptr ir_image, void* cookie);
        # 
        # /** \brief RGB + Depth image callback. */
        # virtual void imageDepthImageCallback (const pcl::io::openni2::Image::Ptr &image, const pcl::io::openni2::DepthImage::Ptr &depth_image);
        # 
        # /** \brief IR + Depth image callback. */
        # virtual void irDepthImageCallback (const pcl::io::openni2::IRImage::Ptr &image, const pcl::io::openni2::DepthImage::Ptr &depth_image);
        # 
        # /** \brief Process changed signals. */
        # virtual void signalsChanged ();
        # 
        # // helper methods
        # /** \brief Check if the RGB and Depth images are required to be synchronized or not. */
        # virtual void checkImageAndDepthSynchronizationRequired ();
        # 
        # /** \brief Check if the RGB image stream is required or not. */
        # virtual void checkImageStreamRequired ();
        # 
        # /** \brief Check if the depth stream is required or not. */
        # virtual void checkDepthStreamRequired ();
        # 
        # /** \brief Check if the IR image stream is required or not. */
        # virtual void checkIRStreamRequired ();
        # 
        # // Point cloud conversion ///////////////////////////////////////////////
        # 
        # /** \brief Convert a Depth image to a pcl::PointCloud<pcl::PointXYZ>
        #   * \param[in] depth the depth image to convert
        #   */
        # boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> >
        # convertToXYZPointCloud (const pcl::io::openni2::DepthImage::Ptr &depth);
        # 
        # /** \brief Convert a Depth + RGB image pair to a pcl::PointCloud<PointT>
        #   * \param[in] image the RGB image to convert
        #   * \param[in] depth_image the depth image to convert
        #   */
        # template <typename PointT> typename pcl::PointCloud<PointT>::Ptr
        # convertToXYZRGBPointCloud (const pcl::io::openni2::Image::Ptr &image, const pcl::io::openni2::DepthImage::Ptr &depth_image);
        # 
        # /** \brief Convert a Depth + Intensity image pair to a pcl::PointCloud<pcl::PointXYZI>
        #   * \param[in] image the IR image to convert
        #   * \param[in] depth_image the depth image to convert
        #   */
        # boost::shared_ptr<pcl::PointCloud<pcl::PointXYZI> >
        # convertToXYZIPointCloud (const pcl::io::openni2::IRImage::Ptr &image, const pcl::io::openni2::DepthImage::Ptr &depth_image);
        # 
        # std::vector<uint8_t> color_resize_buffer_;
        # std::vector<uint16_t> depth_resize_buffer_;
        # std::vector<uint16_t> ir_resize_buffer_;
        # 
        # // Stream callbacks /////////////////////////////////////////////////////
        # void processColorFrame (openni::VideoStream& stream);
        # void processDepthFrame (openni::VideoStream& stream);
        # void processIRFrame (openni::VideoStream& stream);
        # 
        # Synchronizer<pcl::io::openni2::Image::Ptr, pcl::io::openni2::DepthImage::Ptr > rgb_sync_;
        # Synchronizer<pcl::io::openni2::IRImage::Ptr, pcl::io::openni2::DepthImage::Ptr > ir_sync_;
        # 
        # /** \brief The actual openni device. */
        # boost::shared_ptr<pcl::io::openni2::OpenNI2Device> device_;
        # 
        # std::string rgb_frame_id_;
        # std::string depth_frame_id_;
        # unsigned image_width_;
        # unsigned image_height_;
        # unsigned depth_width_;
        # unsigned depth_height_;
        # 
        # bool image_required_;
        # bool depth_required_;
        # bool ir_required_;
        # bool sync_required_;
        # 
        # boost::signals2::signal<sig_cb_openni_image>* image_signal_;
        # boost::signals2::signal<sig_cb_openni_depth_image>* depth_image_signal_;
        # boost::signals2::signal<sig_cb_openni_ir_image>* ir_image_signal_;
        # boost::signals2::signal<sig_cb_openni_image_depth_image>* image_depth_image_signal_;
        # boost::signals2::signal<sig_cb_openni_ir_depth_image>* ir_depth_image_signal_;
        # boost::signals2::signal<sig_cb_openni_point_cloud>* point_cloud_signal_;
        # boost::signals2::signal<sig_cb_openni_point_cloud_i>* point_cloud_i_signal_;
        # boost::signals2::signal<sig_cb_openni_point_cloud_rgb>* point_cloud_rgb_signal_;
        # boost::signals2::signal<sig_cb_openni_point_cloud_rgba>* point_cloud_rgba_signal_;
        # 
        # struct modeComp
        # {
        #     bool operator () (const openni::VideoMode& mode1, const openni::VideoMode & mode2) const
        # };
        # 
        # // Mapping from config (enum) modes to native OpenNI modes
        # std::map<int, pcl::io::openni2::OpenNI2VideoMode> config2oni_map_;
        # 
        # pcl::io::openni2::OpenNI2Device::CallbackHandle depth_callback_handle_;
        # pcl::io::openni2::OpenNI2Device::CallbackHandle image_callback_handle_;
        # pcl::io::openni2::OpenNI2Device::CallbackHandle ir_callback_handle_;
        # bool running_;
        # 
        # CameraParameters rgb_parameters_;
        # CameraParameters depth_parameters_;
        # 
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 
# boost::shared_ptr<pcl::io::openni2::OpenNI2Device>
# OpenNI2Grabber::getDevice () const


###

# pxc_grabber.h
# namespace pcl
# 
# struct PointXYZ;
# struct PointXYZRGB;
# struct PointXYZRGBA;
# struct PointXYZI;
# template <typename T> class PointCloud;
# 
# pxc_grabber.h
# namespace pcl
# /** \brief Grabber for PXC devices
#   * \author Stefan Holzer <holzers@in.tum.de>
#   * \ingroup io
#   */
# class PCL_EXPORTS PXCGrabber : public Grabber
# cdef extern from "pcl/io/pxc_grabber.h" namespace "pcl":
#     cdef cppclass PXCGrabber(Grabber):
#         PXCGrabber()
        # public:
        # 
        # /** \brief Supported modes for grabbing from a PXC device. */
        # typedef enum
        # {
        #   PXC_Default_Mode = 0, 
        # } Mode;
        # 
        # //define callback signature typedefs
        # typedef void (sig_cb_pxc_point_cloud) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&);
        # typedef void (sig_cb_pxc_point_cloud_rgb) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGB> >&);
        # typedef void (sig_cb_pxc_point_cloud_rgba) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA> >&);
        # typedef void (sig_cb_pxc_point_cloud_i) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&);
        # 
        # public:
        # /** \brief Constructor */
        # PXCGrabber ();
        # 
        # /** \brief virtual Destructor inherited from the Grabber interface. It never throws. */
        # virtual ~PXCGrabber () throw ();
        # 
        # /** \brief Start the data acquisition. */
        # virtual void start ();
        # 
        # /** \brief Stop the data acquisition. */
        # virtual void stop ();
        # 
        # /** \brief Check if the data acquisition is still running. */
        # virtual bool isRunning () const;
        # 
        # /** \brief Returns the name of the grabber. */
        # virtual std::string getName () const;
        # 
        # /** \brief Obtain the number of frames per second (FPS). */
        # virtual float getFramesPerSecond () const;
        # 
        # protected:
        # /** \brief Initializes the PXC grabber and the grabbing pipeline. */
        # bool init ();
        # 
        # /** \brief Closes the grabbing pipeline. */
        # void close ();
        # 
        # /** \brief Continously asks for data from the device and publishes it if available. */
        # void processGrabbing ();
        # 
        # // signals to indicate whether new clouds are available
        # boost::signals2::signal<sig_cb_pxc_point_cloud>* point_cloud_signal_;
        # //boost::signals2::signal<sig_cb_fotonic_point_cloud_i>* point_cloud_i_signal_;
        # boost::signals2::signal<sig_cb_pxc_point_cloud_rgb>* point_cloud_rgb_signal_;
        # boost::signals2::signal<sig_cb_pxc_point_cloud_rgba>* point_cloud_rgba_signal_;
        # 
        # protected:
        # // utiliy object for accessing PXC camera
        # UtilPipeline pp_;
        # // indicates whether grabbing is running
        # bool running_;
        # 
        # // FPS computation
        # mutable float fps_;
        # mutable boost::mutex fps_mutex_;
        # 
        # // thread where the grabbing takes place
        # boost::thread grabber_thread_;
        # 
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW


###


# robot_eye_grabber.h
# namespace pcl
# /** \brief Grabber for the Ocular Robotics RobotEye sensor.
#  * \ingroup io
#  */
# class PCL_EXPORTS RobotEyeGrabber : public Grabber
# cdef extern from "pcl/io/robot_eye_grabber.h" namespace "pcl":
#     cdef cppclass RobotEyeGrabber(Grabber):
#         RobotEyeGrabber()
        # public:
        # 
        # /** \brief Signal used for the point cloud callback.
        #  * This signal is sent when the accumulated number of points reaches
        #  * the limit specified by setSignalPointCloudSize().
        #  */
        # typedef void (sig_cb_robot_eye_point_cloud_xyzi) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&);
        # 
        # /** \brief RobotEyeGrabber default constructor. */
        # RobotEyeGrabber ();
        # 
        # /** \brief RobotEyeGrabber constructor taking a specified IP address and data port. */
        # RobotEyeGrabber (const boost::asio::ip::address& ipAddress, unsigned short port=443);
        # 
        # /** \brief virtual Destructor inherited from the Grabber interface. It never throws. */
        # virtual ~RobotEyeGrabber () throw ();
        # 
        # /** \brief Starts the RobotEye grabber.
        #  * The grabber runs on a separate thread, this call will return without blocking. */
        # virtual void start ();
        # 
        # /** \brief Stops the RobotEye grabber. */
        # virtual void stop ();
        # 
        # /** \brief Obtains the name of this I/O Grabber
        #  *  \return The name of the grabber
        #  */
        # virtual std::string getName () const;
        # 
        # /** \brief Check if the grabber is still running.
        #  *  \return TRUE if the grabber is running, FALSE otherwise
        #  */
        # virtual bool isRunning () const;
        # 
        # /** \brief Returns the number of frames per second.
        #  */
        # virtual float getFramesPerSecond () const;
        # 
        # /** \brief Set/get ip address of the sensor that sends the data.
        #  * The default is address_v4::any ().
        #  */
        # void setSensorAddress (const boost::asio::ip::address& ipAddress);
        # const boost::asio::ip::address& getSensorAddress () const;
        # 
        # /** \brief Set/get the port number which receives data from the sensor.
        #  * The default is 443.
        #  */
        # void setDataPort (unsigned short port);
        # unsigned short getDataPort () const;
        # 
        # /** \brief Set/get the number of points to accumulate before the grabber
        #  * callback is signaled.  The default is 1000.
        #  */
        # void setSignalPointCloudSize (std::size_t numerOfPoints);
        # std::size_t getSignalPointCloudSize () const;
        # 
        # /** \brief Returns the point cloud with point accumulated by the grabber.
        #  * It is not safe to access this point cloud except if the grabber is
        #  * stopped or during the grabber callback.
        #  */
        # boost::shared_ptr<pcl::PointCloud<pcl::PointXYZI> > getPointCloud() const;


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
# cdef extern from "pcl/io/openni_grabber.h" namespace "pcl":
#     ctypedef enum Mode2 "pcl::OpenNIGrabber":
#         Grabber_OpenNI_Default_Mode "pcl::OpenNIGrabber::OpenNI_Default_Mode"   # = 0, // This can depend on the device. For now all devices (PSDK, Xtion, Kinect) its VGA@30Hz
#         Grabber_OpenNI_SXGA_15Hz "pcl::OpenNIGrabber::OpenNI_SXGA_15Hz"         # = 1, // Only supported by the Kinect
#         Grabber_OpenNI_VGA_30Hz "pcl::OpenNIGrabber::OpenNI_VGA_30Hz"           # = 2, // Supported by PSDK, Xtion and Kinect
#         Grabber_OpenNI_VGA_25Hz "pcl::OpenNIGrabber::OpenNI_VGA_25Hz"           # = 3, // Supportged by PSDK and Xtion
#         Grabber_OpenNI_QVGA_25Hz "pcl::OpenNIGrabber::OpenNI_QVGA_25Hz"         # = 4, // Supported by PSDK and Xtion
#         Grabber_OpenNI_QVGA_30Hz "pcl::OpenNIGrabber::OpenNI_QVGA_30Hz"         # = 5, // Supported by PSDK, Xtion and Kinect
#         Grabber_OpenNI_QVGA_60Hz "pcl::OpenNIGrabber::OpenNI_QVGA_60Hz"         # = 6, // Supported by PSDK and Xtion
#         Grabber_OpenNI_QQVGA_25Hz "pcl::OpenNIGrabber::OpenNI_QQVGA_25Hz"       # = 7, // Not supported -> using software downsampling (only for integer scale factor and only NN)
#         Grabber_OpenNI_QQVGA_30Hz "pcl::OpenNIGrabber::OpenNI_QQVGA_30Hz"       # = 8, // Not supported -> using software downsampling (only for integer scale factor and only NN)
#         Grabber_OpenNI_QQVGA_60Hz "pcl::OpenNIGrabber::OpenNI_QQVGA_60Hz"       # = 9  // Not supported -> using software downsampling (only for integer scale factor and only NN)

