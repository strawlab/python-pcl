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

# pcl 1.7
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

# pcl 1.7
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
# cdef extern from "pcl/io/openni_grabber.h" namespace "pcl":
#     cdef cppclass OpenNIGrabber(Grabber):
#         # OpenNIGrabber ()
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
# 
# cdef extern from "pcl/io/openni_grabber.h" namespace "pcl":
#   cdef boost::shared_ptr<openni_wrapper::OpenNIDevice>
# cdef extern from "pcl/io/openni_grabber.h" namespace "pcl":
#   cdef OpenNIGrabber::getDevice () const
###

# pcl.1.8.0
# dinast_grabber.h
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/grabber.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <libusb-1.0/libusb.h>
#include <boost/circular_buffer.hpp>
#
# namespace pcl
# /** \brief Grabber for DINAST devices (i.e., IPA-1002, IPA-1110, IPA-2001)
#   * \author Marco A. Gutierrez <marcog@unex.es>
#   * \ingroup io
#   */
# class PCL_EXPORTS DinastGrabber: public Grabber
        # // Define callback signature typedefs
        # typedef void (sig_cb_dinast_point_cloud) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&);
        # 
        # public:
        # /** \brief Constructor that sets up the grabber constants.
        #   * \param[in] device_position Number corresponding the device to grab
        #   */
        # DinastGrabber (const int device_position=1);
        # 
        # /** \brief Destructor. It never throws. */
        # virtual ~DinastGrabber () throw ();
        # 
        # /** \brief Check if the grabber is running
        #  * \return true if grabber is running / streaming. False otherwise.
        #  */
        # virtual bool isRunning () const;
        # 
        # /** \brief Returns the name of the concrete subclass, DinastGrabber.
        #   * \return DinastGrabber.
        #   */
        # virtual std::string getName () const
        # 
        # /** \brief Start the data acquisition process.
        #  */
        # virtual void start ();
        # 
        # /** \brief Stop the data acquisition process.
        #  */
        # virtual void stop ();
        # 
        # /** \brief Obtain the number of frames per second (FPS). */
        # virtual float getFramesPerSecond () const;
        # 
        # /** \brief Get the version number of the currently opened device
        #  */
        # std::string getDeviceVersion ();
        # 
        # protected:  
        # /** \brief On initialization processing. */
        # void onInit (const int device_id);
        # 
        # /** \brief Setup a Dinast 3D camera device
        #   * \param[in] device_position Number corresponding the device to grab
        #   * \param[in] id_vendor The ID of the camera vendor (should be 0x18d1)
        #   * \param[in] id_product The ID of the product (should be 0x1402)
        #   */
        # void
        # setupDevice (int device_position,
        #            const int id_vendor = 0x18d1, 
        #            const int id_product = 0x1402);
        # 
        # /** \brief Send a RX data packet request
        #   * \param[in] req_code the request to send (the request field for the setup packet)
        #   * \param buffer
        #   * \param[in] length the length field for the setup packet. The data buffer should be at least this #size.
        #   */
        # bool
        # USBRxControlData (const unsigned char req_code,
        #                   unsigned char *buffer,
        #                   int length);
        # 
        # /** \brief Send a TX data packet request
        #   * \param[in] req_code the request to send (the request field for the setup packet)
        #   * \param buffer
        #   * \param[in] length the length field for the setup packet. The data buffer should be at least this size.
        #   */
        # bool
        # USBTxControlData (const unsigned char req_code,
        #                   unsigned char *buffer,
        #                   int length);
        # 
        # /** \brief Check if we have a header in the global buffer, and return the position of the next valid image.
        #   * \note If the image in the buffer is partial, return -1, as we have to wait until we add more data to it.
        #   * \return the position of the next valid image (i.e., right after a valid header) or -1 in case the buffer 
        #   * either doesn't have an image or has a partial image
        #   */
        # int checkHeader ();
        # 
        # /** \brief Read image data and leaves it on image_
        #   */
        # void readImage ();
        # 
        # /** \brief Obtains XYZI Point Cloud from the image of the camera
        #   * \return the point cloud from the image data
        #   */
        # pcl::PointCloud<pcl::PointXYZI>::Ptr getXYZIPointCloud ();
        # 
        # /** \brief The function in charge of getting the data from the camera
        #   */
        # void captureThreadFunction ();
        # 
        # /** \brief Width of image */
        # int image_width_;
        # 
        # /** \brief Height of image */
        # int image_height_;
        # 
        # /** \brief Total size of image */
        # int image_size_;
        # 
        # /** \brief Length of a sync packet */
        # int sync_packet_size_;
        # 
        # double dist_max_2d_;
        # 
        # /** \brief diagonal Field of View*/
        # double fov_;
        # 
        # /** \brief Size of pixel */
        # enum pixel_size { RAW8=1, RGB16=2, RGB24=3, RGB32=4 };
        # 
        # /** \brief The libusb context*/
        # libusb_context *context_;
        # 
        # /** \brief the actual device_handle for the camera */
        # struct libusb_device_handle *device_handle_;
        # 
        # /** \brief Temporary USB read buffer, since we read two RGB16 images at a time size is the double of two images
        #   * plus a sync packet.
        #   */
        # unsigned char *raw_buffer_ ;
        # 
        # /** \brief Global circular buffer */
        # boost::circular_buffer<unsigned char> g_buffer_;
        # 
        # /** \brief Bulk endpoint address value */
        # unsigned char bulk_ep_;
        # 
        # /** \brief Device command values */
        # enum { CMD_READ_START=0xC7, CMD_READ_STOP=0xC8, CMD_GET_VERSION=0xDC, CMD_SEND_DATA=0xDE };
        # 
        # unsigned char *image_;
        # 
        # /** \brief Since there is no header after the first image, we need to save the state */
        # bool second_image_;
        # bool running_;
        # boost::thread capture_thread_;
        # 
        # mutable boost::mutex capture_mutex_;
        # boost::signals2::signal<sig_cb_dinast_point_cloud>* point_cloud_signal_;


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

# pcl 1.8.0
# file_grabber.h
##pragma once
##ifndef PCL_IO_FILE_GRABBER_H_
##define PCL_IO_FILE_GRABBER_H_
#
##include <pcl/point_cloud.h>
#
# namespace pcl
# /** \brief FileGrabber provides a container-style interface for grabbers which operate on fixed-size input
#   * \author Stephen Miller
#   * \ingroup io
#   */
# template <typename PointT>
# class PCL_EXPORTS FileGrabber
        # public:
        # 
        # /** \brief Empty destructor */
        # virtual ~FileGrabber () {}
        # 
        # /** \brief operator[] Returns the idx-th cloud in the dataset, without bounds checking.
        #   *  Note that in the future, this could easily be modified to do caching
        #   *  \param[in] idx The frame to load
        #   */
        # virtual const boost::shared_ptr< const pcl::PointCloud<PointT> > operator[] (size_t idx) const = 0;
        # 
        # /** \brief size Returns the number of clouds currently loaded by the grabber */
        # virtual size_t size () const = 0;
        # 
        # /** \brief at Returns the idx-th cloud in the dataset, with bounds checking
        #  *  \param[in] idx The frame to load
        #  */
        # virtual const boost::shared_ptr< const pcl::PointCloud<PointT> > at (size_t idx) const


###

# pcl 1.8.0
# hdl_grabber.h
##include "pcl/pcl_config.h"
#
##ifndef PCL_IO_HDL_GRABBER_H_
##define PCL_IO_HDL_GRABBER_H_
#
##include <pcl/io/grabber.h>
##include <pcl/io/impl/synchronized_queue.hpp>
##include <pcl/point_types.h>
##include <pcl/point_cloud.h>
##include <boost/asio.hpp>
##include <string>
#
##define HDL_Grabber_toRadians(x) ((x) * M_PI / 180.0)
#
# namespace pcl
# /** \brief Grabber for the Velodyne High-Definition-Laser (HDL)
#   * \author Keven Ring <keven@mitre.org>
#   * \ingroup io
#   */
# class PCL_EXPORTS HDLGrabber : public Grabber
        # public:
        # /** \brief Signal used for a single sector
        #   *         Represents 1 corrected packet from the HDL Velodyne
        #   */
        # typedef void
        # (sig_cb_velodyne_hdl_scan_point_cloud_xyz) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&,
        #                                             float,
        #                                             float);
        # /** \brief Signal used for a single sector
        #   *         Represents 1 corrected packet from the HDL Velodyne.  Each laser has a different RGB
        #   */
        # typedef void
        # (sig_cb_velodyne_hdl_scan_point_cloud_xyzrgb) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA> >&,
        #                                                float,
        #                                                float);
        # /** \brief Signal used for a single sector
        #   *         Represents 1 corrected packet from the HDL Velodyne with the returned intensity.
        #   */
        # typedef void
        # (sig_cb_velodyne_hdl_scan_point_cloud_xyzi) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&,
        #                                              float startAngle,
        #                                              float);
        # /** \brief Signal used for a 360 degree sweep
        #   *         Represents multiple corrected packets from the HDL Velodyne
        #   *         This signal is sent when the Velodyne passes angle "0"
        #   */
        # typedef void
        # (sig_cb_velodyne_hdl_sweep_point_cloud_xyz) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&);
        # /** \brief Signal used for a 360 degree sweep
        #   *         Represents multiple corrected packets from the HDL Velodyne with the returned intensity
        #   *         This signal is sent when the Velodyne passes angle "0"
        #   */
        # typedef void
        # (sig_cb_velodyne_hdl_sweep_point_cloud_xyzi) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&);
        # /** \brief Signal used for a 360 degree sweep
        #   *         Represents multiple corrected packets from the HDL Velodyne
        #   *         This signal is sent when the Velodyne passes angle "0".  Each laser has a different RGB
        #   */
        # typedef void
        # (sig_cb_velodyne_hdl_sweep_point_cloud_xyzrgb) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA> >&);
        # 
        # /** \brief Constructor taking an optional path to an HDL corrections file.  The Grabber will listen on the default IP/port for data packets [192.168.3.255/2368]
        #   * \param[in] correctionsFile Path to a file which contains the correction parameters for the HDL.  This parameter is mandatory for the HDL-64, optional for the HDL-32
        #   * \param[in] pcapFile Path to a file which contains previously captured data packets.  This parameter is optional
        #   */
        # HDLGrabber (const std::string& correctionsFile = "", const std::string& pcapFile = "");
        # 
        # /** \brief Constructor taking a pecified IP/port and an optional path to an HDL corrections file.
        #   * \param[in] ipAddress IP Address that should be used to listen for HDL packets
        #   * \param[in] port UDP Port that should be used to listen for HDL packets
        #   * \param[in] correctionsFile Path to a file which contains the correction parameters for the HDL.  This field is mandatory for the HDL-64, optional for the HDL-32
        #   */
        # HDLGrabber (const boost::asio::ip::address& ipAddress, const unsigned short port, const std::string& correctionsFile = "?");
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
        # */
        # virtual float getFramesPerSecond () const;
        # 
        # /** \brief Allows one to filter packets based on the SOURCE IP address and PORT
        # *         This can be used, for instance, if multiple HDL LIDARs are on the same network
        # */
        # void
        # filterPackets (const boost::asio::ip::address& ipAddress, const unsigned short port = 443);
        # 
        # /** \brief Allows one to customize the colors used for each of the lasers.
        # */
        # void
        # setLaserColorRGB (const pcl::RGB& color, unsigned int laserNumber);
        # 
        # /** \brief Any returns from the HDL with a distance less than this are discarded.
        #   *         This value is in meters
        #   *         Default: 0.0
        #   */
        # void setMinimumDistanceThreshold (float & minThreshold);
        # 
        # /** \brief Any returns from the HDL with a distance greater than this are discarded.
        #   *         This value is in meters
        #   *         Default: 10000.0
        #   */
        # void setMaximumDistanceThreshold (float & maxThreshold);
        # 
        # /** \brief Returns the current minimum distance threshold, in meters
        #   */
        # float getMinimumDistanceThreshold ();
        # 
        # /** \brief Returns the current maximum distance threshold, in meters
        #   */
        # float getMaximumDistanceThreshold ();
        # 
        # protected:
        # static const int HDL_DATA_PORT = 2368;
        # static const int HDL_NUM_ROT_ANGLES = 36001;
        # static const int HDL_LASER_PER_FIRING = 32;
        # static const int HDL_MAX_NUM_LASERS = 64;
        # static const int HDL_FIRING_PER_PKT = 12;
        # 
        # enum HDLBlock
        # {
        #   BLOCK_0_TO_31 = 0xeeff, BLOCK_32_TO_63 = 0xddff
        # };
        # 
        # #pragma pack(push, 1)
        # typedef struct HDLLaserReturn
        # {
        #    unsigned short distance;
        #    unsigned char intensity;
        # } HDLLaserReturn;
        # #pragma pack(pop)
        # 
        # struct HDLFiringData
        # {
        #   unsigned short blockIdentifier;
        #   unsigned short rotationalPosition;
        #   HDLLaserReturn laserReturns[HDL_LASER_PER_FIRING];
        # };
        # 
        # struct HDLDataPacket
        # {
        #   HDLFiringData firingData[HDL_FIRING_PER_PKT];
        #   unsigned int gpsTimestamp;
        #   unsigned char mode;
        #   unsigned char sensorType;
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
        # HDLLaserCorrection laser_corrections_[HDL_MAX_NUM_LASERS];
        # unsigned int last_azimuth_;
        # boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > current_scan_xyz_, current_sweep_xyz_;
        # boost::shared_ptr<pcl::PointCloud<pcl::PointXYZI> > current_scan_xyzi_, current_sweep_xyzi_;
        # boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA> > current_scan_xyzrgb_, current_sweep_xyzrgb_;
        # boost::signals2::signal<sig_cb_velodyne_hdl_sweep_point_cloud_xyz>* sweep_xyz_signal_;
        # boost::signals2::signal<sig_cb_velodyne_hdl_sweep_point_cloud_xyzrgb>* sweep_xyzrgb_signal_;
        # boost::signals2::signal<sig_cb_velodyne_hdl_sweep_point_cloud_xyzi>* sweep_xyzi_signal_;
        # boost::signals2::signal<sig_cb_velodyne_hdl_scan_point_cloud_xyz>* scan_xyz_signal_;
        # boost::signals2::signal<sig_cb_velodyne_hdl_scan_point_cloud_xyzrgb>* scan_xyzrgb_signal_;
        # boost::signals2::signal<sig_cb_velodyne_hdl_scan_point_cloud_xyzi>* scan_xyzi_signal_;
        # 
        # void fireCurrentSweep ();
        # 
        # void fireCurrentScan (const unsigned short startAngle, const unsigned short endAngle);
        # void computeXYZI (pcl::PointXYZI& pointXYZI,
        #                   int azimuth,
        #                   HDLLaserReturn laserReturn,
        #                   HDLLaserCorrection correction);


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
###

