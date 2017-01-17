from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp

# boost
from boost_shared_ptr cimport shared_ptr


###############################################################################
# Types
###############################################################################

### base class ###

# quantizable_modality.h
# namespace pcl
# {
# /** \brief Interface for a quantizable modality. 
#  * \author Stefan Holzer
#  */
# class PCL_EXPORTS QuantizableModality
cdef extern from "pcl/Recognition/quantizable_modality.h" namespace "pcl":
    cdef cppclass Feature[In, Out](cpp.PCLBase[In]):
        QuantizableModality ()
        # /** \brief Destructor. */
        # virtual ~QuantizableModality ();
        
        # /** \brief Returns a reference to the internally computed quantized map. */
        # virtual QuantizedMap & getQuantizedMap () = 0;
        
        # /** \brief Returns a reference to the internally computed spreaded quantized map. */
        # virtual QuantizedMap & getSpreadedQuantizedMap () = 0;
        
        # /** \brief Extracts features from this modality within the specified mask.
        #   * \param[in] mask defines the areas where features are searched in. 
        #   * \param[in] nr_features defines the number of features to be extracted 
        #   *            (might be less if not sufficient information is present in the modality).
        #   * \param[in] modality_index the index which is stored in the extracted features.
        #   * \param[out] features the destination for the extracted features.
        # */
        # virtual void extractFeatures (
        #       const MaskMap & mask, size_t nr_features, size_t modality_index, 
        #         std::vector<QuantizedMultiModFeature> & features) const = 0;
        # 
        # /** \brief Extracts all possible features from the modality within the specified mask.
        #   * \param[in] mask defines the areas where features are searched in. 
        #   * \param[in] nr_features IGNORED (TODO: remove this parameter).
        #   * \param[in] modality_index the index which is stored in the extracted features.
        #   * \param[out] features the destination for the extracted features.
        # */
        # virtual void  extractAllFeatures (const MaskMap & mask, size_t nr_features, size_t modality_index, 
        #                                       std::vector<QuantizedMultiModFeature> & features) const = 0;

###

### Inheritance class ###

# auxiliary.h
# #include <pcl/recognition/ransac_based/auxiliary.h>
###

# boost.h
# #include <boost/unordered_map.hpp>
# #include <boost/graph/graph_traits.hpp>
###

# bvh.h
# #include <pcl/recognition/ransac_based/bvh.h>
###

# color_gradient_dot_modality.h
# namespace pcl
# {
# 
#   /** \brief A point structure for representing RGB color
#     * \ingroup common
#     */
#   struct EIGEN_ALIGN16 PointRGB
#   {
#     union
#     {
#       union
#       {
#         struct
#         {
#           uint8_t b;
#           uint8_t g;
#           uint8_t r;
#           uint8_t _unused;
#         };
#         float rgb;
#       };
#       uint32_t rgba;
#     };
# 
#     inline PointRGB ()
#     {}
# 
#     inline PointRGB (const uint8_t b, const uint8_t g, const uint8_t r)
#       : b (b), g (g), r (r), _unused (0)
#     {}
# 
#     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
# 
# 
#   /** \brief A point structure representing Euclidean xyz coordinates, and the intensity value.
#     * \ingroup common
#     */
#   struct EIGEN_ALIGN16 GradientXY
#   {
#     union
#     {
#       struct
#       {
#         float x;
#         float y;
#         float angle;
#         float magnitude;
#       };
#       float data[4];
#     };
#     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
# 
#     inline bool operator< (const GradientXY & rhs)
#     {
#       return (magnitude > rhs.magnitude);
#     }
#   };
#   inline std::ostream & operator << (std::ostream & os, const GradientXY & p)
#   {
#     os << "(" << p.x << "," << p.y << " - " << p.magnitude << ")";
#    return (os);
#   }
# 
#   // --------------------------------------------------------------------------


#   template <typename PointInT>
#   class ColorGradientDOTModality : public DOTModality, public PCLBase<PointInT>
#   {
#     protected:
#       using PCLBase<PointInT>::input_;
# 
#     struct Candidate
#     {
#       GradientXY gradient;
#     
#       int x;
#       int y;  
#     
#       bool operator< (const Candidate & rhs)
#       {
#         return (gradient.magnitude > rhs.gradient.magnitude);
#       }
#     };
# 
#     public:
#       typedef typename pcl::PointCloud<PointInT> PointCloudIn;
# 
#       ColorGradientDOTModality (size_t bin_size);
#   
#       virtual ~ColorGradientDOTModality ();
#   
#       inline void
#       setGradientMagnitudeThreshold (const float threshold)
#       {
#         gradient_magnitude_threshold_ = threshold;
#       }
#   
#       //inline QuantizedMap &
#       //getDominantQuantizedMap () 
#       //{ 
#       //  return (dominant_quantized_color_gradients_);
#       //}
#   
#       inline QuantizedMap &
#       getDominantQuantizedMap () 
#       { 
#         return (dominant_quantized_color_gradients_);
#       }
# 
#       QuantizedMap
#       computeInvariantQuantizedMap (const MaskMap & mask,
#                                    const RegionXY & region);
#   
#       /** \brief Provide a pointer to the input dataset (overwrites the PCLBase::setInputCloud method)
#         * \param cloud the const boost shared pointer to a PointCloud message
#         */
#       virtual void 
#       setInputCloud (const typename PointCloudIn::ConstPtr & cloud) 
#       { 
#         input_ = cloud;
#         //processInputData ();
#       }
# 
#       virtual void
#       processInputData ();
# 
#     protected:
# 
#       void
#       computeMaxColorGradients ();
#   
#       void
#       computeDominantQuantizedGradients ();
#   
#       //void
#       //computeInvariantQuantizedGradients ();
#   
#     private:
#       size_t bin_size_;
# 
#       float gradient_magnitude_threshold_;
#       pcl::PointCloud<pcl::GradientXY> color_gradients_;
#   
#       pcl::QuantizedMap dominant_quantized_color_gradients_;
#       //pcl::QuantizedMap invariant_quantized_color_gradients_;
#   
#   };
# 
# }

# template <typename PointInT>
# pcl::ColorGradientDOTModality<PointInT>::ColorGradientDOTModality (const size_t bin_size)
#   : bin_size_ (bin_size), gradient_magnitude_threshold_ (80.0f), color_gradients_ (), dominant_quantized_color_gradients_ ()
# {
# }
# 
# //////////////////////////////////////////////////////////////////////////////////////////////
# template <typename PointInT>
# pcl::ColorGradientDOTModality<PointInT>::
# ~ColorGradientDOTModality ()
# {
# }
# 
# //////////////////////////////////////////////////////////////////////////////////////////////
# template <typename PointInT>
# void
# pcl::ColorGradientDOTModality<PointInT>::
# processInputData ()
# {
#   // extract color gradients
#   computeMaxColorGradients ();
# 
#   // compute dominant quantized gradient map
#   computeDominantQuantizedGradients ();
# 
#   // compute invariant quantized gradient map
#   //computeInvariantQuantizedGradients ();
# }
# 
# //////////////////////////////////////////////////////////////////////////////////////////////
# template <typename PointInT>
# void
# pcl::ColorGradientDOTModality<PointInT>::
# computeMaxColorGradients ()
# {
#   const int width = input_->width;
#   const int height = input_->height;
# 
#   color_gradients_.points.resize (width*height);
#   color_gradients_.width = width;
#   color_gradients_.height = height;
# 
#   const float pi = tan(1.0f)*4;
#   for (int row_index = 0; row_index < height-2; ++row_index)
#   {
#     for (int col_index = 0; col_index < width-2; ++col_index)
#     {
#       const int index0 = row_index*width+col_index;
#       const int index_c = row_index*width+col_index+2;
#       const int index_r = (row_index+2)*width+col_index;
# 
#       //const int index_d = (row_index+1)*width+col_index+1;
# 
#       const unsigned char r0 = input_->points[index0].r;
#       const unsigned char g0 = input_->points[index0].g;
#       const unsigned char b0 = input_->points[index0].b;
# 
#       const unsigned char r_c = input_->points[index_c].r;
#       const unsigned char g_c = input_->points[index_c].g;
#       const unsigned char b_c = input_->points[index_c].b;
# 
#       const unsigned char r_r = input_->points[index_r].r;
#       const unsigned char g_r = input_->points[index_r].g;
#       const unsigned char b_r = input_->points[index_r].b;
# 
#       const float r_dx = static_cast<float> (r_c) - static_cast<float> (r0);
#       const float g_dx = static_cast<float> (g_c) - static_cast<float> (g0);
#       const float b_dx = static_cast<float> (b_c) - static_cast<float> (b0);
# 
#       const float r_dy = static_cast<float> (r_r) - static_cast<float> (r0);
#       const float g_dy = static_cast<float> (g_r) - static_cast<float> (g0);
#       const float b_dy = static_cast<float> (b_r) - static_cast<float> (b0);
# 
#       const float sqr_mag_r = r_dx*r_dx + r_dy*r_dy;
#       const float sqr_mag_g = g_dx*g_dx + g_dy*g_dy;
#       const float sqr_mag_b = b_dx*b_dx + b_dy*b_dy;
# 
#       GradientXY gradient;
#       gradient.x = col_index;
#       gradient.y = row_index;
#       if (sqr_mag_r > sqr_mag_g && sqr_mag_r > sqr_mag_b)
#       {
#         gradient.magnitude = sqrt (sqr_mag_r);
#         gradient.angle = atan2 (r_dy, r_dx) * 180.0f / pi;
#       }
#       else if (sqr_mag_g > sqr_mag_b)
#       {
#         //GradientXY gradient;
#         gradient.magnitude = sqrt (sqr_mag_g);
#         gradient.angle = atan2 (g_dy, g_dx) * 180.0f / pi;
#         //gradient.x = col_index;
#         //gradient.y = row_index;
# 
#         //color_gradients_ (col_index+1, row_index+1) = gradient;
#       }
#       else
#       {
#         //GradientXY gradient;
#         gradient.magnitude = sqrt (sqr_mag_b);
#         gradient.angle = atan2 (b_dy, b_dx) * 180.0f / pi;
#         //gradient.x = col_index;
#         //gradient.y = row_index;
# 
#         //color_gradients_ (col_index+1, row_index+1) = gradient;
#       }
# 
#       assert (color_gradients_ (col_index+1, row_index+1).angle >= -180 &&
#               color_gradients_ (col_index+1, row_index+1).angle <=  180);
# 
#       color_gradients_ (col_index+1, row_index+1) = gradient;
#     }
#   }
# 
#   return;
# }


# //////////////////////////////////////////////////////////////////////////////////////////////
# template <typename PointInT>
# void
# pcl::ColorGradientDOTModality<PointInT>::
# computeDominantQuantizedGradients ()
# {
#   const size_t input_width = input_->width;
#   const size_t input_height = input_->height;
# 
#   const size_t output_width = input_width / bin_size_;
#   const size_t output_height = input_height / bin_size_;
# 
#   dominant_quantized_color_gradients_.resize (output_width, output_height);
# 
#   //size_t offset_x = 0;
#   //size_t offset_y = 0;
#   
#   const size_t num_gradient_bins = 7;
#   const size_t max_num_of_gradients = 1;
#   
#   const float divisor = 180.0f / (num_gradient_bins - 1.0f);
#   
#   float global_max_gradient = 0.0f;
#   float local_max_gradient = 0.0f;
#   
#   unsigned char * peak_pointer = dominant_quantized_color_gradients_.getData ();
#   memset (peak_pointer, 0, output_width*output_height);
#   
#   //int tmpCounter = 0;
#   for (size_t row_bin_index = 0; row_bin_index < output_height; ++row_bin_index)
#   {
#     for (size_t col_bin_index = 0; col_bin_index < output_width; ++col_bin_index)
#     {
#       const size_t x_position = col_bin_index * bin_size_;
#       const size_t y_position = row_bin_index * bin_size_;
# 
#       //std::vector<int> x_coordinates;
#       //std::vector<int> y_coordinates;
#       //std::vector<float> values;
#       
#       // iteratively search for the largest gradients, set it to -1, search the next largest ... etc.
#       //while (counter < max_num_of_gradients)
#       {
#         float max_gradient;
#         size_t max_gradient_pos_x;
#         size_t max_gradient_pos_y;
#             
#         // find next location and value of maximum gradient magnitude in current region
#         {
#           max_gradient = 0.0f;
#           for (size_t row_sub_index = 0; row_sub_index < bin_size_; ++row_sub_index)
#           {
#             for (size_t col_sub_index = 0; col_sub_index < bin_size_; ++col_sub_index)
#             {
#               const float magnitude = color_gradients_ (col_sub_index + x_position, row_sub_index + y_position).magnitude;
# 
#               if (magnitude > max_gradient)
#               {
#                 max_gradient = magnitude;
#                 max_gradient_pos_x = col_sub_index;
#                 max_gradient_pos_y = row_sub_index;
#               }
#             }
#           }
#         }
#             
#         if (max_gradient >= gradient_magnitude_threshold_)
#         {
#           const size_t angle = static_cast<size_t> (180 + color_gradients_ (max_gradient_pos_x + x_position, max_gradient_pos_y + y_position).angle + 0.5f);
#           const size_t bin_index = static_cast<size_t> ((angle >= 180 ? angle-180 : angle)/divisor);
#             
#           *peak_pointer |= 1 << bin_index;
#         }
#             
#         //++counter;
#             
#         //x_coordinates.push_back (max_gradient_pos_x + x_position);
#         //y_coordinates.push_back (max_gradient_pos_y + y_position);
#         //values.push_back (max_gradient);
#             
#         //color_gradients_ (max_gradient_pos_x + x_position, max_gradient_pos_y + y_position).magnitude = -1.0f;
#       }
# 
#       //// reset values which have been set to -1
#       //for (size_t value_index = 0; value_index < values.size (); ++value_index)
#       //{
#       //  color_gradients_ (x_coordinates[value_index], y_coordinates[value_index]).magnitude = values[value_index];
#       //}
# 
# 
#       if (*peak_pointer == 0)
#       {
#         *peak_pointer |= 1 << 7;
#       }
# 
#       //if (*peakPointer != 0)
#       //{
#       //  ++tmpCounter;
#       //}
#       
#       //++stringPointer;
#       ++peak_pointer;
#       
#       //offset_x += bin_size;
#     }
#     
#     //offset_y += bin_size;
#     //offset_x = bin_size/2+1;
#   }
# }


# //////////////////////////////////////////////////////////////////////////////////////////////
# template <typename PointInT>
# pcl::QuantizedMap
# pcl::ColorGradientDOTModality<PointInT>::
# computeInvariantQuantizedMap (const MaskMap & mask,
#                               const RegionXY & region)
# {
#   const size_t input_width = input_->width;
#   const size_t input_height = input_->height;
# 
#   const size_t output_width = input_width / bin_size_;
#   const size_t output_height = input_height / bin_size_;
# 
#   const size_t sub_start_x = region.x / bin_size_;
#   const size_t sub_start_y = region.y / bin_size_;
#   const size_t sub_width = region.width / bin_size_;
#   const size_t sub_height = region.height / bin_size_;
# 
#   QuantizedMap map;
#   map.resize (sub_width, sub_height);
# 
#   //size_t offset_x = 0;
#   //size_t offset_y = 0;
#   
#   const size_t num_gradient_bins = 7;
#   const size_t max_num_of_gradients = 7;
#   
#   const float divisor = 180.0f / (num_gradient_bins - 1.0f);
#   
#   float global_max_gradient = 0.0f;
#   float local_max_gradient = 0.0f;
#   
#   unsigned char * peak_pointer = map.getData ();
#   
#   //int tmpCounter = 0;
#   for (size_t row_bin_index = 0; row_bin_index < sub_height; ++row_bin_index)
#   {
#     for (size_t col_bin_index = 0; col_bin_index < sub_width; ++col_bin_index)
#     {
#       std::vector<size_t> x_coordinates;
#       std::vector<size_t> y_coordinates;
#       std::vector<float> values;
#       
#       for (int row_pixel_index = -static_cast<int> (bin_size_)/2; 
#            row_pixel_index <= static_cast<int> (bin_size_)/2; 
#            row_pixel_index += static_cast<int> (bin_size_)/2)
#       {
#         const size_t y_position = /*offset_y +*/ row_pixel_index + (sub_start_y + row_bin_index)*bin_size_;
# 
#         if (y_position < 0 || y_position >= input_height) 
#           continue;
# 
#         for (int col_pixel_index = -static_cast<int> (bin_size_)/2; 
#              col_pixel_index <= static_cast<int> (bin_size_)/2; 
#              col_pixel_index += static_cast<int> (bin_size_)/2)
#         {
#           const size_t x_position = /*offset_x +*/ col_pixel_index + (sub_start_x + col_bin_index)*bin_size_;
#           size_t counter = 0;
#           
#           if (x_position < 0 || x_position >= input_width) 
#             continue;
# 
#           // find maximum gradient magnitude in current bin
#           {
#             local_max_gradient = 0.0f;
#             for (size_t row_sub_index = 0; row_sub_index < bin_size_; ++row_sub_index)
#             {
#               for (size_t col_sub_index = 0; col_sub_index < bin_size_; ++col_sub_index)
#               {
#                 const float magnitude = color_gradients_ (col_sub_index + x_position, row_sub_index + y_position).magnitude;
# 
#                 if (magnitude > local_max_gradient)
#                   local_max_gradient = magnitude;
#               }
#             }
#           }
#           
#           //*stringPointer += localMaxGradient;
#           
#           if (local_max_gradient > global_max_gradient)
#           {
#             global_max_gradient = local_max_gradient;
#           }
#           
#           // iteratively search for the largest gradients, set it to -1, search the next largest ... etc.
#           while (true)
#           {
#             float max_gradient;
#             size_t max_gradient_pos_x;
#             size_t max_gradient_pos_y;
#             
#             // find next location and value of maximum gradient magnitude in current region
#             {
#               max_gradient = 0.0f;
#               for (size_t row_sub_index = 0; row_sub_index < bin_size_; ++row_sub_index)
#               {
#                 for (size_t col_sub_index = 0; col_sub_index < bin_size_; ++col_sub_index)
#                 {
#                   const float magnitude = color_gradients_ (col_sub_index + x_position, row_sub_index + y_position).magnitude;
# 
#                   if (magnitude > max_gradient)
#                   {
#                     max_gradient = magnitude;
#                     max_gradient_pos_x = col_sub_index;
#                     max_gradient_pos_y = row_sub_index;
#                   }
#                 }
#               }
#             }
#             
#             // TODO: really localMaxGradient and not maxGradient???
#             if (local_max_gradient < gradient_magnitude_threshold_)
#             {
#               //*peakPointer |= 1 << (numOfGradientBins-1);
#               break;
#             }
#             
#             // TODO: replace gradient_magnitude_threshold_ here by a fixed ratio?
#             if (/*max_gradient < (local_max_gradient * gradient_magnitude_threshold_) ||*/
#                 counter >= max_num_of_gradients)
#             {
#               break;
#             }
#             
#             ++counter;
#             
#             const size_t angle = static_cast<size_t> (180 + color_gradients_ (max_gradient_pos_x + x_position, max_gradient_pos_y + y_position).angle + 0.5f);
#             const size_t bin_index = static_cast<size_t> ((angle >= 180 ? angle-180 : angle)/divisor);
#             
#             *peak_pointer |= 1 << bin_index;
#             
#             x_coordinates.push_back (max_gradient_pos_x + x_position);
#             y_coordinates.push_back (max_gradient_pos_y + y_position);
#             values.push_back (max_gradient);
#             
#             color_gradients_ (max_gradient_pos_x + x_position, max_gradient_pos_y + y_position).magnitude = -1.0f;
#           }
#           
#           // reset values which have been set to -1
#           for (size_t value_index = 0; value_index < values.size (); ++value_index)
#           {
#             color_gradients_ (x_coordinates[value_index], y_coordinates[value_index]).magnitude = values[value_index];
#           }
#           
#           x_coordinates.clear ();
#           y_coordinates.clear ();
#           values.clear ();
#         }
#       }
# 
#       if (*peak_pointer == 0)
#       {
#         *peak_pointer |= 1 << 7;
#       }
# 
#       //if (*peakPointer != 0)
#       //{
#       //  ++tmpCounter;
#       //}
#       
#       //++stringPointer;
#       ++peak_pointer;
#       
#       //offset_x += bin_size;
#     }
#     
#     //offset_y += bin_size;
#     //offset_x = bin_size/2+1;
#   }
# 
#   return map;
# }
# 
# #endif

# color_gradient_modality.h
# namespace pcl
# 
# /** \brief Modality based on max-RGB gradients.
#   * \author Stefan Holzer
#   */
# template <typename PointInT>
# class ColorGradientModality : public QuantizableModality, public PCLBase<PointInT>
  {
    protected:
      using PCLBase<PointInT>::input_;

      /** \brief Candidate for a feature (used in feature extraction methods). */
      struct Candidate
      {
        /** \brief The gradient. */
        GradientXY gradient;
    
        /** \brief The x-position. */
        int x;
        /** \brief The y-position. */
        int y;  
    
        /** \brief Operator for comparing to candidates (by magnitude of the gradient).
          * \param[in] rhs the candidate to compare with.
          */
        bool operator< (const Candidate & rhs) const
        {
          return (gradient.magnitude > rhs.gradient.magnitude);
        }
      };

    public:
      typedef typename pcl::PointCloud<PointInT> PointCloudIn;

      /** \brief Different methods for feature selection/extraction. */
      enum FeatureSelectionMethod
      {
        MASK_BORDER_HIGH_GRADIENTS,
        MASK_BORDER_EQUALLY, // this gives templates most equally to the OpenCV implementation
        DISTANCE_MAGNITUDE_SCORE
      };

      /** \brief Constructor. */
      ColorGradientModality ();
      /** \brief Destructor. */
      virtual ~ColorGradientModality ();
  
      /** \brief Sets the threshold for the gradient magnitude which is used when quantizing the data.
        *        Gradients with a smaller magnitude are ignored. 
        * \param[in] threshold the new gradient magnitude threshold.
        */
      inline void
      setGradientMagnitudeThreshold (const float threshold)
      {
        gradient_magnitude_threshold_ = threshold;
      }

      /** \brief Sets the threshold for the gradient magnitude which is used for feature extraction.
        *        Gradients with a smaller magnitude are ignored. 
        * \param[in] threshold the new gradient magnitude threshold.
        */
      inline void
      setGradientMagnitudeThresholdForFeatureExtraction (const float threshold)
      {
        gradient_magnitude_threshold_feature_extraction_ = threshold;
      }

      /** \brief Sets the feature selection method.
        * \param[in] method the feature selection method.
        */
      inline void
      setFeatureSelectionMethod (const FeatureSelectionMethod method)
      {
        feature_selection_method_ = method;
      }
  
      /** \brief Sets the spreading size for spreading the quantized data. */
      inline void
      setSpreadingSize (const size_t spreading_size)
      {
        spreading_size_ = spreading_size;
      }

      /** \brief Sets whether variable feature numbers for feature extraction is enabled.
        * \param[in] enabled enables/disables variable feature numbers for feature extraction.
        */
      inline void
      setVariableFeatureNr (const bool enabled)
      {
        variable_feature_nr_ = enabled;
      }

      /** \brief Returns a reference to the internally computed quantized map. */
      inline QuantizedMap &
      getQuantizedMap () 
      { 
        return (filtered_quantized_color_gradients_);
      }
  
      /** \brief Returns a reference to the internally computed spreaded quantized map. */
      inline QuantizedMap &
      getSpreadedQuantizedMap () 
      { 
        return (spreaded_filtered_quantized_color_gradients_);
      }

      /** \brief Returns a point cloud containing the max-RGB gradients. */
      inline pcl::PointCloud<pcl::GradientXY> &
      getMaxColorGradients ()
      {
        return (color_gradients_);
      }
  
      /** \brief Extracts features from this modality within the specified mask.
        * \param[in] mask defines the areas where features are searched in. 
        * \param[in] nr_features defines the number of features to be extracted 
        *            (might be less if not sufficient information is present in the modality).
        * \param[in] modalityIndex the index which is stored in the extracted features.
        * \param[out] features the destination for the extracted features.
        */
      void
      extractFeatures (const MaskMap & mask, size_t nr_features, size_t modalityIndex,
                       std::vector<QuantizedMultiModFeature> & features) const;
  
      /** \brief Extracts all possible features from the modality within the specified mask.
        * \param[in] mask defines the areas where features are searched in. 
        * \param[in] nr_features IGNORED (TODO: remove this parameter).
        * \param[in] modalityIndex the index which is stored in the extracted features.
        * \param[out] features the destination for the extracted features.
        */
      void
      extractAllFeatures (const MaskMap & mask, size_t nr_features, size_t modalityIndex,
                          std::vector<QuantizedMultiModFeature> & features) const;
  
      /** \brief Provide a pointer to the input dataset (overwrites the PCLBase::setInputCloud method)
        * \param cloud the const boost shared pointer to a PointCloud message
        */
      virtual void 
      setInputCloud (const typename PointCloudIn::ConstPtr & cloud) 
      { 
        input_ = cloud;
      }

      /** \brief Processes the input data (smoothing, computing gradients, quantizing, filtering, spreading). */
      virtual void
      processInputData ();

      /** \brief Processes the input data assuming that everything up to filtering is already done/available 
        *        (so only spreading is performed). */
      virtual void
      processInputDataFromFiltered ();

    protected:

      /** \brief Computes the Gaussian kernel used for smoothing. 
        * \param[in] kernel_size the size of the Gaussian kernel. 
        * \param[in] sigma the sigma.
        * \param[out] kernel_values the destination for the values of the kernel. */
      void
      computeGaussianKernel (const size_t kernel_size, const float sigma, std::vector <float> & kernel_values);

      /** \brief Computes the max-RGB gradients for the specified cloud.
        * \param[in] cloud the cloud for which the gradients are computed.
        */
      void
      computeMaxColorGradients (const typename pcl::PointCloud<pcl::RGB>::ConstPtr & cloud);

      /** \brief Computes the max-RGB gradients for the specified cloud using sobel.
        * \param[in] cloud the cloud for which the gradients are computed.
        */
      void
      computeMaxColorGradientsSobel (const typename pcl::PointCloud<pcl::RGB>::ConstPtr & cloud);
  
      /** \brief Quantizes the color gradients. */
      void
      quantizeColorGradients ();
  
      /** \brief Filters the quantized gradients. */
      void
      filterQuantizedColorGradients ();

      /** \brief Erodes a mask.
        * \param[in] mask_in the mask which will be eroded.
        * \param[out] mask_out the destination for the eroded mask.
        */
      static void
      erode (const pcl::MaskMap & mask_in, pcl::MaskMap & mask_out);
  
    private:

      /** \brief Determines whether variable numbers of features are extracted or not. */
      bool variable_feature_nr_;

      /** \brief Stores a smoothed verion of the input cloud. */
        pcl::PointCloud<pcl::RGB>::Ptr smoothed_input_;

      /** \brief Defines which feature selection method is used. */
      FeatureSelectionMethod feature_selection_method_;

      /** \brief The threshold applied on the gradient magnitudes (for quantization). */
      float gradient_magnitude_threshold_;
      /** \brief The threshold applied on the gradient magnitudes for feature extraction. */
      float gradient_magnitude_threshold_feature_extraction_;

      /** \brief The point cloud which holds the max-RGB gradients. */
      pcl::PointCloud<pcl::GradientXY> color_gradients_;

      /** \brief The spreading size. */
      size_t spreading_size_;
  
      /** \brief The map which holds the quantized max-RGB gradients. */
      pcl::QuantizedMap quantized_color_gradients_;
      /** \brief The map which holds the filtered quantized data. */
      pcl::QuantizedMap filtered_quantized_color_gradients_;
      /** \brief The map which holds the spreaded quantized data. */
      pcl::QuantizedMap spreaded_filtered_quantized_color_gradients_;
  
  };

}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
pcl::ColorGradientModality<PointInT>::
ColorGradientModality ()
  : variable_feature_nr_ (false)
  , smoothed_input_ (new pcl::PointCloud<pcl::RGB> ())
  , feature_selection_method_ (DISTANCE_MAGNITUDE_SCORE)
  , gradient_magnitude_threshold_ (10.0f)
  , gradient_magnitude_threshold_feature_extraction_ (55.0f)
  , color_gradients_ ()
  , spreading_size_ (8)
  , quantized_color_gradients_ ()
  , filtered_quantized_color_gradients_ ()
  , spreaded_filtered_quantized_color_gradients_ ()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
pcl::ColorGradientModality<PointInT>::
~ColorGradientModality ()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT> void
pcl::ColorGradientModality<PointInT>::
computeGaussianKernel (const size_t kernel_size, const float sigma, std::vector <float> & kernel_values)
{
  // code taken from OpenCV
  const int n = int (kernel_size);
  const int SMALL_GAUSSIAN_SIZE = 7;
  static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
  {
      {1.f},
      {0.25f, 0.5f, 0.25f},
      {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
      {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
  };

  const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ?
      small_gaussian_tab[n>>1] : 0;

  //CV_Assert( ktype == CV_32F || ktype == CV_64F );
  /*Mat kernel(n, 1, ktype);*/
  kernel_values.resize (n);
  float* cf = &(kernel_values[0]);
  //double* cd = (double*)kernel.data;

  double sigmaX = sigma > 0 ? sigma : ((n-1)*0.5 - 1)*0.3 + 0.8;
  double scale2X = -0.5/(sigmaX*sigmaX);
  double sum = 0;

  int i;
  for( i = 0; i < n; i++ )
  {
    double x = i - (n-1)*0.5;
    double t = fixed_kernel ? double (fixed_kernel[i]) : std::exp (scale2X*x*x);

    cf[i] = float (t);
    sum += cf[i];
  }

  sum = 1./sum;
  for (i = 0; i < n; i++ )
  {
    cf[i] = float (cf[i]*sum);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void
pcl::ColorGradientModality<PointInT>::
processInputData ()
{
  // compute gaussian kernel values
  const size_t kernel_size = 7;
  std::vector<float> kernel_values;
  computeGaussianKernel (kernel_size, 0.0f, kernel_values);

  // smooth input
    pcl::filters::Convolution<pcl::RGB, pcl::RGB> convolution;
    Eigen::ArrayXf gaussian_kernel(kernel_size);
    //gaussian_kernel << 1.f/16, 1.f/8, 3.f/16, 2.f/8, 3.f/16, 1.f/8, 1.f/16;
    //gaussian_kernel << 16.f/1600.f,  32.f/1600.f,  64.f/1600.f, 128.f/1600.f, 256.f/1600.f, 128.f/1600.f,  64.f/1600.f,  32.f/1600.f,  16.f/1600.f;
  gaussian_kernel << kernel_values[0], kernel_values[1], kernel_values[2], kernel_values[3], kernel_values[4], kernel_values[5], kernel_values[6];

  pcl::PointCloud<pcl::RGB>::Ptr rgb_input_ (new pcl::PointCloud<pcl::RGB>());
  
  const uint32_t width = input_->width;
  const uint32_t height = input_->height;

  rgb_input_->resize (width*height);
  rgb_input_->width = width;
  rgb_input_->height = height;
  rgb_input_->is_dense = input_->is_dense;
  for (size_t row_index = 0; row_index < height; ++row_index)
  {
    for (size_t col_index = 0; col_index < width; ++col_index)
    {
      (*rgb_input_) (col_index, row_index).r = (*input_) (col_index, row_index).r;
      (*rgb_input_) (col_index, row_index).g = (*input_) (col_index, row_index).g;
      (*rgb_input_) (col_index, row_index).b = (*input_) (col_index, row_index).b;
    }
  }

    convolution.setInputCloud (rgb_input_);
    convolution.setKernel (gaussian_kernel);

  convolution.convolve (*smoothed_input_);

  // extract color gradients
  computeMaxColorGradientsSobel (smoothed_input_);

  // quantize gradients
  quantizeColorGradients ();

  // filter quantized gradients to get only dominants one + thresholding
  filterQuantizedColorGradients ();

  // spread filtered quantized gradients
  //spreadFilteredQunatizedColorGradients ();
  pcl::QuantizedMap::spreadQuantizedMap (filtered_quantized_color_gradients_,
                                         spreaded_filtered_quantized_color_gradients_, 
                                         spreading_size_);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void
pcl::ColorGradientModality<PointInT>::
processInputDataFromFiltered ()
{
  // spread filtered quantized gradients
  //spreadFilteredQunatizedColorGradients ();
  pcl::QuantizedMap::spreadQuantizedMap (filtered_quantized_color_gradients_,
                                         spreaded_filtered_quantized_color_gradients_, 
                                         spreading_size_);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void pcl::ColorGradientModality<PointInT>::
extractFeatures (const MaskMap & mask, const size_t nr_features, const size_t modality_index,
                 std::vector<QuantizedMultiModFeature> & features) const
{
  const size_t width = mask.getWidth ();
  const size_t height = mask.getHeight ();
  
  std::list<Candidate> list1;
  std::list<Candidate> list2;


  if (feature_selection_method_ == DISTANCE_MAGNITUDE_SCORE)
  {
    for (size_t row_index = 0; row_index < height; ++row_index)
    {
      for (size_t col_index = 0; col_index < width; ++col_index)
      {
        if (mask (col_index, row_index) != 0)
        {
          const GradientXY & gradient = color_gradients_ (col_index, row_index);
          if (gradient.magnitude > gradient_magnitude_threshold_feature_extraction_
            && filtered_quantized_color_gradients_ (col_index, row_index) != 0)
          {
            Candidate candidate;
            candidate.gradient = gradient;
            candidate.x = static_cast<int> (col_index);
            candidate.y = static_cast<int> (row_index);

            list1.push_back (candidate);
          }
        }
      }
    }

    list1.sort();

    if (variable_feature_nr_)
    {
      list2.push_back (*(list1.begin ()));
      //while (list2.size () != nr_features)
      bool feature_selection_finished = false;
      while (!feature_selection_finished)
      {
        float best_score = 0.0f;
        typename std::list<Candidate>::iterator best_iter = list1.end ();
        for (typename std::list<Candidate>::iterator iter1 = list1.begin (); iter1 != list1.end (); ++iter1)
        {
          // find smallest distance
          float smallest_distance = std::numeric_limits<float>::max ();
          for (typename std::list<Candidate>::iterator iter2 = list2.begin (); iter2 != list2.end (); ++iter2)
          {
            const float dx = static_cast<float> (iter1->x) - static_cast<float> (iter2->x);
            const float dy = static_cast<float> (iter1->y) - static_cast<float> (iter2->y);

            const float distance = dx*dx + dy*dy;

            if (distance < smallest_distance)
            {
              smallest_distance = distance;
            }
          }

          const float score = smallest_distance * iter1->gradient.magnitude;

          if (score > best_score)
          {
            best_score = score;
            best_iter = iter1;
          }
        }


        float min_min_sqr_distance = std::numeric_limits<float>::max ();
        float max_min_sqr_distance = 0;
        for (typename std::list<Candidate>::iterator iter2 = list2.begin (); iter2 != list2.end (); ++iter2)
        {
          float min_sqr_distance = std::numeric_limits<float>::max ();
          for (typename std::list<Candidate>::iterator iter3 = list2.begin (); iter3 != list2.end (); ++iter3)
          {
            if (iter2 == iter3)
              continue;

            const float dx = static_cast<float> (iter2->x) - static_cast<float> (iter3->x);
            const float dy = static_cast<float> (iter2->y) - static_cast<float> (iter3->y);

            const float sqr_distance = dx*dx + dy*dy;

            if (sqr_distance < min_sqr_distance)
            {
              min_sqr_distance = sqr_distance;
            }

            //std::cerr << min_sqr_distance;
          }
          //std::cerr << std::endl;

          // check current feature
          {
            const float dx = static_cast<float> (iter2->x) - static_cast<float> (best_iter->x);
            const float dy = static_cast<float> (iter2->y) - static_cast<float> (best_iter->y);

            const float sqr_distance = dx*dx + dy*dy;

            if (sqr_distance < min_sqr_distance)
            {
              min_sqr_distance = sqr_distance;
            }
          }

          if (min_sqr_distance < min_min_sqr_distance)
            min_min_sqr_distance = min_sqr_distance;
          if (min_sqr_distance > max_min_sqr_distance)
            max_min_sqr_distance = min_sqr_distance;

          //std::cerr << min_sqr_distance << ", " << min_min_sqr_distance << ", " << max_min_sqr_distance << std::endl;
        }

        if (best_iter != list1.end ())
        {
          //std::cerr << "feature_index: " << list2.size () << std::endl;
          //std::cerr << "min_min_sqr_distance: " << min_min_sqr_distance << std::endl;
          //std::cerr << "max_min_sqr_distance: " << max_min_sqr_distance << std::endl;

          if (min_min_sqr_distance < 50)
          {
            feature_selection_finished = true;
            break;
          }

          list2.push_back (*best_iter);
        }
      } 
    }
    else
    {
      if (list1.size () <= nr_features)
      {
        for (typename std::list<Candidate>::iterator iter1 = list1.begin (); iter1 != list1.end (); ++iter1)
        {
          QuantizedMultiModFeature feature;
          
          feature.x = iter1->x;
          feature.y = iter1->y;
          feature.modality_index = modality_index;
          feature.quantized_value = filtered_quantized_color_gradients_ (iter1->x, iter1->y);

          features.push_back (feature);
        }
        return;
      }

      list2.push_back (*(list1.begin ()));
      while (list2.size () != nr_features)
      {
        float best_score = 0.0f;
        typename std::list<Candidate>::iterator best_iter = list1.end ();
        for (typename std::list<Candidate>::iterator iter1 = list1.begin (); iter1 != list1.end (); ++iter1)
        {
          // find smallest distance
          float smallest_distance = std::numeric_limits<float>::max ();
          for (typename std::list<Candidate>::iterator iter2 = list2.begin (); iter2 != list2.end (); ++iter2)
          {
            const float dx = static_cast<float> (iter1->x) - static_cast<float> (iter2->x);
            const float dy = static_cast<float> (iter1->y) - static_cast<float> (iter2->y);

            const float distance = dx*dx + dy*dy;

            if (distance < smallest_distance)
            {
              smallest_distance = distance;
            }
          }

          const float score = smallest_distance * iter1->gradient.magnitude;

          if (score > best_score)
          {
            best_score = score;
            best_iter = iter1;
          }
        }

        if (best_iter != list1.end ())
        {
          list2.push_back (*best_iter);
        }
        else
        {
          break;
        }
      }  
    }
  }
  else if (feature_selection_method_ == MASK_BORDER_HIGH_GRADIENTS || feature_selection_method_ == MASK_BORDER_EQUALLY)
  {
    MaskMap eroded_mask;
    erode (mask, eroded_mask);

    MaskMap diff_mask;
    MaskMap::getDifferenceMask (mask, eroded_mask, diff_mask);

    for (size_t row_index = 0; row_index < height; ++row_index)
    {
      for (size_t col_index = 0; col_index < width; ++col_index)
      {
        if (diff_mask (col_index, row_index) != 0)
        {
          const GradientXY & gradient = color_gradients_ (col_index, row_index);
          if ((feature_selection_method_ == MASK_BORDER_EQUALLY || gradient.magnitude > gradient_magnitude_threshold_feature_extraction_)
            && filtered_quantized_color_gradients_ (col_index, row_index) != 0)
          {
            Candidate candidate;
            candidate.gradient = gradient;
            candidate.x = static_cast<int> (col_index);
            candidate.y = static_cast<int> (row_index);

            list1.push_back (candidate);
          }
        }
      }
    }

    list1.sort();

    if (list1.size () <= nr_features)
    {
      for (typename std::list<Candidate>::iterator iter1 = list1.begin (); iter1 != list1.end (); ++iter1)
      {
        QuantizedMultiModFeature feature;
          
        feature.x = iter1->x;
        feature.y = iter1->y;
        feature.modality_index = modality_index;
        feature.quantized_value = filtered_quantized_color_gradients_ (iter1->x, iter1->y);

        features.push_back (feature);
      }
      return;
    }

    size_t distance = list1.size () / nr_features + 1; // ??? 
    while (list2.size () != nr_features)
    {
      const size_t sqr_distance = distance*distance;
      for (typename std::list<Candidate>::iterator iter1 = list1.begin (); iter1 != list1.end (); ++iter1)
      {
        bool candidate_accepted = true;

        for (typename std::list<Candidate>::iterator iter2 = list2.begin (); iter2 != list2.end (); ++iter2)
        {
          const int dx = iter1->x - iter2->x;
          const int dy = iter1->y - iter2->y;
          const unsigned int tmp_distance = dx*dx + dy*dy;

          //if (tmp_distance < distance) 
          if (tmp_distance < sqr_distance)
          {
            candidate_accepted = false;
            break;
          }
        }

        if (candidate_accepted)
          list2.push_back (*iter1);

        if (list2.size () == nr_features)
          break;
      }
      --distance;
    }
  }

  for (typename std::list<Candidate>::iterator iter2 = list2.begin (); iter2 != list2.end (); ++iter2)
  {
    QuantizedMultiModFeature feature;
    
    feature.x = iter2->x;
    feature.y = iter2->y;
    feature.modality_index = modality_index;
    feature.quantized_value = filtered_quantized_color_gradients_ (iter2->x, iter2->y);

    features.push_back (feature);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT> void 
pcl::ColorGradientModality<PointInT>::
extractAllFeatures (const MaskMap & mask, const size_t, const size_t modality_index,
                 std::vector<QuantizedMultiModFeature> & features) const
{
  const size_t width = mask.getWidth ();
  const size_t height = mask.getHeight ();
  
  std::list<Candidate> list1;
  std::list<Candidate> list2;


  for (size_t row_index = 0; row_index < height; ++row_index)
  {
    for (size_t col_index = 0; col_index < width; ++col_index)
    {
      if (mask (col_index, row_index) != 0)
      {
        const GradientXY & gradient = color_gradients_ (col_index, row_index);
        if (gradient.magnitude > gradient_magnitude_threshold_feature_extraction_
          && filtered_quantized_color_gradients_ (col_index, row_index) != 0)
        {
          Candidate candidate;
          candidate.gradient = gradient;
          candidate.x = static_cast<int> (col_index);
          candidate.y = static_cast<int> (row_index);

          list1.push_back (candidate);
        }
      }
    }
  }

  list1.sort();

  for (typename std::list<Candidate>::iterator iter1 = list1.begin (); iter1 != list1.end (); ++iter1)
  {
    QuantizedMultiModFeature feature;
          
    feature.x = iter1->x;
    feature.y = iter1->y;
    feature.modality_index = modality_index;
    feature.quantized_value = filtered_quantized_color_gradients_ (iter1->x, iter1->y);

    features.push_back (feature);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void
pcl::ColorGradientModality<PointInT>::
computeMaxColorGradients (const typename pcl::PointCloud<pcl::RGB>::ConstPtr & cloud)
{
  const int width = cloud->width;
  const int height = cloud->height;

  color_gradients_.points.resize (width*height);
  color_gradients_.width = width;
  color_gradients_.height = height;

  const float pi = tan (1.0f) * 2;
  for (int row_index = 0; row_index < height-2; ++row_index)
  {
    for (int col_index = 0; col_index < width-2; ++col_index)
    {
      const int index0 = row_index*width+col_index;
      const int index_c = row_index*width+col_index+2;
      const int index_r = (row_index+2)*width+col_index;

      //const int index_d = (row_index+1)*width+col_index+1;

      const unsigned char r0 = cloud->points[index0].r;
      const unsigned char g0 = cloud->points[index0].g;
      const unsigned char b0 = cloud->points[index0].b;

      const unsigned char r_c = cloud->points[index_c].r;
      const unsigned char g_c = cloud->points[index_c].g;
      const unsigned char b_c = cloud->points[index_c].b;

      const unsigned char r_r = cloud->points[index_r].r;
      const unsigned char g_r = cloud->points[index_r].g;
      const unsigned char b_r = cloud->points[index_r].b;

      const float r_dx = static_cast<float> (r_c) - static_cast<float> (r0);
      const float g_dx = static_cast<float> (g_c) - static_cast<float> (g0);
      const float b_dx = static_cast<float> (b_c) - static_cast<float> (b0);

      const float r_dy = static_cast<float> (r_r) - static_cast<float> (r0);
      const float g_dy = static_cast<float> (g_r) - static_cast<float> (g0);
      const float b_dy = static_cast<float> (b_r) - static_cast<float> (b0);

      const float sqr_mag_r = r_dx*r_dx + r_dy*r_dy;
      const float sqr_mag_g = g_dx*g_dx + g_dy*g_dy;
      const float sqr_mag_b = b_dx*b_dx + b_dy*b_dy;

      if (sqr_mag_r > sqr_mag_g && sqr_mag_r > sqr_mag_b)
      {
        GradientXY gradient;
        gradient.magnitude = sqrt (sqr_mag_r);
        gradient.angle = atan2 (r_dy, r_dx) * 180.0f / pi;
        gradient.x = static_cast<float> (col_index);
        gradient.y = static_cast<float> (row_index);

        color_gradients_ (col_index+1, row_index+1) = gradient;
      }
      else if (sqr_mag_g > sqr_mag_b)
      {
        GradientXY gradient;
        gradient.magnitude = sqrt (sqr_mag_g);
        gradient.angle = atan2 (g_dy, g_dx) * 180.0f / pi;
        gradient.x = static_cast<float> (col_index);
        gradient.y = static_cast<float> (row_index);

        color_gradients_ (col_index+1, row_index+1) = gradient;
      }
      else
      {
        GradientXY gradient;
        gradient.magnitude = sqrt (sqr_mag_b);
        gradient.angle = atan2 (b_dy, b_dx) * 180.0f / pi;
        gradient.x = static_cast<float> (col_index);
        gradient.y = static_cast<float> (row_index);

        color_gradients_ (col_index+1, row_index+1) = gradient;
      }

      assert (color_gradients_ (col_index+1, row_index+1).angle >= -180 &&
              color_gradients_ (col_index+1, row_index+1).angle <=  180);
    }
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void
pcl::ColorGradientModality<PointInT>::
computeMaxColorGradientsSobel (const typename pcl::PointCloud<pcl::RGB>::ConstPtr & cloud)
{
  const int width = cloud->width;
  const int height = cloud->height;

  color_gradients_.points.resize (width*height);
  color_gradients_.width = width;
  color_gradients_.height = height;

  const float pi = tanf (1.0f) * 2.0f;
  for (int row_index = 1; row_index < height-1; ++row_index)
  {
    for (int col_index = 1; col_index < width-1; ++col_index)
    {
      const int r7 = static_cast<int> (cloud->points[(row_index-1)*width + (col_index-1)].r);
      const int g7 = static_cast<int> (cloud->points[(row_index-1)*width + (col_index-1)].g);
      const int b7 = static_cast<int> (cloud->points[(row_index-1)*width + (col_index-1)].b);
      const int r8 = static_cast<int> (cloud->points[(row_index-1)*width + (col_index)].r);
      const int g8 = static_cast<int> (cloud->points[(row_index-1)*width + (col_index)].g);
      const int b8 = static_cast<int> (cloud->points[(row_index-1)*width + (col_index)].b);
      const int r9 = static_cast<int> (cloud->points[(row_index-1)*width + (col_index+1)].r);
      const int g9 = static_cast<int> (cloud->points[(row_index-1)*width + (col_index+1)].g);
      const int b9 = static_cast<int> (cloud->points[(row_index-1)*width + (col_index+1)].b);
      const int r4 = static_cast<int> (cloud->points[(row_index)*width + (col_index-1)].r);
      const int g4 = static_cast<int> (cloud->points[(row_index)*width + (col_index-1)].g);
      const int b4 = static_cast<int> (cloud->points[(row_index)*width + (col_index-1)].b);
      const int r6 = static_cast<int> (cloud->points[(row_index)*width + (col_index+1)].r);
      const int g6 = static_cast<int> (cloud->points[(row_index)*width + (col_index+1)].g);
      const int b6 = static_cast<int> (cloud->points[(row_index)*width + (col_index+1)].b);
      const int r1 = static_cast<int> (cloud->points[(row_index+1)*width + (col_index-1)].r);
      const int g1 = static_cast<int> (cloud->points[(row_index+1)*width + (col_index-1)].g);
      const int b1 = static_cast<int> (cloud->points[(row_index+1)*width + (col_index-1)].b);
      const int r2 = static_cast<int> (cloud->points[(row_index+1)*width + (col_index)].r);
      const int g2 = static_cast<int> (cloud->points[(row_index+1)*width + (col_index)].g);
      const int b2 = static_cast<int> (cloud->points[(row_index+1)*width + (col_index)].b);
      const int r3 = static_cast<int> (cloud->points[(row_index+1)*width + (col_index+1)].r);
      const int g3 = static_cast<int> (cloud->points[(row_index+1)*width + (col_index+1)].g);
      const int b3 = static_cast<int> (cloud->points[(row_index+1)*width + (col_index+1)].b);

      //const int r_tmp1 = - r7 + r3;
      //const int r_tmp2 = - r1 + r9;
      //const int g_tmp1 = - g7 + g3;
      //const int g_tmp2 = - g1 + g9;
      //const int b_tmp1 = - b7 + b3;
      //const int b_tmp2 = - b1 + b9;
      ////const int gx = - r7 - (r4<<2) - r1 + r3 + (r6<<2) + r9;
      ////const int gy = - r7 - (r8<<2) - r9 + r1 + (r2<<2) + r3;
      //const int r_dx = r_tmp1 + r_tmp2 - (r4<<2) + (r6<<2);
      //const int r_dy = r_tmp1 - r_tmp2 - (r8<<2) + (r2<<2);
      //const int g_dx = g_tmp1 + g_tmp2 - (g4<<2) + (g6<<2);
      //const int g_dy = g_tmp1 - g_tmp2 - (g8<<2) + (g2<<2);
      //const int b_dx = b_tmp1 + b_tmp2 - (b4<<2) + (b6<<2);
      //const int b_dy = b_tmp1 - b_tmp2 - (b8<<2) + (b2<<2);

      //const int r_tmp1 = - r7 + r3;
      //const int r_tmp2 = - r1 + r9;
      //const int g_tmp1 = - g7 + g3;
      //const int g_tmp2 = - g1 + g9;
      //const int b_tmp1 = - b7 + b3;
      //const int b_tmp2 = - b1 + b9;
      //const int gx = - r7 - (r4<<2) - r1 + r3 + (r6<<2) + r9;
      //const int gy = - r7 - (r8<<2) - r9 + r1 + (r2<<2) + r3;
      const int r_dx = r9 + 2*r6 + r3 - (r7 + 2*r4 + r1);
      const int r_dy = r1 + 2*r2 + r3 - (r7 + 2*r8 + r9);
      const int g_dx = g9 + 2*g6 + g3 - (g7 + 2*g4 + g1);
      const int g_dy = g1 + 2*g2 + g3 - (g7 + 2*g8 + g9);
      const int b_dx = b9 + 2*b6 + b3 - (b7 + 2*b4 + b1);
      const int b_dy = b1 + 2*b2 + b3 - (b7 + 2*b8 + b9);

      const int sqr_mag_r = r_dx*r_dx + r_dy*r_dy;
      const int sqr_mag_g = g_dx*g_dx + g_dy*g_dy;
      const int sqr_mag_b = b_dx*b_dx + b_dy*b_dy;

      if (sqr_mag_r > sqr_mag_g && sqr_mag_r > sqr_mag_b)
      {
        GradientXY gradient;
        gradient.magnitude = sqrtf (static_cast<float> (sqr_mag_r));
        gradient.angle = atan2f (static_cast<float> (r_dy), static_cast<float> (r_dx)) * 180.0f / pi;
        if (gradient.angle < -180.0f) gradient.angle += 360.0f;
        if (gradient.angle >= 180.0f) gradient.angle -= 360.0f;
        gradient.x = static_cast<float> (col_index);
        gradient.y = static_cast<float> (row_index);

        color_gradients_ (col_index, row_index) = gradient;
      }
      else if (sqr_mag_g > sqr_mag_b)
      {
        GradientXY gradient;
        gradient.magnitude = sqrtf (static_cast<float> (sqr_mag_g));
        gradient.angle = atan2f (static_cast<float> (g_dy), static_cast<float> (g_dx)) * 180.0f / pi;
        if (gradient.angle < -180.0f) gradient.angle += 360.0f;
        if (gradient.angle >= 180.0f) gradient.angle -= 360.0f;
        gradient.x = static_cast<float> (col_index);
        gradient.y = static_cast<float> (row_index);

        color_gradients_ (col_index, row_index) = gradient;
      }
      else
      {
        GradientXY gradient;
        gradient.magnitude = sqrtf (static_cast<float> (sqr_mag_b));
        gradient.angle = atan2f (static_cast<float> (b_dy), static_cast<float> (b_dx)) * 180.0f / pi;
        if (gradient.angle < -180.0f) gradient.angle += 360.0f;
        if (gradient.angle >= 180.0f) gradient.angle -= 360.0f;
        gradient.x = static_cast<float> (col_index);
        gradient.y = static_cast<float> (row_index);

        color_gradients_ (col_index, row_index) = gradient;
      }

      assert (color_gradients_ (col_index, row_index).angle >= -180 &&
              color_gradients_ (col_index, row_index).angle <=  180);
    }
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void
pcl::ColorGradientModality<PointInT>::
quantizeColorGradients ()
{
  //std::cerr << "quantize this, bastard!!!" << std::endl;

  //unsigned char quantization_map[16] = {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7};
  //unsigned char quantization_map[16] = {1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8};

  //for (float angle = 0.0f; angle < 360.0f; angle += 1.0f)
  //{
  //  const int quantized_value = quantization_map[static_cast<int> (angle * angleScale)];
  //  std::cerr << angle << ": " << quantized_value << std::endl;
  //}


  const size_t width = input_->width;
  const size_t height = input_->height;

  quantized_color_gradients_.resize (width, height);

  const float angleScale = 16.0f/360.0f;

  //float min_angle = std::numeric_limits<float>::max ();
  //float max_angle = -std::numeric_limits<float>::max ();
  for (size_t row_index = 0; row_index < height; ++row_index)
  {
    for (size_t col_index = 0; col_index < width; ++col_index)
    {
      if (color_gradients_ (col_index, row_index).magnitude < gradient_magnitude_threshold_) 
      {
        quantized_color_gradients_ (col_index, row_index) = 0;
        continue;
      }

      const float angle = 11.25f + color_gradients_ (col_index, row_index).angle + 180.0f;
      const int quantized_value = (static_cast<int> (angle * angleScale)) & 7;
      quantized_color_gradients_ (col_index, row_index) = static_cast<unsigned char> (quantized_value + 1); 

      //const float angle = color_gradients_ (col_index, row_index).angle + 180.0f;

      //min_angle = std::min (min_angle, angle);
      //max_angle = std::max (max_angle, angle);

      //if (angle < 0.0f || angle >= 360.0f)
      //{
      //  std::cerr << "angle shitty: " << angle << std::endl;
      //}

      //const int quantized_value = quantization_map[static_cast<int> (angle * angleScale)];
      //quantized_color_gradients_ (col_index, row_index) = static_cast<unsigned char> (quantized_value); 

      //assert (0 <= quantized_value && quantized_value < 16);
      //quantized_color_gradients_ (col_index, row_index) = quantization_map[quantized_value];
      //quantized_color_gradients_ (col_index, row_index) = static_cast<unsigned char> ((quantized_value & 7) + 1); // = (quantized_value % 8) + 1
    }
  }

  //std::cerr << ">>>>> " << min_angle << ", " << max_angle << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void
pcl::ColorGradientModality<PointInT>::
filterQuantizedColorGradients ()
{
  const size_t width = input_->width;
  const size_t height = input_->height;

  filtered_quantized_color_gradients_.resize (width, height);

  // filter data
  for (size_t row_index = 1; row_index < height-1; ++row_index)
  {
    for (size_t col_index = 1; col_index < width-1; ++col_index)
    {
      unsigned char histogram[9] = {0,0,0,0,0,0,0,0,0};

      {
        const unsigned char * data_ptr = quantized_color_gradients_.getData () + (row_index-1)*width+col_index-1;
        assert (data_ptr[0] < 9 && data_ptr[1] < 9 && data_ptr[2] < 9);
        ++histogram[data_ptr[0]];
        ++histogram[data_ptr[1]];
        ++histogram[data_ptr[2]];
      }
      {
        const unsigned char * data_ptr = quantized_color_gradients_.getData () + row_index*width+col_index-1;
        assert (data_ptr[0] < 9 && data_ptr[1] < 9 && data_ptr[2] < 9);
        ++histogram[data_ptr[0]];
        ++histogram[data_ptr[1]];
        ++histogram[data_ptr[2]];
      }
      {
        const unsigned char * data_ptr = quantized_color_gradients_.getData () + (row_index+1)*width+col_index-1;
        assert (data_ptr[0] < 9 && data_ptr[1] < 9 && data_ptr[2] < 9);
        ++histogram[data_ptr[0]];
        ++histogram[data_ptr[1]];
        ++histogram[data_ptr[2]];
      }

      unsigned char max_hist_value = 0;
      int max_hist_index = -1;

      // for (int i = 0; i < 8; ++i)
      // {
      //   if (max_hist_value < histogram[i+1])
      //   {
      //     max_hist_index = i;
      //     max_hist_value = histogram[i+1]
      //   }
      // }
      // Unrolled for performance optimization:
      if (max_hist_value < histogram[1]) {max_hist_index = 0; max_hist_value = histogram[1];}
      if (max_hist_value < histogram[2]) {max_hist_index = 1; max_hist_value = histogram[2];}
      if (max_hist_value < histogram[3]) {max_hist_index = 2; max_hist_value = histogram[3];}
      if (max_hist_value < histogram[4]) {max_hist_index = 3; max_hist_value = histogram[4];}
      if (max_hist_value < histogram[5]) {max_hist_index = 4; max_hist_value = histogram[5];}
      if (max_hist_value < histogram[6]) {max_hist_index = 5; max_hist_value = histogram[6];}
      if (max_hist_value < histogram[7]) {max_hist_index = 6; max_hist_value = histogram[7];}
      if (max_hist_value < histogram[8]) {max_hist_index = 7; max_hist_value = histogram[8];}

      if (max_hist_index != -1 && max_hist_value >= 5)
        filtered_quantized_color_gradients_ (col_index, row_index) = static_cast<unsigned char> (0x1 << max_hist_index);
      else
        filtered_quantized_color_gradients_ (col_index, row_index) = 0;

    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void
pcl::ColorGradientModality<PointInT>::
erode (const pcl::MaskMap & mask_in, 
       pcl::MaskMap & mask_out)
{
  const size_t width = mask_in.getWidth ();
  const size_t height = mask_in.getHeight ();

  mask_out.resize (width, height);

  for (size_t row_index = 1; row_index < height-1; ++row_index)
  {
    for (size_t col_index = 1; col_index < width-1; ++col_index)
    {
      if (mask_in (col_index, row_index-1) == 0 ||
          mask_in (col_index-1, row_index) == 0 ||
          mask_in (col_index+1, row_index) == 0 ||
          mask_in (col_index, row_index+1) == 0)
      {
        mask_out (col_index, row_index) = 0;
      }
      else
      {
        mask_out (col_index, row_index) = 255;
      }
    }
  }
}

#endif 

###

# color_modality.h
namespace pcl
{

  // --------------------------------------------------------------------------

  template <typename PointInT>
  class ColorModality
    : public QuantizableModality, public PCLBase<PointInT>
  {
    protected:
      using PCLBase<PointInT>::input_;

      struct Candidate
      {
        float distance;

        unsigned char bin_index;
    
        size_t x;
        size_t y;   

        bool 
        operator< (const Candidate & rhs)
        {
          return (distance > rhs.distance);
        }
      };

    public:
      typedef typename pcl::PointCloud<PointInT> PointCloudIn;

      ColorModality ();
  
      virtual ~ColorModality ();
  
      inline QuantizedMap &
      getQuantizedMap () 
      { 
        return (filtered_quantized_colors_);
      }
  
      inline QuantizedMap &
      getSpreadedQuantizedMap () 
      { 
        return (spreaded_filtered_quantized_colors_);
      }
  
      void
      extractFeatures (const MaskMap & mask, size_t nr_features, size_t modalityIndex,
                       std::vector<QuantizedMultiModFeature> & features) const;
  
      /** \brief Provide a pointer to the input dataset (overwrites the PCLBase::setInputCloud method)
        * \param cloud the const boost shared pointer to a PointCloud message
        */
      virtual void 
      setInputCloud (const typename PointCloudIn::ConstPtr & cloud) 
      { 
        input_ = cloud;
      }

      virtual void
      processInputData ();

    protected:

      void
      quantizeColors ();
  
      void
      filterQuantizedColors ();

      static inline int
      quantizeColorOnRGBExtrema (const float r,
                                 const float g,
                                 const float b);
  
      void
      computeDistanceMap (const MaskMap & input, DistanceMap & output) const;

    private:
      float feature_distance_threshold_;
      
      pcl::QuantizedMap quantized_colors_;
      pcl::QuantizedMap filtered_quantized_colors_;
      pcl::QuantizedMap spreaded_filtered_quantized_colors_;
  
  };

}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
pcl::ColorModality<PointInT>::ColorModality ()
  : feature_distance_threshold_ (1.0f), quantized_colors_ (), filtered_quantized_colors_ (), spreaded_filtered_quantized_colors_ ()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
pcl::ColorModality<PointInT>::~ColorModality ()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void
pcl::ColorModality<PointInT>::processInputData ()
{
  // quantize gradients
  quantizeColors ();

  // filter quantized gradients to get only dominants one + thresholding
  filterQuantizedColors ();

  // spread filtered quantized gradients
  //spreadFilteredQunatizedColorGradients ();
  const int spreading_size = 8;
  pcl::QuantizedMap::spreadQuantizedMap (filtered_quantized_colors_,
                                         spreaded_filtered_quantized_colors_, spreading_size);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void pcl::ColorModality<PointInT>::extractFeatures (const MaskMap & mask, 
                                                    const size_t nr_features, 
                                                    const size_t modality_index,
                                                    std::vector<QuantizedMultiModFeature> & features) const
{
  const size_t width = mask.getWidth ();
  const size_t height = mask.getHeight ();

  MaskMap mask_maps[8];
  for (size_t map_index = 0; map_index < 8; ++map_index)
    mask_maps[map_index].resize (width, height);

  unsigned char map[255];
  memset(map, 0, 255);

  map[0x1<<0] = 0;
  map[0x1<<1] = 1;
  map[0x1<<2] = 2;
  map[0x1<<3] = 3;
  map[0x1<<4] = 4;
  map[0x1<<5] = 5;
  map[0x1<<6] = 6;
  map[0x1<<7] = 7;

  QuantizedMap distance_map_indices (width, height);
  //memset (distance_map_indices.data, 0, sizeof (distance_map_indices.data[0])*width*height);

  for (size_t row_index = 0; row_index < height; ++row_index)
  {
    for (size_t col_index = 0; col_index < width; ++col_index)
    {
      if (mask (col_index, row_index) != 0)
      {
        //const unsigned char quantized_value = quantized_surface_normals_ (row_index, col_index);
        const unsigned char quantized_value = filtered_quantized_colors_ (col_index, row_index);

        if (quantized_value == 0) 
          continue;
        const int dist_map_index = map[quantized_value];

        distance_map_indices (col_index, row_index) = dist_map_index;
        //distance_maps[dist_map_index].at<unsigned char>(row_index, col_index) = 255;
        mask_maps[dist_map_index] (col_index, row_index) = 255;
      }
    }
  }

  DistanceMap distance_maps[8];
  for (int map_index = 0; map_index < 8; ++map_index)
    computeDistanceMap (mask_maps[map_index], distance_maps[map_index]);

  std::list<Candidate> list1;
  std::list<Candidate> list2;

  float weights[8] = {0,0,0,0,0,0,0,0};

  const size_t off = 4;
  for (size_t row_index = off; row_index < height-off; ++row_index)
  {
    for (size_t col_index = off; col_index < width-off; ++col_index)
    {
      if (mask (col_index, row_index) != 0)
      {
        //const unsigned char quantized_value = quantized_surface_normals_ (row_index, col_index);
        const unsigned char quantized_value = filtered_quantized_colors_ (col_index, row_index);

        //const float nx = surface_normals_ (col_index, row_index).normal_x;
        //const float ny = surface_normals_ (col_index, row_index).normal_y;
        //const float nz = surface_normals_ (col_index, row_index).normal_z;

        if (quantized_value != 0)
        {
          const int distance_map_index = map[quantized_value];

          //const float distance = distance_maps[distance_map_index].at<float> (row_index, col_index);
          const float distance = distance_maps[distance_map_index] (col_index, row_index);

          if (distance >= feature_distance_threshold_)
          {
            Candidate candidate;

            candidate.distance = distance;
            candidate.x = col_index;
            candidate.y = row_index;
            candidate.bin_index = distance_map_index;

            list1.push_back (candidate);

            ++weights[distance_map_index];
          }
        }
      }
    }
  }

  for (typename std::list<Candidate>::iterator iter = list1.begin (); iter != list1.end (); ++iter)
    iter->distance *= 1.0f / weights[iter->bin_index];

  list1.sort ();

  if (list1.size () <= nr_features)
  {
    features.reserve (list1.size ());
    for (typename std::list<Candidate>::iterator iter = list1.begin (); iter != list1.end (); ++iter)
    {
      QuantizedMultiModFeature feature;

      feature.x = static_cast<int> (iter->x);
      feature.y = static_cast<int> (iter->y);
      feature.modality_index = modality_index;
      feature.quantized_value = filtered_quantized_colors_ (iter->x, iter->y);

      features.push_back (feature);
    }

    return;
  }

  int distance = static_cast<int> (list1.size () / nr_features + 1); // ???  @todo:!:!:!:!:!:!
  while (list2.size () != nr_features)
  {
    const int sqr_distance = distance*distance;
    for (typename std::list<Candidate>::iterator iter1 = list1.begin (); iter1 != list1.end (); ++iter1)
    {
      bool candidate_accepted = true;

      for (typename std::list<Candidate>::iterator iter2 = list2.begin (); iter2 != list2.end (); ++iter2)
      {
        const int dx = static_cast<int> (iter1->x) - static_cast<int> (iter2->x);
        const int dy = static_cast<int> (iter1->y) - static_cast<int> (iter2->y);
        const int tmp_distance = dx*dx + dy*dy;

        if (tmp_distance < sqr_distance)
        {
          candidate_accepted = false;
          break;
        }
      }

      if (candidate_accepted)
        list2.push_back (*iter1);

      if (list2.size () == nr_features) break;
    }
    --distance;
  }

  for (typename std::list<Candidate>::iterator iter2 = list2.begin (); iter2 != list2.end (); ++iter2)
  {
    QuantizedMultiModFeature feature;

    feature.x = static_cast<int> (iter2->x);
    feature.y = static_cast<int> (iter2->y);
    feature.modality_index = modality_index;
    feature.quantized_value = filtered_quantized_colors_ (iter2->x, iter2->y);

    features.push_back (feature);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void
pcl::ColorModality<PointInT>::quantizeColors ()
{
  const size_t width = input_->width;
  const size_t height = input_->height;

  quantized_colors_.resize (width, height);

  for (size_t row_index = 0; row_index < height; ++row_index)
  {
    for (size_t col_index = 0; col_index < width; ++col_index)
    {
      const float r = static_cast<float> ((*input_) (col_index, row_index).r);
      const float g = static_cast<float> ((*input_) (col_index, row_index).g);
      const float b = static_cast<float> ((*input_) (col_index, row_index).b);

      quantized_colors_ (col_index, row_index) = quantizeColorOnRGBExtrema (r, g, b);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void
pcl::ColorModality<PointInT>::filterQuantizedColors ()
{
  const size_t width = input_->width;
  const size_t height = input_->height;

  filtered_quantized_colors_.resize (width, height);

  // filter data
  for (size_t row_index = 1; row_index < height-1; ++row_index)
  {
    for (size_t col_index = 1; col_index < width-1; ++col_index)
    {
      unsigned char histogram[8] = {0,0,0,0,0,0,0,0};

      {
        const unsigned char * data_ptr = quantized_colors_.getData () + (row_index-1)*width+col_index-1;
        assert (0 <= data_ptr[0] && data_ptr[0] < 9 && 
                0 <= data_ptr[1] && data_ptr[1] < 9 && 
                0 <= data_ptr[2] && data_ptr[2] < 9);
        ++histogram[data_ptr[0]];
        ++histogram[data_ptr[1]];
        ++histogram[data_ptr[2]];
      }
      {
        const unsigned char * data_ptr = quantized_colors_.getData () + row_index*width+col_index-1;
        assert (0 <= data_ptr[0] && data_ptr[0] < 9 && 
                0 <= data_ptr[1] && data_ptr[1] < 9 && 
                0 <= data_ptr[2] && data_ptr[2] < 9);
        ++histogram[data_ptr[0]];
        ++histogram[data_ptr[1]];
        ++histogram[data_ptr[2]];
      }
      {
        const unsigned char * data_ptr = quantized_colors_.getData () + (row_index+1)*width+col_index-1;
        assert (0 <= data_ptr[0] && data_ptr[0] < 9 && 
                0 <= data_ptr[1] && data_ptr[1] < 9 && 
                0 <= data_ptr[2] && data_ptr[2] < 9);
        ++histogram[data_ptr[0]];
        ++histogram[data_ptr[1]];
        ++histogram[data_ptr[2]];
      }

      unsigned char max_hist_value = 0;
      int max_hist_index = -1;

      // for (int i = 0; i < 8; ++i)
      // {
      //   if (max_hist_value < histogram[i+1])
      //   {
      //     max_hist_index = i;
      //     max_hist_value = histogram[i+1]
      //   }
      // }
      // Unrolled for performance optimization:
      if (max_hist_value < histogram[0]) {max_hist_index = 0; max_hist_value = histogram[0];}
      if (max_hist_value < histogram[1]) {max_hist_index = 1; max_hist_value = histogram[1];}
      if (max_hist_value < histogram[2]) {max_hist_index = 2; max_hist_value = histogram[2];}
      if (max_hist_value < histogram[3]) {max_hist_index = 3; max_hist_value = histogram[3];}
      if (max_hist_value < histogram[4]) {max_hist_index = 4; max_hist_value = histogram[4];}
      if (max_hist_value < histogram[5]) {max_hist_index = 5; max_hist_value = histogram[5];}
      if (max_hist_value < histogram[6]) {max_hist_index = 6; max_hist_value = histogram[6];}
      if (max_hist_value < histogram[7]) {max_hist_index = 7; max_hist_value = histogram[7];}

      //if (max_hist_index != -1 && max_hist_value >= 5)
        filtered_quantized_colors_ (col_index, row_index) = 0x1 << max_hist_index;
      //else
      //  filtered_quantized_color_gradients_ (col_index, row_index) = 0;

    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
int
pcl::ColorModality<PointInT>::quantizeColorOnRGBExtrema (const float r,
                                                         const float g,
                                                         const float b)
{
  const float r_inv = 255.0f-r;
  const float g_inv = 255.0f-g;
  const float b_inv = 255.0f-b;

  const float dist_0 = (r*r + g*g + b*b)*2.0f;
  const float dist_1 = r*r + g*g + b_inv*b_inv;
  const float dist_2 = r*r + g_inv*g_inv+ b*b;
  const float dist_3 = r*r + g_inv*g_inv + b_inv*b_inv;
  const float dist_4 = r_inv*r_inv + g*g + b*b;
  const float dist_5 = r_inv*r_inv + g*g + b_inv*b_inv;
  const float dist_6 = r_inv*r_inv + g_inv*g_inv+ b*b;
  const float dist_7 = (r_inv*r_inv + g_inv*g_inv + b_inv*b_inv)*1.5f;

  const float min_dist = std::min (std::min (std::min (dist_0, dist_1), std::min (dist_2, dist_3)), std::min (std::min (dist_4, dist_5), std::min (dist_6, dist_7)));

  if (min_dist == dist_0)
  {
    return 0;
  }
  if (min_dist == dist_1)
  {
    return 1;
  }
  if (min_dist == dist_2)
  {
    return 2;
  }
  if (min_dist == dist_3)
  {
    return 3;
  }
  if (min_dist == dist_4)
  {
    return 4;
  }
  if (min_dist == dist_5)
  {
    return 5;
  }
  if (min_dist == dist_6)
  {
    return 6;
  }
  return 7;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT> void
pcl::ColorModality<PointInT>::computeDistanceMap (const MaskMap & input, 
                                                  DistanceMap & output) const
{
  const size_t width = input.getWidth ();
  const size_t height = input.getHeight ();

  output.resize (width, height);

  // compute distance map
  //float *distance_map = new float[input_->points.size ()];
  const unsigned char * mask_map = input.getData ();
  float * distance_map = output.getData ();
  for (size_t index = 0; index < width*height; ++index)
  {
    if (mask_map[index] == 0)
      distance_map[index] = 0.0f;
    else
      distance_map[index] = static_cast<float> (width + height);
  }

  // first pass
  float * previous_row = distance_map;
  float * current_row = previous_row + width;
  for (size_t ri = 1; ri < height; ++ri)
  {
    for (size_t ci = 1; ci < width; ++ci)
    {
      const float up_left  = previous_row [ci - 1] + 1.4f; //distance_map[(ri-1)*input_->width + ci-1] + 1.4f;
      const float up       = previous_row [ci]     + 1.0f; //distance_map[(ri-1)*input_->width + ci] + 1.0f;
      const float up_right = previous_row [ci + 1] + 1.4f; //distance_map[(ri-1)*input_->width + ci+1] + 1.4f;
      const float left     = current_row  [ci - 1] + 1.0f; //distance_map[ri*input_->width + ci-1] + 1.0f;
      const float center   = current_row  [ci];            //distance_map[ri*input_->width + ci];

      const float min_value = std::min (std::min (up_left, up), std::min (left, up_right));

      if (min_value < center)
        current_row[ci] = min_value; //distance_map[ri * input_->width + ci] = min_value;
    }
    previous_row = current_row;
    current_row += width;
  }

  // second pass
  float * next_row = distance_map + width * (height - 1);
  current_row = next_row - width;
  for (int ri = static_cast<int> (height)-2; ri >= 0; --ri)
  {
    for (int ci = static_cast<int> (width)-2; ci >= 0; --ci)
    {
      const float lower_left  = next_row    [ci - 1] + 1.4f; //distance_map[(ri+1)*input_->width + ci-1] + 1.4f;
      const float lower       = next_row    [ci]     + 1.0f; //distance_map[(ri+1)*input_->width + ci] + 1.0f;
      const float lower_right = next_row    [ci + 1] + 1.4f; //distance_map[(ri+1)*input_->width + ci+1] + 1.4f;
      const float right       = current_row [ci + 1] + 1.0f; //distance_map[ri*input_->width + ci+1] + 1.0f;
      const float center      = current_row [ci];            //distance_map[ri*input_->width + ci];

      const float min_value = std::min (std::min (lower_left, lower), std::min (right, lower_right));

      if (min_value < center)
        current_row[ci] = min_value; //distance_map[ri*input_->width + ci] = min_value;
    }
    next_row = current_row;
    current_row -= width;
  }
}


#endif 

###

# crh_alignment.h
namespace pcl
{

  /** \brief CRHAlignment uses two Camera Roll Histograms (CRH) to find the
   * roll rotation that aligns both views. See:
   *   - CAD-Model Recognition and 6 DOF Pose Estimation
   *     A. Aldoma, N. Blodow, D. Gossow, S. Gedikli, R.B. Rusu, M. Vincze and G. Bradski
   *     ICCV 2011, 3D Representation and Recognition (3dRR11) workshop
   *     Barcelona, Spain, (2011)
   *
   * \author Aitor Aldoma
   * \ingroup recognition
   */

  template<typename PointT, int nbins_>
    class PCL_EXPORTS CRHAlignment
    {
    private:

      /** \brief Sorts peaks */
      typedef struct
      {
        bool
        operator() (std::pair<float, int> const& a, std::pair<float, int> const& b)
        {
          return a.first > b.first;
        }
      } peaks_ordering;

      typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;

      /** \brief View of the model to be aligned to input_view_ */
      PointTPtr target_view_;
      /** \brief View of the input */
      PointTPtr input_view_;
      /** \brief Centroid of the model_view_ */
      Eigen::Vector3f centroid_target_;
      /** \brief Centroid of the input_view_ */
      Eigen::Vector3f centroid_input_;
      /** \brief transforms from model view to input view */
      std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_;
      /** \brief Allowed maximum number of peaks  */
      int max_peaks_;
      /** \brief Quantile of peaks after sorting to be checked  */
      float quantile_;
      /** \brief Threshold for a peak to be accepted.
       * If peak_i >= (max_peak * accept_threhsold_) => peak is accepted
       */
      float accept_threshold_;

      /** \brief computes the transformation to the z-axis
        * \param[in] centroid
        * \param[out] trasnformation to z-axis
        */
      void
      computeTransformToZAxes (Eigen::Vector3f & centroid, Eigen::Affine3f & transform)
      {
        Eigen::Vector3f plane_normal;
        plane_normal[0] = -centroid[0];
        plane_normal[1] = -centroid[1];
        plane_normal[2] = -centroid[2];
        Eigen::Vector3f z_vector = Eigen::Vector3f::UnitZ ();
        plane_normal.normalize ();
        Eigen::Vector3f axis = plane_normal.cross (z_vector);
        double rotation = -asin (axis.norm ());
        axis.normalize ();
        transform = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float>(rotation), axis));
      }

      /** \brief computes the roll transformation
        * \param[in] centroid input
        * \param[in] centroid view
        * \param[in] roll_angle
        * \param[out] roll transformation
        */
      void
      computeRollTransform (Eigen::Vector3f & centroidInput, Eigen::Vector3f & centroidResult, double roll_angle, Eigen::Affine3f & final_trans)
      {
        Eigen::Affine3f transformInputToZ;
        computeTransformToZAxes (centroidInput, transformInputToZ);

        transformInputToZ = transformInputToZ.inverse ();
        Eigen::Affine3f transformRoll (Eigen::AngleAxisf (-static_cast<float>(roll_angle * M_PI / 180), Eigen::Vector3f::UnitZ ()));
        Eigen::Affine3f transformDBResultToZ;
        computeTransformToZAxes (centroidResult, transformDBResultToZ);

        final_trans = transformInputToZ * transformRoll * transformDBResultToZ;
      }
    public:

      /** \brief Constructor. */
      CRHAlignment() {
        max_peaks_ = 5;
        quantile_ = 0.2f;
        accept_threshold_ = 0.8f;
      }

      /** \brief returns the computed transformations
       * \param[out] transforms transformations
       */
      void getTransforms(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & transforms) {
        transforms = transforms_;
      }

      /** \brief sets model and input views
       * \param[in] input_view
       * \param[in] target_view
       */
      void
      setInputAndTargetView (PointTPtr & input_view, PointTPtr & target_view)
      {
        target_view_ = target_view;
        input_view_ = input_view;
      }

      /** \brief sets model and input centroids
        * \param[in] c1 model view centroid
        * \param[in] c2 input view centroid
        */
      void
      setInputAndTargetCentroids (Eigen::Vector3f & c1, Eigen::Vector3f & c2)
      {
        centroid_target_ = c2;
        centroid_input_ = c1;
      }

      /** \brief Computes the transformation aligning model to input
       * \param[in] input_ftt CRH histogram of the input cloud
       * \param[in] target_ftt CRH histogram of the target cloud
       */
      void
      align (pcl::PointCloud<pcl::Histogram<nbins_> > & input_ftt, pcl::PointCloud<pcl::Histogram<nbins_> > & target_ftt)
      {

        transforms_.clear(); //clear from last round...

        std::vector<float> peaks;
        computeRollAngle (input_ftt, target_ftt, peaks);

        //if the number of peaks is too big, we should try to reduce using siluette matching

        for (size_t i = 0; i < peaks.size(); i++)
        {
          Eigen::Affine3f rollToRot;
          computeRollTransform (centroid_input_, centroid_target_, peaks[i], rollToRot);

          Eigen::Matrix4f rollHomMatrix = Eigen::Matrix4f ();
          rollHomMatrix.setIdentity (4, 4);
          rollHomMatrix = rollToRot.matrix ();

          Eigen::Matrix4f translation2;
          translation2.setIdentity (4, 4);
          Eigen::Vector3f centr = rollToRot * centroid_target_;
          translation2 (0, 3) = centroid_input_[0] - centr[0];
          translation2 (1, 3) = centroid_input_[1] - centr[1];
          translation2 (2, 3) = centroid_input_[2] - centr[2];

          Eigen::Matrix4f resultHom (translation2 * rollHomMatrix);
          transforms_.push_back(resultHom.inverse());
        }

      }

      /** \brief Computes the roll angle that aligns input to modle.
       * \param[in] input_ftt CRH histogram of the input cloud
       * \param[in] target_ftt CRH histogram of the target cloud
       * \param[out] peaks Vector containing angles where the histograms correlate
       */
      void
      computeRollAngle (pcl::PointCloud<pcl::Histogram<nbins_> > & input_ftt, pcl::PointCloud<pcl::Histogram<nbins_> > & target_ftt,
                        std::vector<float> & peaks)
      {

        pcl::PointCloud<pcl::Histogram<nbins_> > input_ftt_negate (input_ftt);

        for (int i = 2; i < (nbins_); i += 2)
          input_ftt_negate.points[0].histogram[i] = -input_ftt_negate.points[0].histogram[i];

        int nr_bins_after_padding = 180;
        int peak_distance = 5;
        int cutoff = nbins_ - 1;

        kiss_fft_cpx * multAB = new kiss_fft_cpx[nr_bins_after_padding];
        for (int i = 0; i < nr_bins_after_padding; i++)
          multAB[i].r = multAB[i].i = 0.f;

        int k = 0;
        multAB[k].r = input_ftt_negate.points[0].histogram[0] * target_ftt.points[0].histogram[0];
        k++;

        float a, b, c, d;
        for (int i = 1; i < cutoff; i += 2, k++)
        {
          a = input_ftt_negate.points[0].histogram[i];
          b = input_ftt_negate.points[0].histogram[i + 1];
          c = target_ftt.points[0].histogram[i];
          d = target_ftt.points[0].histogram[i + 1];
          multAB[k].r = a * c - b * d;
          multAB[k].i = b * c + a * d;

          float tmp = sqrtf (multAB[k].r * multAB[k].r + multAB[k].i * multAB[k].i);

          multAB[k].r /= tmp;
          multAB[k].i /= tmp;
        }

        multAB[nbins_ - 1].r = input_ftt_negate.points[0].histogram[nbins_ - 1] * target_ftt.points[0].histogram[nbins_ - 1];

        kiss_fft_cfg mycfg = kiss_fft_alloc (nr_bins_after_padding, 1, NULL, NULL);
        kiss_fft_cpx * invAB = new kiss_fft_cpx[nr_bins_after_padding];
        kiss_fft (mycfg, multAB, invAB);

        std::vector < std::pair<float, int> > scored_peaks (nr_bins_after_padding);
        for (int i = 0; i < nr_bins_after_padding; i++)
          scored_peaks[i] = std::make_pair (invAB[i].r, i);

        std::sort (scored_peaks.begin (), scored_peaks.end (), peaks_ordering ());

        std::vector<int> peaks_indices;
        std::vector<float> peaks_values;

        // we look at the upper quantile_
        float quantile = quantile_;
        int max_inserted= max_peaks_;

        int inserted=0;
        bool stop=false;
        for (int i = 0; (i < static_cast<int> (quantile * static_cast<float> (nr_bins_after_padding))) && !stop; i++)
        {
          if (scored_peaks[i].first >= scored_peaks[0].first * accept_threshold_)
          {
            bool insert = true;

            for (size_t j = 0; j < peaks_indices.size (); j++)
            { //check inserted peaks, first pick always inserted
              if (std::abs (peaks_indices[j] - scored_peaks[i].second) <= peak_distance || std::abs (
                                                                                             peaks_indices[j] - (scored_peaks[i].second
                                                                                                 - nr_bins_after_padding)) <= peak_distance)
              {
                insert = false;
                break;
              }
            }

            if (insert)
            {
              peaks_indices.push_back (scored_peaks[i].second);
              peaks_values.push_back (scored_peaks[i].first);
              peaks.push_back (static_cast<float> (scored_peaks[i].second * (360 / nr_bins_after_padding)));
              inserted++;
              if(inserted >= max_inserted)
                stop = true;
            }
          }
        }
      }
    };
}

#endif /* CRH_ALIGNMENT_H_ */
###

# dense_quantized_multi_mod_template.h
namespace pcl
{

  struct DenseQuantizedSingleModTemplate
  {
    std::vector<unsigned char> features;

    void 
    serialize (std::ostream & stream) const
    {
      const size_t num_of_features = static_cast<size_t> (features.size ());
      write (stream, num_of_features);
      for (size_t feature_index = 0; feature_index < num_of_features; ++feature_index)
      {
        write (stream, features[feature_index]);
      }
    }

    void 
    deserialize (std::istream & stream)
    {
      features.clear ();

      size_t num_of_features;
      read (stream, num_of_features);
      features.resize (num_of_features);
      for (size_t feature_index = 0; feature_index < num_of_features; ++feature_index)
      {
        read (stream, features[feature_index]);
      }
    }
  };

  struct DenseQuantizedMultiModTemplate
  {
    std::vector<DenseQuantizedSingleModTemplate> modalities;
    float response_factor;

    RegionXY region;

    void 
    serialize (std::ostream & stream) const
    {
      const size_t num_of_modalities = static_cast<size_t> (modalities.size ());
      write (stream, num_of_modalities);
      for (size_t modality_index = 0; modality_index < num_of_modalities; ++modality_index)
      {
        modalities[modality_index].serialize (stream);
      }

      region.serialize (stream);
    }

    void 
    deserialize (std::istream & stream)
    {
      modalities.clear ();

      size_t num_of_modalities;
      read (stream, num_of_modalities);
      modalities.resize (num_of_modalities);
      for (size_t modality_index = 0; modality_index < num_of_modalities; ++modality_index)
      {
        modalities[modality_index].deserialize (stream);
      }

      region.deserialize (stream);
    }
  };

}

#endif 
###

# distance_map.h
namespace pcl
{

  /** \brief Represents a distance map obtained from a distance transformation. 
    * \author Stefan Holzer
    */
  class DistanceMap
  {
    public:
      /** \brief Constructor. */
      DistanceMap () : data_ (0), width_ (0), height_ (0) {}
      /** \brief Destructor. */
      virtual ~DistanceMap () {}

      /** \brief Returns the width of the map. */
      inline size_t 
      getWidth () const
      {
        return (width_); 
      }

      /** \brief Returns the height of the map. */
      inline size_t 
      getHeight () const
      { 
        return (height_); 
      }
    
      /** \brief Returns a pointer to the beginning of map. */
      inline float * 
      getData () 
      { 
        return (&data_[0]); 
      }

      /** \brief Resizes the map to the specified size.
        * \param[in] width the new width of the map.
        * \param[in] height the new height of the map.
        */
      void 
      resize (const size_t width, const size_t height)
      {
        data_.resize (width*height);
        width_ = width;
        height_ = height;
      }

      /** \brief Operator to access an element of the map.
        * \param[in] col_index the column index of the element to access.
        * \param[in] row_index the row index of the element to access.
        */
      inline float & 
      operator() (const size_t col_index, const size_t row_index)
      {
        return (data_[row_index*width_ + col_index]);
      }

      /** \brief Operator to access an element of the map.
        * \param[in] col_index the column index of the element to access.
        * \param[in] row_index the row index of the element to access.
        */
      inline const float & 
      operator() (const size_t col_index, const size_t row_index) const
      {
        return (data_[row_index*width_ + col_index]);
      }

    protected:
      /** \brief The storage for the distance map data. */
      std::vector<float> data_;
      /** \brief The width of the map. */
      size_t width_;
      /** \brief The height of the map. */
      size_t height_;
  };

}


#endif 

###

# dotmod.h
namespace pcl
{
  class PCL_EXPORTS DOTModality
  {
    public:

      virtual ~DOTModality () {};

      //virtual QuantizedMap &
      //getDominantQuantizedMap () = 0;

      virtual QuantizedMap &
      getDominantQuantizedMap () = 0;

      virtual QuantizedMap
      computeInvariantQuantizedMap (const MaskMap & mask,
                                    const RegionXY & region) = 0;

  };
}

#endif    // PCL_FEATURES_DOT_MODALITY

###

# dot_modality.h
namespace pcl
{

  struct DOTMODDetection
  {
    size_t bin_x;
    size_t bin_y;
    size_t template_id;
    float score;
  };

  /**
    * \brief Template matching using the DOTMOD approach.
    * \author Stefan Holzer, Stefan Hinterstoisser
    */
  class PCL_EXPORTS DOTMOD
  {
    public:
      /** \brief Constructor */
      DOTMOD (size_t template_width,
              size_t template_height);

      /** \brief Destructor */
      virtual ~DOTMOD ();

      /** \brief Creates a template from the specified data and adds it to the matching queue. 
        * \param modalities
        * \param masks
        * \param template_anker_x
        * \param template_anker_y
        * \param region
        */
      size_t 
      createAndAddTemplate (const std::vector<DOTModality*> & modalities,
                            const std::vector<MaskMap*> & masks,
                            size_t template_anker_x,
                            size_t template_anker_y,
                            const RegionXY & region);

      void
      detectTemplates (const std::vector<DOTModality*> & modalities,
                       float template_response_threshold,
                       std::vector<DOTMODDetection> & detections,
                       const size_t bin_size) const;

      inline const DenseQuantizedMultiModTemplate &
      getTemplate (size_t template_id) const
      { 
        return (templates_[template_id]);
      }

      inline size_t
      getNumOfTemplates ()
      {
        return (templates_.size ());
      }

      void
      saveTemplates (const char * file_name) const;

      void
      loadTemplates (const char * file_name);

      void 
      serialize (std::ostream & stream) const;

      void 
      deserialize (std::istream & stream);


    private:
      /** template width */
      size_t template_width_;
      /** template height */
      size_t template_height_;
      /** template storage */
      std::vector<DenseQuantizedMultiModTemplate> templates_;
  };

}

#endif 

###

# hypothesis.h
# ransac_based
namespace pcl
{
  namespace recognition
  {
    class HypothesisBase
    {
      public:
        HypothesisBase (const ModelLibrary::Model* obj_model)
        : obj_model_ (obj_model)
        {}

        HypothesisBase (const ModelLibrary::Model* obj_model, const float* rigid_transform)
        : obj_model_ (obj_model)
        {
          memcpy (rigid_transform_, rigid_transform, 12*sizeof (float));
        }

        virtual  ~HypothesisBase (){}

        void
        setModel (const ModelLibrary::Model* model)
        {
          obj_model_ = model;
        }

      public:
        float rigid_transform_[12];
        const ModelLibrary::Model* obj_model_;
    };

    class Hypothesis: public HypothesisBase
    {
      public:
        Hypothesis (const ModelLibrary::Model* obj_model = NULL)
         : HypothesisBase (obj_model),
           match_confidence_ (-1.0f),
           linear_id_ (-1)
        {
        }

        Hypothesis (const Hypothesis& src)
        : HypothesisBase (src.obj_model_, src.rigid_transform_),
          match_confidence_  (src.match_confidence_),
          explained_pixels_ (src.explained_pixels_)
        {
        }

        virtual ~Hypothesis (){}

        const Hypothesis&
        operator =(const Hypothesis& src)
        {
          memcpy (this->rigid_transform_, src.rigid_transform_, 12*sizeof (float));
          this->obj_model_  = src.obj_model_;
          this->match_confidence_  = src.match_confidence_;
          this->explained_pixels_ = src.explained_pixels_;

          return *this;
        }

        void
        setLinearId (int id)
        {
          linear_id_ = id;
        }

        int
        getLinearId () const
        {
          return (linear_id_);
        }

        void
        computeBounds (float bounds[6]) const
        {
          const float *b = obj_model_->getBoundsOfOctreePoints ();
          float p[3];

          // Initialize 'bounds'
          aux::transform (rigid_transform_, b[0], b[2], b[4], p);
          bounds[0] = bounds[1] = p[0];
          bounds[2] = bounds[3] = p[1];
          bounds[4] = bounds[5] = p[2];

          // Expand 'bounds' to contain the other 7 points of the octree bounding box
          aux::transform (rigid_transform_, b[0], b[2], b[5], p); aux::expandBoundingBoxToContainPoint (bounds, p);
          aux::transform (rigid_transform_, b[0], b[3], b[4], p); aux::expandBoundingBoxToContainPoint (bounds, p);
          aux::transform (rigid_transform_, b[0], b[3], b[5], p); aux::expandBoundingBoxToContainPoint (bounds, p);
          aux::transform (rigid_transform_, b[1], b[2], b[4], p); aux::expandBoundingBoxToContainPoint (bounds, p);
          aux::transform (rigid_transform_, b[1], b[2], b[5], p); aux::expandBoundingBoxToContainPoint (bounds, p);
          aux::transform (rigid_transform_, b[1], b[3], b[4], p); aux::expandBoundingBoxToContainPoint (bounds, p);
          aux::transform (rigid_transform_, b[1], b[3], b[5], p); aux::expandBoundingBoxToContainPoint (bounds, p);
        }

        void
        computeCenterOfMass (float center_of_mass[3]) const
        {
          aux::transform (rigid_transform_, obj_model_->getOctreeCenterOfMass (), center_of_mass);
        }

      public:
        float match_confidence_;
        std::set<int> explained_pixels_;
        int linear_id_;
    };
  } // namespace recognition
} // namespace pcl

#endif /* PCL_RECOGNITION_HYPOTHESIS_H_ */
###

# implicit_shape_model.h
namespace pcl
{
  /** \brief This struct is used for storing peak. */
  struct ISMPeak
  {
    /** \brief Point were this peak is located. */
    PCL_ADD_POINT4D;

    /** \brief Density of this peak. */
    double density;

    /** \brief Determines which class this peak belongs. */
    int class_id;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  } EIGEN_ALIGN16;

  namespace features
  {
    /** \brief This class is used for storing, analyzing and manipulating votes
      * obtained from ISM algorithm. */
    template <typename PointT>
    class PCL_EXPORTS ISMVoteList
    {
      public:

        /** \brief Empty constructor with member variables initialization. */
        ISMVoteList ();

        /** \brief virtual descriptor. */
        virtual
        ~ISMVoteList ();

        /** \brief This method simply adds another vote to the list.
          * \param[in] in_vote vote to add
          * \param[in] vote_origin origin of the added vote
          * \param[in] in_class class for which this vote is cast
          */
        void
        addVote (pcl::InterestPoint& in_vote, const PointT &vote_origin, int in_class);

        /** \brief Returns the colored cloud that consists of votes for center (blue points) and
          * initial point cloud (if it was passed).
          * \param[in] cloud cloud that needs to be merged with votes for visualizing. */
        typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        getColoredCloud (typename pcl::PointCloud<PointT>::Ptr cloud = 0);

        /** \brief This method finds the strongest peaks (points were density has most higher values).
          * It is based on the non maxima supression principles.
          * \param[out] out_peaks it will contain the strongest peaks
          * \param[in] in_class_id class of interest for which peaks are evaluated
          * \param[in] in_non_maxima_radius non maxima supression radius. The shapes radius is recommended for this value.
          * \param in_sigma
          */
        void
        findStrongestPeaks (std::vector<ISMPeak, Eigen::aligned_allocator<ISMPeak> > &out_peaks, int in_class_id, double in_non_maxima_radius, double in_sigma);

        /** \brief Returns the density at the specified point.
          * \param[in] point point of interest
          * \param[in] sigma_dist
          */
        double
        getDensityAtPoint (const PointT &point, double sigma_dist);

        /** \brief This method simply returns the number of votes. */
        unsigned int
        getNumberOfVotes ();

      protected:

        /** \brief this method is simply setting up the search tree. */
        void
        validateTree ();

        Eigen::Vector3f
        shiftMean (const Eigen::Vector3f& snapPt, const double in_dSigmaDist);

      protected:

        /** \brief Stores all votes. */
        pcl::PointCloud<pcl::InterestPoint>::Ptr votes_;

        /** \brief Signalizes if the tree is valid. */
        bool tree_is_valid_;

        /** \brief Stores the origins of the votes. */
        typename pcl::PointCloud<PointT>::Ptr votes_origins_;

        /** \brief Stores classes for which every single vote was cast. */
        std::vector<int> votes_class_;

        /** \brief Stores the search tree. */
        pcl::KdTreeFLANN<pcl::InterestPoint>::Ptr tree_;

        /** \brief Stores neighbours indices. */
        std::vector<int> k_ind_;

        /** \brief Stores square distances to the corresponding neighbours. */
        std::vector<float> k_sqr_dist_;
    };
 
    /** \brief The assignment of this structure is to store the statistical/learned weights and other information
      * of the trained Implict Shape Model algorithm.
      */
    struct PCL_EXPORTS ISMModel
    {
      /** \brief Simple constructor that initializes the structure. */
      ISMModel ();

      /** \brief Copy constructor for deep copy. */
      ISMModel (ISMModel const & copy);

      /** Destructor that frees memory. */
      virtual
      ~ISMModel ();

      /** \brief This method simply saves the trained model for later usage.
        * \param[in] file_name path to file for saving model
        */
      bool
      saveModelToFile (std::string& file_name);

      /** \brief This method loads the trained model from file.
        * \param[in] file_name path to file which stores trained model
        */
      bool
      loadModelFromfile (std::string& file_name);

      /** \brief this method resets all variables and frees memory. */
      void
      reset ();

      /** Operator overloading for deep copy. */
      ISMModel & operator = (const ISMModel& other);

      /** \brief Stores statistical weights. */
      std::vector<std::vector<float> > statistical_weights_;

      /** \brief Stores learned weights. */
      std::vector<float> learned_weights_;

      /** \brief Stores the class label for every direction. */
      std::vector<unsigned int> classes_;

      /** \brief Stores the sigma value for each class. This values were used to compute the learned weights. */
      std::vector<float> sigmas_;

      /** \brief Stores the directions to objects center for each visual word. */
      Eigen::MatrixXf directions_to_center_;

      /** \brief Stores the centers of the clusters that were obtained during the visual words clusterization. */
      Eigen::MatrixXf clusters_centers_;

      /** \brief This is an array of clusters. Each cluster stores the indices of the visual words that it contains. */
      std::vector<std::vector<unsigned int> > clusters_;

      /** \brief Stores the number of classes. */
      unsigned int number_of_classes_;

      /** \brief Stores the number of visual words. */
      unsigned int number_of_visual_words_;

      /** \brief Stores the number of clusters. */
      unsigned int number_of_clusters_;

      /** \brief Stores descriptors dimension. */
      unsigned int descriptors_dimension_;

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
  }

  namespace ism
  {
    /** \brief This class implements Implicit Shape Model algorithm described in
      * "Hough Transforms and 3D SURF for robust three dimensional classication"
      * by Jan Knopp1, Mukta Prasad, Geert Willems1, Radu Timofte, and Luc Van Gool.
      * It has two main member functions. One for training, using the data for which we know
      * which class it belongs to. And second for investigating a cloud for the presence
      * of the class of interest.
      * Implementation of the ISM algorithm described in "Hough Transforms and 3D SURF for robust three dimensional classication"
      * by Jan Knopp, Mukta Prasad, Geert Willems, Radu Timofte, and Luc Van Gool
      *
      * Authors: Roman Shapovalov, Alexander Velizhev, Sergey Ushakov
      */
    template <int FeatureSize, typename PointT, typename NormalT = pcl::Normal>
    class PCL_EXPORTS ImplicitShapeModelEstimation
    {
      public:

        typedef boost::shared_ptr<pcl::features::ISMModel> ISMModelPtr;

      protected:

        /** \brief This structure stores the information about the keypoint. */
        struct PCL_EXPORTS LocationInfo
        {
          /** \brief Location info constructor.
            * \param[in] model_num number of training model.
            * \param[in] dir_to_center expected direction to center
            * \param[in] origin initial point
            * \param[in] normal normal of the initial point
            */
          LocationInfo (unsigned int model_num, const PointT& dir_to_center, const PointT& origin, const NormalT& normal) :
            model_num_ (model_num),
            dir_to_center_ (dir_to_center),
            point_ (origin),
            normal_ (normal) {};

          /** \brief Tells from which training model this keypoint was extracted. */
          unsigned int model_num_;

          /** \brief Expected direction to center for this keypoint. */
          PointT dir_to_center_;

          /** \brief Stores the initial point. */
          PointT point_;

          /** \brief Stores the normal of the initial point. */
          NormalT normal_;
        };

        /** \brief This structure is used for determining the end of the
          * k-means clustering process. */
        typedef struct PCL_EXPORTS TC
        {
          enum
          {
            COUNT = 1,
            EPS = 2
          };

          /** \brief Termination criteria constructor.
            * \param[in] type defines the condition of termination(max iter., desired accuracy)
            * \param[in] max_count defines the max number of iterations
            * \param[in] epsilon defines the desired accuracy
            */
          TC(int type, int max_count, float epsilon) :
            type_ (type),
            max_count_ (max_count),
            epsilon_ (epsilon) {};

          /** \brief Flag that determines when the k-means clustering must be stopped.
            * If type_ equals COUNT then it must be stopped when the max number of iterations will be
            * reached. If type_ eaquals EPS then it must be stopped when the desired accuracy will be reached.
            * These flags can be used together, in that case the clustering will be finished when one of these
            * conditions will be reached.
            */
          int type_;

          /** \brief Defines maximum number of iterations for k-means clustering. */
          int max_count_;

          /** \brief Defines the accuracy for k-means clustering. */
          float epsilon_;
        } TermCriteria;

        /** \brief Structure for storing the visual word. */
        struct PCL_EXPORTS VisualWordStat
        {
          /** \brief Empty constructor with member variables initialization. */
          VisualWordStat () :
            class_ (-1),
            learned_weight_ (0.0f),
            dir_to_center_ (0.0f, 0.0f, 0.0f) {};

          /** \brief Which class this vote belongs. */
          int class_;

          /** \brief Weight of the vote. */
          float learned_weight_;

          /** \brief Expected direction to center. */
          pcl::PointXYZ dir_to_center_;
        };

      public:

        /** \brief Simple constructor that initializes everything. */
        ImplicitShapeModelEstimation ();

        /** \brief Simple destructor. */
        virtual
        ~ImplicitShapeModelEstimation ();

        /** \brief This method simply returns the clouds that were set as the training clouds. */
        std::vector<typename pcl::PointCloud<PointT>::Ptr>
        getTrainingClouds ();

        /** \brief Allows to set clouds for training the ISM model.
          * \param[in] training_clouds array of point clouds for training
          */
        void
        setTrainingClouds (const std::vector< typename pcl::PointCloud<PointT>::Ptr >& training_clouds);

        /** \brief Returns the array of classes that indicates which class the corresponding training cloud belongs. */
        std::vector<unsigned int>
        getTrainingClasses ();

        /** \brief Allows to set the class labels for the corresponding training clouds.
          * \param[in] training_classes array of class labels
          */
        void
        setTrainingClasses (const std::vector<unsigned int>& training_classes);

        /** \brief This method returns the coresponding cloud of normals for every training point cloud. */
        std::vector<typename pcl::PointCloud<NormalT>::Ptr>
        getTrainingNormals ();

        /** \brief Allows to set normals for the training clouds that were passed through setTrainingClouds method.
          * \param[in] training_normals array of clouds, each cloud is the cloud of normals
          */
        void
        setTrainingNormals (const std::vector< typename pcl::PointCloud<NormalT>::Ptr >& training_normals);

        /** \brief Returns the sampling size used for cloud simplification. */
        float
        getSamplingSize ();

        /** \brief Changes the sampling size used for cloud simplification.
          * \param[in] sampling_size desired size of grid bin
          */
        void
        setSamplingSize (float sampling_size);

        /** \brief Returns the current feature estimator used for extraction of the descriptors. */
        boost::shared_ptr<pcl::Feature<PointT, pcl::Histogram<FeatureSize> > >
        getFeatureEstimator ();

        /** \brief Changes the feature estimator.
          * \param[in] feature feature estimator that will be used to extract the descriptors.
          * Note that it must be fully initialized and configured.
          */
        void
        setFeatureEstimator (boost::shared_ptr<pcl::Feature<PointT, pcl::Histogram<FeatureSize> > > feature);

        /** \brief Returns the number of clusters used for descriptor clustering. */
        unsigned int
        getNumberOfClusters ();

        /** \brief Changes the number of clusters.
          * \param num_of_clusters desired number of clusters
          */
        void
        setNumberOfClusters (unsigned int num_of_clusters);

        /** \brief Returns the array of sigma values. */
        std::vector<float>
        getSigmaDists ();

        /** \brief This method allows to set the value of sigma used for calculating the learned weights for every single class.
          * \param[in] training_sigmas new sigmas for every class. If you want these values to be computed automatically,
          * just pass the empty array. The automatic regime calculates the maximum distance between the objects points and takes 10% of
          * this value as recomended in the article. If there are several objects of the same class,
          * then it computes the average maximum distance and takes 10%. Note that each class has its own sigma value.
          */
        void
        setSigmaDists (const std::vector<float>& training_sigmas);

        /** \brief Returns the state of Nvot coeff from [Knopp et al., 2010, (4)],
          * if set to false then coeff is taken as 1.0. It is just a kind of heuristic.
          * The default behavior is as in the article. So you can ignore this if you want.
          */
        bool
        getNVotState ();

        /** \brief Changes the state of the Nvot coeff from [Knopp et al., 2010, (4)].
          * \param[in] state desired state, if false then Nvot is taken as 1.0
          */
        void
        setNVotState (bool state);

        /** \brief This method performs training and forms a visual vocabulary. It returns a trained model that
          * can be saved to file for later usage.
          * \param[out] trained_model trained model
          */
        bool
        trainISM (ISMModelPtr& trained_model);

        /** \brief This function is searching for the class of interest in a given cloud
          * and returns the list of votes.
          * \param[in] model trained model which will be used for searching the objects
          * \param[in] in_cloud input cloud that need to be investigated
          * \param[in] in_normals cloud of normals coresponding to the input cloud
          * \param[in] in_class_of_interest class which we are looking for
          */
        boost::shared_ptr<pcl::features::ISMVoteList<PointT> >
        findObjects (ISMModelPtr model, typename pcl::PointCloud<PointT>::Ptr in_cloud, typename pcl::PointCloud<Normal>::Ptr in_normals, int in_class_of_interest);

      protected:

        /** \brief Extracts the descriptors from the input clouds.
          * \param[out] histograms it will store the descriptors for each key point
          * \param[out] locations it will contain the comprehensive information (such as direction, initial keypoint)
          * for the corresponding descriptors
          */
        bool
        extractDescriptors (std::vector<pcl::Histogram<FeatureSize> >& histograms,
                            std::vector<LocationInfo, Eigen::aligned_allocator<LocationInfo> >& locations);

        /** \brief This method performs descriptor clustering.
          * \param[in] histograms descriptors to cluster
          * \param[out] labels it contains labels for each descriptor
          * \param[out] clusters_centers stores the centers of clusters
          */
        bool
        clusterDescriptors (std::vector< pcl::Histogram<FeatureSize> >& histograms, Eigen::MatrixXi& labels, Eigen::MatrixXf& clusters_centers);

        /** \brief This method calculates the value of sigma used for calculating the learned weights for every single class.
          * \param[out] sigmas computed sigmas.
          */
        void
        calculateSigmas (std::vector<float>& sigmas);

        /** \brief This function forms a visual vocabulary and evaluates weights
          * described in [Knopp et al., 2010, (5)].
          * \param[in] locations array containing description of each keypoint: its position, which cloud belongs
          * and expected direction to center
          * \param[in] labels labels that were obtained during k-means clustering
          * \param[in] sigmas array of sigmas for each class
          * \param[in] clusters clusters that were obtained during k-means clustering
          * \param[out] statistical_weights stores the computed statistical weights
          * \param[out] learned_weights stores the computed learned weights
          */
        void
        calculateWeights (const std::vector< LocationInfo, Eigen::aligned_allocator<LocationInfo> >& locations,
                          const Eigen::MatrixXi &labels,
                          std::vector<float>& sigmas,
                          std::vector<std::vector<unsigned int> >& clusters,
                          std::vector<std::vector<float> >& statistical_weights,
                          std::vector<float>& learned_weights);

        /** \brief Simplifies the cloud using voxel grid principles.
          * \param[in] in_point_cloud cloud that need to be simplified
          * \param[in] in_normal_cloud normals of the cloud that need to be simplified
          * \param[out] out_sampled_point_cloud simplified cloud
          * \param[out] out_sampled_normal_cloud and the corresponding normals
          */
        void
        simplifyCloud (typename pcl::PointCloud<PointT>::ConstPtr in_point_cloud,
                       typename pcl::PointCloud<NormalT>::ConstPtr in_normal_cloud,
                       typename pcl::PointCloud<PointT>::Ptr out_sampled_point_cloud,
                       typename pcl::PointCloud<NormalT>::Ptr out_sampled_normal_cloud);

        /** \brief This method simply shifts the clouds points relative to the passed point.
          * \param[in] in_cloud cloud to shift
          * \param[in] shift_point point relative to which the cloud will be shifted
          */
        void
        shiftCloud (typename pcl::PointCloud<PointT>::Ptr in_cloud, Eigen::Vector3f shift_point);

        /** \brief This method simply computes the rotation matrix, so that the given normal
          * would match the Y axis after the transformation. This is done because the algorithm needs to be invariant
          * to the affine transformations.
          * \param[in] in_normal normal for which the rotation matrix need to be computed
          */
        Eigen::Matrix3f
        alignYCoordWithNormal (const NormalT& in_normal);

        /** \brief This method applies transform set in in_transform to vector io_vector.
          * \param[in] io_vec vector that need to be transformed
          * \param[in] in_transform matrix that contains the transformation
          */
        void
        applyTransform (Eigen::Vector3f& io_vec, const Eigen::Matrix3f& in_transform);

        /** \brief This method estimates features for the given point cloud.
          * \param[in] sampled_point_cloud sampled point cloud for which the features must be computed
          * \param[in] normal_cloud normals for the original point cloud
          * \param[out] feature_cloud it will store the computed histograms (features) for the given cloud
          */
        void
        estimateFeatures (typename pcl::PointCloud<PointT>::Ptr sampled_point_cloud,
                          typename pcl::PointCloud<NormalT>::Ptr normal_cloud,
                          typename pcl::PointCloud<pcl::Histogram<FeatureSize> >::Ptr feature_cloud);

        /** \brief Performs K-means clustering.
          * \param[in] points_to_cluster points to cluster
          * \param[in] number_of_clusters desired number of clusters
          * \param[out] io_labels output parameter, which stores the label for each point
          * \param[in] criteria defines when the computational process need to be finished. For example if the
          * desired accuracy is achieved or the iteration number exceeds given value
          * \param[in] attempts number of attempts to compute clustering
          * \param[in] flags if set to USE_INITIAL_LABELS then initial approximation of labels is taken from io_labels
          * \param[out] cluster_centers it will store the cluster centers
          */
        double
        computeKMeansClustering (const Eigen::MatrixXf& points_to_cluster,
                                 int number_of_clusters,
                                 Eigen::MatrixXi& io_labels,
                                 TermCriteria criteria,
                                 int attempts,
                                 int flags,
                                 Eigen::MatrixXf& cluster_centers);

        /** \brief Generates centers for clusters as described in 
          * Arthur, David and Sergei Vassilvitski (2007) k-means++: The Advantages of Careful Seeding.
          * \param[in] data points to cluster
          * \param[out] out_centers it will contain generated centers
          * \param[in] number_of_clusters defines the number of desired cluster centers
          * \param[in] trials number of trials to generate a center
          */
        void
        generateCentersPP (const Eigen::MatrixXf& data,
                           Eigen::MatrixXf& out_centers,
                           int number_of_clusters,
                           int trials);

        /** \brief Generates random center for cluster.
          * \param[in] boxes contains min and max values for each dimension
          * \param[out] center it will the contain generated center
          */
        void
        generateRandomCenter (const std::vector<Eigen::Vector2f>& boxes, Eigen::VectorXf& center);

        /** \brief Computes the square distance beetween two vectors.
          * \param[in] vec_1 first vector
          * \param[in] vec_2 second vector
          */
        float
        computeDistance (Eigen::VectorXf& vec_1, Eigen::VectorXf& vec_2);

        /** \brief Forbids the assignment operator. */
        ImplicitShapeModelEstimation&
        operator= (const ImplicitShapeModelEstimation&);

      protected:

        /** \brief Stores the clouds used for training. */
        std::vector<typename pcl::PointCloud<PointT>::Ptr> training_clouds_;

        /** \brief Stores the class number for each cloud from training_clouds_. */
        std::vector<unsigned int> training_classes_;

        /** \brief Stores the normals for each training cloud. */
        std::vector<typename pcl::PointCloud<NormalT>::Ptr> training_normals_;

        /** \brief This array stores the sigma values for each training class. If this array has a size equals 0, then
          * sigma values will be calculated automatically.
          */
        std::vector<float> training_sigmas_;

        /** \brief This value is used for the simplification. It sets the size of grid bin. */
        float sampling_size_;

        /** \brief Stores the feature estimator. */
        boost::shared_ptr<pcl::Feature<PointT, pcl::Histogram<FeatureSize> > > feature_estimator_;

        /** \brief Number of clusters, is used for clustering descriptors during the training. */
        unsigned int number_of_clusters_;

        /** \brief If set to false then Nvot coeff from [Knopp et al., 2010, (4)] is equal 1.0. */
        bool n_vot_ON_;

        /** \brief This const value is used for indicating that for k-means clustering centers must
          * be generated as described in
          * Arthur, David and Sergei Vassilvitski (2007) k-means++: The Advantages of Careful Seeding. */
        static const int PP_CENTERS = 2;

        /** \brief This const value is used for indicating that input labels must be taken as the
          * initial approximation for k-means clustering. */
        static const int USE_INITIAL_LABELS = 1;
    };
  }
}

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::ISMPeak,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, density, ism_density)
  (float, class_id, ism_class_id)
)

#endif  //#ifndef PCL_IMPLICIT_SHAPE_MODEL_H_
###

# linemod.h
namespace pcl
{

  /** \brief Stores a set of energy maps.
    * \author Stefan Holzer
    */
  class PCL_EXPORTS EnergyMaps
  {
    public:
      /** \brief Constructor. */
      EnergyMaps () : width_ (0), height_ (0), nr_bins_ (0), maps_ () 
      {
      }

      /** \brief Destructor. */
      virtual ~EnergyMaps () 
      {
      }

      /** \brief Returns the width of the energy maps. */
      inline size_t 
      getWidth () const 
      { 
        return (width_); 
      }
      
      /** \brief Returns the height of the energy maps. */
      inline size_t 
      getHeight () const 
      { 
        return (height_); 
      }
      
      /** \brief Returns the number of bins used for quantization (which is equal to the number of energy maps). */
      inline size_t 
      getNumOfBins () const
      { 
        return (nr_bins_);
      }

      /** \brief Initializes the set of energy maps.
        * \param[in] width the width of the energy maps.
        * \param[in] height the height of the energy maps.
        * \param[in] nr_bins the number of bins used for quantization.
        */
      void 
      initialize (const size_t width, const size_t height, const size_t nr_bins)
      {
        maps_.resize(nr_bins, NULL);
        width_ = width;
        height_ = height;
        nr_bins_ = nr_bins;

        const size_t mapsSize = width*height;

        for (size_t map_index = 0; map_index < maps_.size (); ++map_index)
        {
          //maps_[map_index] = new unsigned char[mapsSize];
          maps_[map_index] = reinterpret_cast<unsigned char*> (aligned_malloc (mapsSize));
          memset (maps_[map_index], 0, mapsSize);
        }
      }

      /** \brief Releases the internal data. */
      void 
      releaseAll ()
      {
        for (size_t map_index = 0; map_index < maps_.size (); ++map_index)
          //if (maps_[map_index] != NULL) delete[] maps_[map_index];
          if (maps_[map_index] != NULL) aligned_free (maps_[map_index]);

        maps_.clear ();
        width_ = 0;
        height_ = 0;
        nr_bins_ = 0;
      }

      /** \brief Operator for accessing a specific element in the set of energy maps.
        * \param[in] bin_index the quantization bin (states which of the energy maps to access).
        * \param[in] col_index the column index within the specified energy map.
        * \param[in] row_index the row index within the specified energy map.
        */
      inline unsigned char & 
      operator() (const size_t bin_index, const size_t col_index, const size_t row_index)
      {
        return (maps_[bin_index][row_index*width_ + col_index]);
      }

      /** \brief Operator for accessing a specific element in the set of energy maps.
        * \param[in] bin_index the quantization bin (states which of the energy maps to access).
        * \param[in] index the element index within the specified energy map.
        */
      inline unsigned char & 
      operator() (const size_t bin_index, const size_t index)
      {
        return (maps_[bin_index][index]);
      }

      /** \brief Returns a pointer to the data of the specified energy map.
        * \param[in] bin_index the index of the energy map to return (== the quantization bin).
        */
      inline unsigned char * 
      operator() (const size_t bin_index)
      {
        return (maps_[bin_index]);
      }

      /** \brief Operator for accessing a specific element in the set of energy maps.
        * \param[in] bin_index the quantization bin (states which of the energy maps to access).
        * \param[in] col_index the column index within the specified energy map.
        * \param[in] row_index the row index within the specified energy map.
        */
      inline const unsigned char & 
      operator() (const size_t bin_index, const size_t col_index, const size_t row_index) const
      {
        return (maps_[bin_index][row_index*width_ + col_index]);
      }

      /** \brief Operator for accessing a specific element in the set of energy maps.
        * \param[in] bin_index the quantization bin (states which of the energy maps to access).
        * \param[in] index the element index within the specified energy map.
        */
      inline const unsigned char & 
      operator() (const size_t bin_index, const size_t index) const
      {
        return (maps_[bin_index][index]);
      }

      /** \brief Returns a pointer to the data of the specified energy map.
        * \param[in] bin_index the index of the energy map to return (== the quantization bin).
        */
      inline const unsigned char * 
      operator() (const size_t bin_index) const
      {
        return (maps_[bin_index]);
      }

    private:
      /** \brief The width of the energy maps. */
      size_t width_;
      /** \brief The height of the energy maps. */
      size_t height_;
      /** \brief The number of quantization bins (== the number of internally stored energy maps). */
      size_t nr_bins_;
      /** \brief Storage for the energy maps. */
      std::vector<unsigned char*> maps_;
  };

  /** \brief Stores a set of linearized maps.
    * \author Stefan Holzer
    */
  class PCL_EXPORTS LinearizedMaps
  {
    public:
      /** \brief Constructor. */
      LinearizedMaps () : width_ (0), height_ (0), mem_width_ (0), mem_height_ (0), step_size_ (0), maps_ ()
      {
      }
      
      /** \brief Destructor. */
      virtual ~LinearizedMaps () 
      {
      }

      /** \brief Returns the width of the linearized map. */
      inline size_t 
      getWidth () const { return (width_); }
      
      /** \brief Returns the height of the linearized map. */
      inline size_t 
      getHeight () const { return (height_); }
      
      /** \brief Returns the step-size used to construct the linearized map. */
      inline size_t 
      getStepSize () const { return (step_size_); }
      
      /** \brief Returns the size of the memory map. */
      inline size_t 
      getMapMemorySize () const { return (mem_width_ * mem_height_); }

      /** \brief Initializes the linearized map.
        * \param[in] width the width of the source map.
        * \param[in] height the height of the source map.
        * \param[in] step_size the step-size used to sample the source map.
        */
      void 
      initialize (const size_t width, const size_t height, const size_t step_size)
      {
        maps_.resize(step_size*step_size, NULL);
        width_ = width;
        height_ = height;
        mem_width_ = width / step_size;
        mem_height_ = height / step_size;
        step_size_ = step_size;

        const size_t mapsSize = mem_width_ * mem_height_;

        for (size_t map_index = 0; map_index < maps_.size (); ++map_index)
        {
          //maps_[map_index] = new unsigned char[2*mapsSize];
          maps_[map_index] = reinterpret_cast<unsigned char*> (aligned_malloc (2*mapsSize));
          memset (maps_[map_index], 0, 2*mapsSize);
        }
      }

      /** \brief Releases the internal memory. */
      void 
      releaseAll ()
      {
        for (size_t map_index = 0; map_index < maps_.size (); ++map_index)
          //if (maps_[map_index] != NULL) delete[] maps_[map_index];
          if (maps_[map_index] != NULL) aligned_free (maps_[map_index]);

        maps_.clear ();
        width_ = 0;
        height_ = 0;
        mem_width_ = 0;
        mem_height_ = 0;
        step_size_ = 0;
      }

      /** \brief Operator to access elements of the linearized map by column and row index.
        * \param[in] col_index the column index.
        * \param[in] row_index the row index.
        */
      inline unsigned char * 
      operator() (const size_t col_index, const size_t row_index)
      {
        return (maps_[row_index*step_size_ + col_index]);
      }

      /** \brief Returns a linearized map starting at the specified position.
        * \param[in] col_index the column index at which the returned map starts.
        * \param[in] row_index the row index at which the returned map starts.
        */
      inline unsigned char * 
      getOffsetMap (const size_t col_index, const size_t row_index)
      {
        const size_t map_col = col_index % step_size_;
        const size_t map_row = row_index % step_size_;

        const size_t map_mem_col_index = col_index / step_size_;
        const size_t map_mem_row_index = row_index / step_size_;

        return (maps_[map_row*step_size_ + map_col] + map_mem_row_index*mem_width_ + map_mem_col_index);
      }

    private:
      /** \brief the original width of the data represented by the map. */
      size_t width_;
      /** \brief the original height of the data represented by the map. */
      size_t height_;
      /** \brief the actual width of the linearized map. */
      size_t mem_width_;
      /** \brief the actual height of the linearized map. */
      size_t mem_height_;
      /** \brief the step-size used for sampling the original data. */
      size_t step_size_;
      /** \brief a vector containing all the linearized maps. */
      std::vector<unsigned char*> maps_;
  };

  /** \brief Represents a detection of a template using the LINEMOD approach.
    * \author Stefan Holzer
    */
  struct PCL_EXPORTS LINEMODDetection
  {
    /** \brief Constructor. */
    LINEMODDetection () : x (0), y (0), template_id (0), score (0.0f), scale (1.0f) {}

    /** \brief x-position of the detection. */
    int x;
    /** \brief y-position of the detection. */
    int y;
    /** \brief ID of the detected template. */
    int template_id;
    /** \brief score of the detection. */
    float score;
    /** \brief scale at which the template was detected. */
    float scale;
  };

  /**
    * \brief Template matching using the LINEMOD approach.
    * \author Stefan Holzer, Stefan Hinterstoisser
    */
  class PCL_EXPORTS LINEMOD
  {
    public:
      /** \brief Constructor */
      LINEMOD ();

      /** \brief Destructor */
      virtual ~LINEMOD ();

      /** \brief Creates a template from the specified data and adds it to the matching queue. 
        * \param[in] modalities the modalities used to create the template.
        * \param[in] masks the masks that determine which parts of the modalities are used for creating the template.
        * \param[in] region the region which will be associated with the template (can be larger than the actual modality-maps).
        */
      int 
      createAndAddTemplate (const std::vector<QuantizableModality*> & modalities,
                            const std::vector<MaskMap*> & masks,
                            const RegionXY & region);

      /** \brief Adds the specified template to the matching queue.
        * \param[in] linemod_template the template to add.
        */
      int
      addTemplate (const SparseQuantizedMultiModTemplate & linemod_template);

      /** \brief Detects the stored templates in the supplied modality data.
        * \param[in] modalities the modalities that will be used for detection.
        * \param[out] detections the destination for the detections.
        */
      void
      detectTemplates (const std::vector<QuantizableModality*> & modalities,
                       std::vector<LINEMODDetection> & detections) const;

      /** \brief Detects the stored templates in a semi scale invariant manner 
        *        by applying the detection to multiple scaled versions of the input data.
        * \param[in] modalities the modalities that will be used for detection.
        * \param[out] detections the destination for the detections.
        * \param[in] min_scale the minimum scale.
        * \param[in] max_scale the maximum scale.
        * \param[in] scale_multiplier the multiplier for getting from one scale to the next.
        */
      void
      detectTemplatesSemiScaleInvariant (const std::vector<QuantizableModality*> & modalities,
                                         std::vector<LINEMODDetection> & detections,
                                         float min_scale = 0.6944444f,
                                         float max_scale = 1.44f,
                                         float scale_multiplier = 1.2f) const;

      /** \brief Matches the stored templates to the supplied modality data.
        * \param[in] modalities the modalities that will be used for matching.
        * \param[out] matches the found matches.
        */
      void
      matchTemplates (const std::vector<QuantizableModality*> & modalities,
                      std::vector<LINEMODDetection> & matches) const;

      /** \brief Sets the detection threshold. 
        * \param[in] threshold the detection threshold.
        */
      inline void
      setDetectionThreshold (float threshold)
      {
        template_threshold_ = threshold;
      }

      /** \brief Enables/disables non-maximum suppression.
        * \param[in] use_non_max_suppression determines whether to use non-maximum suppression or not.
        */
      inline void
      setNonMaxSuppression (bool use_non_max_suppression)
      {
        use_non_max_suppression_ = use_non_max_suppression;
      }

      /** \brief Enables/disables averaging of close detections.
        * \param[in] average_detections determines whether to average close detections or not.
        */
      inline void
      setDetectionAveraging (bool average_detections)
      {
        average_detections_ = average_detections;
      }

      /** \brief Returns the template with the specified ID.
        * \param[in] template_id the ID of the template to return.
        */
      inline const SparseQuantizedMultiModTemplate &
      getTemplate (int template_id) const
      { 
        return (templates_[template_id]);
      }

      /** \brief Returns the number of stored/trained templates. */
      inline size_t
      getNumOfTemplates () const
      {
        return (templates_.size ());
      }

      /** \brief Saves the stored templates to the specified file.
        * \param[in] file_name the name of the file to save the templates to.
        */
      void
      saveTemplates (const char * file_name) const;

      /** \brief Loads templates from the specified file.
        * \param[in] file_name the name of the file to load the template from.
        */
      void
      loadTemplates (const char * file_name);

      /** \brief Loads templates from the specified files.
        * \param[in] file_names vector of files to load the templates from.
        */

      void
      loadTemplates (std::vector<std::string> & file_names);

      /** \brief Serializes the stored templates to the specified stream.
        * \param[in] stream the stream the templates will be written to.
        */
      void 
      serialize (std::ostream & stream) const;

      /** \brief Deserializes templates from the specified stream.
        * \param[in] stream the stream the templates will be read from.
        */
      void 
      deserialize (std::istream & stream);


    private:
      /** template response threshold */
      float template_threshold_;
      /** states whether non-max-suppression on detections is enabled or not */
      bool use_non_max_suppression_;
      /** states whether to return an averaged detection */
      bool average_detections_;
      /** template storage */
      std::vector<SparseQuantizedMultiModTemplate> templates_;
  };

}

#endif 

###

# line_rgbd.h
# namespace pcl
# struct BoundingBoxXYZ
    # /** \brief Constructor. */
    # BoundingBoxXYZ () : x (0.0f), y (0.0f), z (0.0f), width (0.0f), height (0.0f), depth (0.0f) {}
	# 
    # /** \brief X-coordinate of the upper left front point */
    # float x;
    # /** \brief Y-coordinate of the upper left front point */
    # float y;
    # /** \brief Z-coordinate of the upper left front point */
    # float z;
	# 
    # /** \brief Width of the bounding box */
    # float width;
    # /** \brief Height of the bounding box */
    # float height;
    # /** \brief Depth of the bounding box */
    # float depth;
	
	# /** \brief High-level class for template matching using the LINEMOD approach based on RGB and Depth data.
    # * \author Stefan Holzer
    # */
  	# template <typename PointXYZT, typename PointRGBT=PointXYZT>
  	# class PCL_EXPORTS LineRGBD
  	# {
    	# public:
		# 
      	# /** \brief A LineRGBD detection. */
      	# struct Detection
      		# /** \brief Constructor. */
        	# Detection () : template_id (0), object_id (0), detection_id (0), response (0.0f), bounding_box () {}
			# 
        	# /** \brief The ID of the template. */
        	# size_t template_id;
        	# /** \brief The ID of the object corresponding to the template. */
        	# size_t object_id;
        	# /** \brief The ID of this detection. This is only valid for the last call of the method detect (...). */
        	# size_t detection_id;
        	# /** \brief The response of this detection. Responses are between 0 and 1, where 1 is best. */
        	# float response;
        	# /** \brief The 3D bounding box of the detection. */
        	# BoundingBoxXYZ bounding_box;
        	# /** \brief The 2D template region of the detection. */
        	# RegionXY region;
		
		
      	# /** \brief Constructor */
      	# LineRGBD ()
        # : intersection_volume_threshold_ (1.0f)
        # , linemod_ ()
        # , color_gradient_mod_ ()
        # , surface_normal_mod_ ()
        # , cloud_xyz_ ()
        # , cloud_rgb_ ()
        # , template_point_clouds_ ()
        # , bounding_boxes_ ()
        # , object_ids_ ()
        # , detections_ ()
		# /** \brief Destructor */
      	# virtual ~LineRGBD ()
      	# /** \brief Loads templates from a LMT (LineMod Template) file. Overrides old templates.
        # 	* LineMod Template files are TAR files that store pairs of PCD datasets
        # 	* together with their LINEMOD signatures in \ref
        # 	* SparseQuantizedMultiModTemplate format.
        # 	* \param[in] file_name The name of the file that stores the templates.
        # 	* \param object_id
        # 	* \return true, if the operation was successful, false otherwise.
        # */
      	# bool loadTemplates (const std::string &file_name, size_t object_id = 0);
		# 
      	# bool addTemplate (const SparseQuantizedMultiModTemplate & sqmmt, pcl::PointCloud<pcl::PointXYZRGBA> & cloud, size_t object_id = 0);
		# 
      	# /** \brief Sets the threshold for the detection responses. Responses are between 0 and 1, where 1 is a best. 
        #  * \param[in] threshold The threshold used to decide where a template is detected.
        # */
      	# inline void setDetectionThreshold (float threshold)
		# 
      	# /** \brief Sets the threshold on the magnitude of color gradients. Color gradients with a magnitude below 
        #  *        this threshold are not considered in the detection process.
        #  * \param[in] threshold The threshold on the magnitude of color gradients.
        # */
      	# inline void setGradientMagnitudeThreshold (const float threshold)
		# 
      	# /** \brief Sets the threshold for the decision whether two detections of the same template are merged or not. 
        #  *        If ratio between the intersection of the bounding boxes of two detections and the original bounding 
        #  *        boxes is larger than the specified threshold then they are merged. If detection A overlaps with 
        #  *        detection B and B with C than A, B, and C are merged. Threshold has to be between 0 and 1.
        #  * \param[in] threshold The threshold on the ratio between the intersection bounding box and the original 
        #  *                      bounding box.
        # */
      	# inline void setIntersectionVolumeThreshold (const float threshold = 1.0f)
		# 
      	# /** \brief Sets the input cloud with xyz point coordinates. The cloud has to be organized. 
        #  * \param[in] cloud The input cloud with xyz point coordinates.
        # */
      	# inline void setInputCloud (const typename pcl::PointCloud<PointXYZT>::ConstPtr & cloud)
      	# 
      	# /** \brief Sets the input cloud with rgb values. The cloud has to be organized. 
        #  * \param[in] cloud The input cloud with rgb values.
        # */
      	# inline void setInputColors (const typename pcl::PointCloud<PointRGBT>::ConstPtr & cloud)
		# 
      	# /** \brief Creates a template from the specified data and adds it to the matching queue. 
        #  * \param cloud
        #  * \param object_id
        #  * \param[in] mask_xyz the mask that determine which parts of the xyz-modality are used for creating the template.
        #  * \param[in] mask_rgb the mask that determine which parts of the rgb-modality are used for creating the template.
        #  * \param[in] region the region which will be associated with the template (can be larger than the actual modality-maps).
        #  */
      	# int createAndAddTemplate (
        # 	pcl::PointCloud<pcl::PointXYZRGBA> & cloud,
        # 	const size_t object_id,
        # 	const MaskMap & mask_xyz,
        # 	const MaskMap & mask_rgb,
        # 	const RegionXY & region);
		# 
      	# /** \brief Applies the detection process and fills the supplied vector with the detection instances. 
        #  * \param[out] detections The storage for the detection instances.
        #  */
      	# void detect (std::vector<typename pcl::LineRGBD<PointXYZT, PointRGBT>::Detection> & detections);
		# 
      	# /** \brief Applies the detection process in a semi-scale-invariant manner. This is done by acutally
        #  *        scaling the template to different sizes.
        #  */
      	# void detectSemiScaleInvariant (std::vector<typename pcl::LineRGBD<PointXYZT, PointRGBT>::Detection> & detections,
        #                         		float min_scale = 0.6944444f,
        #                         		float max_scale = 1.44f,
        #                         		float scale_multiplier = 1.2f);
		# 
      	# /** \brief Computes and returns the point cloud of the specified detection. This is the template point 
        #  *        cloud transformed to the detection coordinates. The detection ID refers to the last call of 
        #  *        the method detect (...).
        #  * \param[in] detection_id The ID of the detection (according to the last call of the method detect (...)).
        #  * \param[out] cloud The storage for the transformed points.
        #  */
      	# void computeTransformedTemplatePoints (const size_t detection_id,
        #                                 pcl::PointCloud<pcl::PointXYZRGBA> & cloud);
		# 
      	# /** \brief Finds the indices of the points in the input cloud which correspond to the specified detection. 
        #  *        The detection ID refers to the last call of the method detect (...).
        #  * \param[in] detection_id The ID of the detection (according to the last call of the method detect (...)).
        #  */
      	# inline std::vector<size_t> findObjectPointIndices (const size_t detection_id)

###

# mask_map.h
# namespace pcl
	# class PCL_EXPORTS MaskMap
  	# public:
    	# MaskMap ();
      	# MaskMap (size_t width, size_t height);
      	# virtual ~MaskMap ();
		# 
      	# void resize (size_t width, size_t height);
		# 
      	# inline size_t getWidth () const { return (width_); }
      	# inline size_t getHeight () const { return (height_); }
        # inline unsigned char* getData () { return (&data_[0]); }
		# inline const unsigned char* getData () const { return (&data_[0]); }
      	# static void getDifferenceMask (const MaskMap & mask0,
        #             					const MaskMap & mask1,
        #                  				MaskMap & diff_mask);
		# 
      	# inline void set (const size_t x, const size_t y)
      	# inline void unset (const size_t x, const size_t y)
	    # inline bool isSet (const size_t x, const size_t y) const
      	# inline void reset ()
      	# inline unsigned char & operator() (const size_t x, const size_t y) 
      	# inline const unsigned char &operator() (const size_t x, const size_t y) const
      	# void erode (MaskMap & eroded_mask) const;


###

# model_library.h
# #include <pcl/recognition/ransac_based/model_library.h>
# namespace pcl
# namespace recognition
    # class PCL_EXPORTS ModelLibrary
    	# public:
        # typedef pcl::PointCloud<pcl::PointXYZ> PointCloudIn;
        # typedef pcl::PointCloud<pcl::Normal> PointCloudN;
		# 
        # /** \brief Stores some information about the model. */
        # class Model
        	# public:
            # Model (const PointCloudIn& points, const PointCloudN& normals, float voxel_size, const std::string& object_name,
            #        float frac_of_points_for_registration, void* user_data = NULL)
            # : obj_name_(object_name),
            #   user_data_ (user_data)
            virtual ~Model ()
            inline const std::string& getObjectName () const
            inline const ORROctree& getOctree () const
            inline void* getUserData () const
            inline const float* getOctreeCenterOfMass () const
            inline const float* getBoundsOfOctreePoints () const
            inline const PointCloudIn& getPointsForRegistration () const
          	
		# typedef std::list<std::pair<const ORROctree::Node::Data*, const ORROctree::Node::Data*> > node_data_pair_list;
        # typedef std::map<const Model*, node_data_pair_list> HashTableCell;
        # typedef VoxelStructure<HashTableCell, float> HashTable;
		
      	# public:
        # /** \brief This class is used by 'ObjRecRANSAC' to maintain the object models to be recognized. Normally, you do not need to use
        #   * this class directly. */
        # ModelLibrary (float pair_width, float voxel_size, float max_coplanarity_angle = 3.0f*AUX_DEG_TO_RADIANS/*3 degrees*/);
        # virtual ~ModelLibrary ()
        # 
        # /** \brief Removes all models from the library and clears the hash table. */
        # void removeAllModels ();
		# 
        # /** \brief This is a threshold. The larger the value the more point pairs will be considered as co-planar and will
        #   * be ignored in the off-line model pre-processing and in the online recognition phases. This makes sense only if
        #   * "ignore co-planar points" is on. Call this method before calling addModel. */
        # inline void setMaxCoplanarityAngleDegrees (float max_coplanarity_angle_degrees)
		# 
        # /** \brief Call this method in order NOT to add co-planar point pairs to the hash table. The default behavior
        #   * is ignoring co-planar points on. */
        # inline void ignoreCoplanarPointPairsOn ()
		# 
        # /** \brief Call this method in order to add all point pairs (co-planar as well) to the hash table. The default
        #   * behavior is ignoring co-planar points on. */
        # inline void ignoreCoplanarPointPairsOff ()
		# 
        # /** \brief Adds a model to the hash table.
        #   *
        #   * \param[in] points represents the model to be added.
        #   * \param[in] normals are the normals at the model points.
        #   * \param[in] object_name is the unique name of the object to be added.
        #   * \param[in] frac_of_points_for_registration is the number of points used for fast ICP registration prior to hypothesis testing
        #   * \param[in] user_data is a pointer to some data (can be NULL)
        #   *
        #   * Returns true if model successfully added and false otherwise (e.g., if object_name is not unique). */
        # bool addModel (
        # 		const PointCloudIn& points, const PointCloudN& normals, const std::string& object_name,
        #         float frac_of_points_for_registration, void* user_data = NULL);
		# 
        # /** \brief Returns the hash table built by this instance. */
        # inline const HashTable& getHashTable () const
		# inline const Model* getModel (const std::string& name) const
        # inline const std::map<std::string,Model*>& getModels () const


###

# obj_rec_ransac.h
# #include <pcl/recognition/ransac_based/obj_rec_ransac.h>
# #error "Using pcl/recognition/obj_rec_ransac.h is deprecated, please use pcl/recognition/ransac_based/obj_rec_ransac.h instead."
# namespace pcl
# namespace recognition
    # /** \brief This is a RANSAC-based 3D object recognition method. Do the following to use it: (i) call addModel() k times with k different models
    #   * representing the objects to be recognized and (ii) call recognize() with the 3D scene in which the objects should be recognized. Recognition means both
    #   * object identification and pose (position + orientation) estimation. Check the method descriptions for more details.
    #   *
    #   * \note If you use this code in any academic work, please cite:
    #   *
    #   *   - Chavdar Papazov, Sami Haddadin, Sven Parusel, Kai Krieger and Darius Burschka.
    #   *     Rigid 3D geometry matching for grasping of known objects in cluttered scenes.
    #   *     The International Journal of Robotics Research 2012. DOI: 10.1177/0278364911436019
    #   *
    #   *   - Chavdar Papazov and Darius Burschka.
    #   *     An Efficient RANSAC for 3D Object Recognition in Noisy and Occluded Scenes.
    #   *     In Proceedings of the 10th Asian Conference on Computer Vision (ACCV'10),
    #   *     November 2010.
    #   *
    #   *
    #   * \author Chavdar Papazov
    #   * \ingroup recognition
    #   */
    # class PCL_EXPORTS ObjRecRANSAC
    	# public:
        # typedef ModelLibrary::PointCloudIn PointCloudIn;
        # typedef ModelLibrary::PointCloudN PointCloudN;
		# 
        # typedef BVH<Hypothesis*> BVHH;
		# 
        # /** \brief This is an output item of the ObjRecRANSAC::recognize() method. It contains the recognized model, its name (the ones passed to
        #   * ObjRecRANSAC::addModel()), the rigid transform which aligns the model with the input scene and the match confidence which is a number
        #   * in the interval (0, 1] which gives the fraction of the model surface area matched to the scene. E.g., a match confidence of 0.3 means
        #   * that 30% of the object surface area was matched to the scene points. If the scene is represented by a single range image, the match
        #   * confidence can not be greater than 0.5 since the range scanner sees only one side of each object.
        #   */
        # class Output
        	# public:
            # Output (const std::string& object_name, const float rigid_transform[12], float match_confidence, void* user_data) :
            #   object_name_ (object_name),
            #   match_confidence_ (match_confidence),
            #   user_data_ (user_data)
            # virtual ~Output (){}
			# 
          	# public:
            # std::string object_name_;
            # float rigid_transform_[12];
            # float match_confidence_;
            # void* user_data_;

        # class OrientedPointPair
        	# public:
            # OrientedPointPair (const float *p1, const float *n1, const float *p2, const float *n2)
            #   : p1_ (p1), n1_ (n1), p2_ (p2), n2_ (n2)
            # 
            # virtual ~OrientedPointPair (){}
			# 
            # public:
            #   const float *p1_, *n1_, *p2_, *n2_;

        # class HypothesisCreator
        	# public:
            # HypothesisCreator (){}
            # virtual ~HypothesisCreator (){}
			# 
            # Hypothesis* create (const SimpleOctree<Hypothesis, HypothesisCreator, float>::Node* ) const { return new Hypothesis ();}
        # typedef SimpleOctree<Hypothesis, HypothesisCreator, float> HypothesisOctree;
		# 
		# 
      	# public:
        # /** \brief Constructor with some important parameters which can not be changed once an instance of that class is created.
        #   *
        #   * \param[in] pair_width should be roughly half the extent of the visible object part. This means, for each object point p there should be (at least)
        #   * one point q (from the same object) such that ||p - q|| <= pair_width. Tradeoff: smaller values allow for detection in more occluded scenes but lead
        #   * to more imprecise alignment. Bigger values lead to better alignment but require large visible object parts (i.e., less occlusion).
        #   *
        #   * \param[in] voxel_size is the size of the leafs of the octree, i.e., the "size" of the discretization. Tradeoff: High values lead to less
        #   * computation time but ignore object details. Small values allow to better distinguish between objects, but will introduce more holes in the resulting
        #   * "voxel-surface" (especially for a sparsely sampled scene). */
        # ObjRecRANSAC (float pair_width, float voxel_size);
        # virtual ~ObjRecRANSAC ()
        # 
        # /** \brief Removes all models from the model library and releases some memory dynamically allocated by this instance. */
        # void inline clear()
        # 
        # /** \brief This is a threshold. The larger the value the more point pairs will be considered as co-planar and will
        #   * be ignored in the off-line model pre-processing and in the online recognition phases. This makes sense only if
        #   * "ignore co-planar points" is on. Call this method before calling addModel. This method calls the corresponding
        #   * method of the model library. */
        # inline void setMaxCoplanarityAngleDegrees (float max_coplanarity_angle_degrees)
        # inline void setSceneBoundsEnlargementFactor (float value)
		# 
        # /** \brief Default is on. This method calls the corresponding method of the model library. */
        # inline void ignoreCoplanarPointPairsOn ()
        # 
        # /** \brief Default is on. This method calls the corresponding method of the model library. */
        # inline void ignoreCoplanarPointPairsOff ()
        # 
        # inline void icpHypothesesRefinementOn ()
        # inline void icpHypothesesRefinementOff ()
		# 
        # /** \brief Add an object model to be recognized.
        #   *
        #   * \param[in] points are the object points.
        #   * \param[in] normals at each point.
        #   * \param[in] object_name is an identifier for the object. If that object is detected in the scene 'object_name'
        #   * is returned by the recognition method and you know which object has been detected. Note that 'object_name' has
        #   * to be unique!
        #   * \param[in] user_data is a pointer to some data (can be NULL)
        #   *
        #   * The method returns true if the model was successfully added to the model library and false otherwise (e.g., if 'object_name' is already in use).
        #   */
        # inline bool addModel (const PointCloudIn& points, const PointCloudN& normals, const std::string& object_name, void* user_data = NULL)
        # 
        # /** \brief This method performs the recognition of the models loaded to the model library with the method addModel().
        #   *
        #   * \param[in]  scene is the 3d scene in which the object should be recognized.
        #   * \param[in]  normals are the scene normals.
        #   * \param[out] recognized_objects is the list of output items each one containing the recognized model instance, its name, the aligning rigid transform
        #   * and the match confidence (see ObjRecRANSAC::Output for further explanations).
        #   * \param[in]  success_probability is the user-defined probability of detecting all objects in the scene.
        #   */
        # void recognize (const PointCloudIn& scene, const PointCloudN& normals, std::list<ObjRecRANSAC::Output>& recognized_objects, double success_probability = 0.99);
		# 
        # inline void enterTestModeSampleOPP ()
        # inline void enterTestModeTestHypotheses ()
        # inline void leaveTestMode ()
		# 
        # /** \brief This function is useful for testing purposes. It returns the oriented point pairs which were sampled from the
        #   * scene during the recognition process. Makes sense only if some of the testing modes are active. */
        # inline const std::list<ObjRecRANSAC::OrientedPointPair>& getSampledOrientedPointPairs () const
		# 
        # /** \brief This function is useful for testing purposes. It returns the accepted hypotheses generated during the
        #   * recognition process. Makes sense only if some of the testing modes are active. */
        # inline const std::vector<Hypothesis>& getAcceptedHypotheses () const
		# 
        # /** \brief This function is useful for testing purposes. It returns the accepted hypotheses generated during the
        #   * recognition process. Makes sense only if some of the testing modes are active. */
        # inline void getAcceptedHypotheses (std::vector<Hypothesis>& out) const
		# 
        # /** \brief Returns the hash table in the model library. */
        # inline const pcl::recognition::ModelLibrary::HashTable& getHashTable () const
		# 
        # inline const ModelLibrary& getModelLibrary () const
        # inline const ModelLibrary::Model* getModel (const std::string& name) const
        # inline const ORROctree& getSceneOctree () const
        # inline RigidTransformSpace& getRigidTransformSpace ()
        # inline float getPairWidth () const
		# 
      	# protected:
        # enum Recognition_Mode {SAMPLE_OPP, TEST_HYPOTHESES, /*BUILD_CONFLICT_GRAPH,*/ FULL_RECOGNITION};
        # friend class ModelLibrary;
		# 
        # inline int computeNumberOfIterations (double success_probability) const
        # inline void clearTestData ()
        # void sampleOrientedPointPairs (int num_iterations, const std::vector<ORROctree::Node*>& full_scene_leaves, std::list<OrientedPointPair>& output) const;
        # int generateHypotheses (const std::list<OrientedPointPair>& pairs, std::list<HypothesisBase>& out) const;
		# 
        # /** \brief Groups close hypotheses in 'hypotheses'. Saves a representative for each group in 'out'. Returns the
        #   * number of hypotheses after grouping. */
        # int groupHypotheses(std::list<HypothesisBase>& hypotheses, int num_hypotheses, RigidTransformSpace& transform_space, HypothesisOctree& grouped_hypotheses) const;
        # inline void testHypothesis (Hypothesis* hypothesis, int& match, int& penalty) const;
        # inline void testHypothesisNormalBased (Hypothesis* hypothesis, float& match) const;
        # void buildGraphOfCloseHypotheses (HypothesisOctree& hypotheses, ORRGraph<Hypothesis>& graph) const;
        # void filterGraphOfCloseHypotheses (ORRGraph<Hypothesis>& graph, std::vector<Hypothesis>& out) const;
        # void buildGraphOfConflictingHypotheses (const BVHH& bvh, ORRGraph<Hypothesis*>& graph) const;
        # void filterGraphOfConflictingHypotheses (ORRGraph<Hypothesis*>& graph, std::list<ObjRecRANSAC::Output>& recognized_objects) const;
		# 
        # /** \brief Computes the rigid transform that maps the line (a1, b1) to (a2, b2).
        #  * The computation is based on the corresponding points 'a1' <-> 'a2' and 'b1' <-> 'b2'
        #  * and the normals 'a1_n', 'b1_n', 'a2_n', and 'b2_n'. The result is saved in
        #  * 'rigid_transform' which is an array of length 12. The first 9 elements are the
        #  * rotational part (row major order) and the last 3 are the translation. */
        # inline void computeRigidTransform(
        #   const float *a1, const float *a1_n, const float *b1, const float* b1_n,
        #   const float *a2, const float *a2_n, const float *b2, const float* b2_n,
        #   float* rigid_transform) const
		# 
        # /** \brief Computes the signature of the oriented point pair ((p1, n1), (p2, n2)) consisting of the angles between
        #   * \param p1
        #   * \param n1
        #   * \param p2
        #   * \param n2
        #   * \param[out] signature is an array of three doubles saving the three angles in the order shown above. */
        # static inline void compute_oriented_point_pair_signature (const float *p1, const float *n1, const float *p2, const float *n2, float signature[3])


###

# orr_graph.h
# #include <pcl/recognition/ransac_based/orr_graph.h>
# #error "Using pcl/recognition/orr_graph.h is deprecated, please use pcl/recognition/ransac_based/orr_graph.h instead."
# namespace pcl
# namespace recognition
# template<class NodeData>
# class ORRGraph
		# public:
        # class Node
        	# public:
            # enum State {ON, OFF, UNDEF};
			# 
            # Node (int id)
            # : id_ (id),
            #   state_(UNDEF)
            # virtual ~Node (){}
			# inline const std::set<Node*>& getNeighbors () const
            # inline const NodeData& getData () const
            # inline void setData (const NodeData& data)
            # inline int getId () const
            # inline void setId (int id)
            # inline void setFitness (int fitness)
            # static inline bool compare (const Node* a, const Node* b)
            # friend class ORRGraph;
		# public:
        # ORRGraph (){}
        # virtual ~ORRGraph (){ this->clear ();}
		# inline void clear ()
		# 
        # /** \brief Drops all existing graph nodes and creates 'n' new ones. */
        # inline void resize (int n)
        # inline void computeMaximalOnOffPartition (std::list<Node*>& on_nodes, std::list<Node*>& off_nodes)
        # inline void insertUndirectedEdge (int id1, int id2)
        # inline void insertDirectedEdge (int id1, int id2)
        # inline void deleteUndirectedEdge (int id1, int id2)
        # inline void deleteDirectedEdge (int id1, int id2)
        # inline typename std::vector<Node*>& getNodes (){ return nodes_;}
		# 
      	# public:
        # typename std::vector<Node*> nodes_;


###

# orr_octree.h
# #include <pcl/recognition/ransac_based/orr_octree.h>
# #error "Using pcl/recognition/orr_octree.h is deprecated, please use pcl/recognition/ransac_based/orr_octree.h instead."
# namespace pcl
# namespace recognition
    # /** \brief That's a very specialized and simple octree class. That's the way it is intended to
    #   * be, that's why no templates and stuff like this.
    #   *
    #   * \author Chavdar Papazov
    #   * \ingroup recognition
    #   */
    # class PCL_EXPORTS ORROctree
    	# public:
        # typedef pcl::PointCloud<pcl::PointXYZ> PointCloudIn;
        # typedef pcl::PointCloud<pcl::PointXYZ> PointCloudOut;
        # typedef pcl::PointCloud<pcl::Normal> PointCloudN;
		# 
        # class Node
        	# public:
            # class Data
            	# public:
                # Data (int id_x, int id_y, int id_z, int lin_id, void* user_data = NULL) 
                # : id_x_ (id_x), id_y_ (id_y), id_z_ (id_z), lin_id_ (lin_id), num_points_ (0), user_data_ (user_data)
                # virtual~ Data (){}
				# 
                # inline void addToPoint (float x, float y, float z)
                # inline void computeAveragePoint ()
                # inline void addToNormal (float x, float y, float z) { n_[0] += x; n_[1] += y; n_[2] += z;}
                # inline const float* getPoint () const { return p_;}
                # inline float* getPoint (){ return p_;}
                # inline const float* getNormal () const { return n_;}
                # inline float* getNormal (){ return n_;}
                # inline void get3dId (int id[3]) const
                # inline int get3dIdX () const {return id_x_;}
                # inline int get3dIdY () const {return id_y_;}
                # inline int get3dIdZ () const {return id_z_;}
                # inline int getLinearId () const { return lin_id_;}
                # inline void setUserData (void* user_data){ user_data_ = user_data;}
                # inline void* getUserData () const { return user_data_;}
                # inline void insertNeighbor (Node* node){ neighbors_.insert (node);}
                # inline const std::set<Node*>& getNeighbors () const { return (neighbors_);}

            # Node ()
            # : data_ (NULL),
            #   parent_ (NULL),
            #   children_(NULL)
            # virtual~ Node ()
            # 
            # inline void setCenter(const float *c) { center_[0] = c[0]; center_[1] = c[1]; center_[2] = c[2];}
			# 
            # inline void setBounds(const float *b) { bounds_[0] = b[0]; bounds_[1] = b[1]; bounds_[2] = b[2]; bounds_[3] = b[3]; bounds_[4] = b[4]; bounds_[5] = b[5];}
            # inline void setParent(Node* parent) { parent_ = parent;}
            # inline void setData(Node::Data* data) { data_ = data;}
            # /** \brief Computes the "radius" of the node which is half the diagonal length. */
            # inline void computeRadius()
            # inline const float* getCenter() const { return center_;}
            # inline const float* getBounds() const { return bounds_;}
            # inline void getBounds(float b[6]) const
            # inline Node* getChild (int id) { return &children_[id];}
            # inline Node* getChildren () { return children_;}
            # inline Node::Data* getData (){ return data_;}
            # inline const Node::Data* getData () const { return data_;}
            # inline void setUserData (void* user_data){ data_->setUserData (user_data);}
            # inline Node* getParent (){ return parent_;}
            # inline bool hasData (){ return static_cast<bool> (data_);}
            # inline bool hasChildren (){ return static_cast<bool> (children_);}
            # /** \brief Computes the "radius" of the node which is half the diagonal length. */
            # inline float getRadius (){ return radius_;}
            # bool createChildren ();
			# inline void deleteChildren ()
			# inline void deleteData ()
            # 
            # /** \brief Make this and 'node' neighbors by inserting each node in the others node neighbor set. Nothing happens
            #   * of either of the nodes has no data. */
            # inline void makeNeighbors (Node* node)
		
		
        # ORROctree ();
        # virtual ~ORROctree (){ this->clear ();}
		# void clear ();
		# 
        # /** \brief Creates an octree which encloses 'points' and with leaf size equal to 'voxel_size'.
        #   * 'enlarge_bounds' makes sure that no points from the input will lie on the octree boundary
        #   * by enlarging the bounds by that factor. For example, enlarge_bounds = 1 means that the
        #   * bounds will be enlarged by 100%. The default value is fine. */
        # void build (const PointCloudIn& points, float voxel_size, const PointCloudN* normals = NULL, float enlarge_bounds = 0.00001f);
		# 
        # /** \brief Creates an empty octree with bounds at least as large as the ones provided as input and with leaf
        #   * size equal to 'voxel_size'. */
        # void build (const float* bounds, float voxel_size);
		# 
        # /** \brief Creates the leaf containing p = (x, y, z) and returns a pointer to it, however, only if p lies within
        #   * the octree bounds! A more general version which allows p to be out of bounds is not implemented yet. The method
        #   * returns NULL if p is not within the root bounds. If the leaf containing p already exists nothing happens and
        #   * method just returns a pointer to the leaf. */
        # inline ORROctree::Node* createLeaf (float x, float y, float z)
		# 
        # /** \brief This method returns a super set of the full leavess which are intersected by the sphere
        #   * with radius 'radius' and centered at 'p'. Pointers to the intersected full leaves are saved in
        #   * 'out'. The method computes a super set in the sense that in general not all leaves saved in 'out'
        #   * are really intersected by the sphere. The intersection test is based on the leaf radius (since
        #   * its faster than checking all leaf corners and sides), so we report more leaves than we should,
        #   * but still, this is a fair approximation. */
        # void getFullLeavesIntersectedBySphere (const float* p, float radius, std::list<ORROctree::Node*>& out) const;
		# 
        # /** \brief Randomly chooses and returns a full leaf that is intersected by the sphere with center 'p'
        #   * and 'radius'. Returns NULL if no leaf is intersected by that sphere. */
        # ORROctree::Node* getRandomFullLeafOnSphere (const float* p, float radius) const;
		# 
        # /** \brief Since the leaves are aligned in a rectilinear grid, each leaf has a unique id. The method returns the leaf
        #   * with id [i, j, k] or NULL is no such leaf exists. */
        # ORROctree::Node* getLeaf (int i, int j, int k)
		# 
        # /** \brief Returns a pointer to the leaf containing p = (x, y, z) or NULL if no such leaf exists. */
        # inline ORROctree::Node* getLeaf (float x, float y, float z)
        # 
        # /** \brief Deletes the branch 'node' is part of. */
        # void deleteBranch (Node* node);
		# 
        # /** \brief Returns a vector with all octree leaves which contain at least one point. */
        # inline std::vector<ORROctree::Node*>& getFullLeaves () { return full_leaves_;}
        # inline const std::vector<ORROctree::Node*>& getFullLeaves () const { return full_leaves_;}
        # void getFullLeavesPoints (PointCloudOut& out) const;
        # void getNormalsOfFullLeaves (PointCloudN& out) const;
        # inline ORROctree::Node* getRoot (){ return root_;}
        # inline const float* getBounds () const
        # inline void getBounds (float b[6]) const
        # inline float getVoxelSize () const { return voxel_size_;}
        # inline void insertNeighbors (Node* node)


###

# orr_octree_zprojection.h
# #include <pcl/recognition/ransac_based/orr_octree_zprojection.h>
# #error "Using pcl/recognition/orr_octree_zprojection.h is deprecated, please use pcl/recognition/ransac_based/orr_octree_zprojection.h instead."
# namespace pcl
# namespace recognition
	# class ORROctree;
    # class PCL_EXPORTS ORROctreeZProjection
    	# public:
        # class Pixel
        	# public:
            # Pixel (int id): id_ (id) {}
            # inline void set_z1 (float z1) { z1_ = z1;}
            # inline void set_z2 (float z2) { z2_ = z2;}
            # float z1 () const { return z1_;}
            # float z2 () const { return z2_;}
            # int getId () const { return id_;}

          	# protected:
            # float z1_, z2_;
            # int id_;

    	# public:
        # class Set
        	# public:
            # Set (int x, int y) : nodes_ (compare_nodes_z), x_ (x), y_ (y)
			# 
            # static inline bool compare_nodes_z (ORROctree::Node* node1, ORROctree::Node* node2)
			# inline void insert (ORROctree::Node* leaf) { nodes_.insert(leaf);}
			# inline std::set<ORROctree::Node*, bool(*)(ORROctree::Node*,ORROctree::Node*)>& get_nodes (){ return nodes_;}
			# inline int get_x () const { return x_;}
			# inline int get_y () const { return y_;}

      	# public:
        # ORROctreeZProjection () : pixels_(NULL), sets_(NULL)
        # virtual ~ORROctreeZProjection (){ this->clear();}
		# void build (const ORROctree& input, float eps_front, float eps_back);
		# void clear ();
        # inline void getPixelCoordinates (const float* p, int& x, int& y) const
		# inline const Pixel* getPixel (const float* p) const
		# inline Pixel* getPixel (const float* p)
		# inline const std::set<ORROctree::Node*, bool(*)(ORROctree::Node*,ORROctree::Node*)>* getOctreeNodes (const float* p) const
		# 
        # inline std::list<Pixel*>& getFullPixels (){ return full_pixels_;}
        # inline const Pixel* getPixel (int i, int j) const
        # inline float getPixelSize () const
        # inline const float* getBounds () const
        # /** \brief Get the width ('num_x') and height ('num_y') of the image. */
        # inline void getNumberOfPixels (int& num_x, int& num_y) const


###

# point_types.h
# namespace pcl
# /** \brief A point structure representing Euclidean xyz coordinates, and the intensity value.
# * \ingroup common
# */
# struct EIGEN_ALIGN16 GradientXY
    # union
    # {
    	# struct
      	# {
	    #     float x;
        # 	float y;
        # 	float angle;
        # 	float magnitude;
      	# };
      	# float data[4];
    # };
    # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	# 
    # inline bool operator< (const GradientXY & rhs)
	# 
	# inline std::ostream & operator << (std::ostream & os, const GradientXY & p)

###

# quantized_map.h
# namespace pcl
	# class PCL_EXPORTS QuantizedMap
  	# {
    	# public:
      	# QuantizedMap ();
      	# QuantizedMap (size_t width, size_t height);
      	# QuantizedMap (const QuantizedMap & copy_me);
      	# virtual ~QuantizedMap ();
		# 
      	# inline size_t getWidth () const { return (width_); }
      	# inline size_t getHeight () const { return (height_); }
      	# inline unsigned char* getData () { return (&data_[0]); }
		# inline const unsigned char* getData () const { return (&data_[0]); }
		# inline QuantizedMap getSubMap (
		# 		size_t x,
        #         size_t y,
        #         size_t width,
        #         size_t height)
		# 
      	# void resize (size_t width, size_t height);
		# 
      	# inline unsigned char & operator() (const size_t x, const size_t y) 
	    # inline const unsigned char &operator() (const size_t x, const size_t y) const
		# 
      	# static void spreadQuantizedMap (const QuantizedMap & input_map, QuantizedMap & output_map, size_t spreading_size);
		# 
      	# void serialize (std::ostream & stream) const
		# 
      	# void deserialize (std::istream & stream)
    	# //private:
      	# std::vector<unsigned char> data_;
      	# size_t width_;
      	# size_t height_;  

###

# region_xy.h
# namespace pcl
    # /** \brief Function for reading data from a stream. */
    # template <class Type> void read (std::istream & stream, Type & value)
	# 
  	# /** \brief Function for reading data arrays from a stream. */
  	# template <class Type> void read (std::istream & stream, Type * value, int nr_values)
  	# 
  	# /** \brief Function for writing data to a stream. */
  	# template <class Type> void write (std::ostream & stream, Type value)
	# 
  	# /** \brief Function for writing data arrays to a stream. */
  	# template <class Type> void write (std::ostream & stream, Type * value, int nr_values)
	# 
  	# /** \brief Defines a region in XY-space.
    #  * \author Stefan Holzer
    #  */
  	# struct PCL_EXPORTS RegionXY
    # 	/** \brief Constructor. */
    # 	RegionXY () : x (0), y (0), width (0), height (0) {}
	# 	
    # 	/** \brief x-position of the region. */
    # 	int x;
    # 	/** \brief y-position of the region. */
    # 	int y;
    # 	/** \brief width of the region. */
    # 	int width;
    # 	/** \brief height of the region. */
    # 	int height;
	# 	/** \brief Serializes the object to the specified stream.
    #    * \param[out] stream the stream the object will be serialized to. */
    # 	void serialize (std::ostream & stream) const
	# 	
    # 	/** \brief Deserializes the object from the specified stream.
    #    * \param[in] stream the stream the object will be deserialized from. */
    # 	void deserialize (::std::istream & stream)

###

# rigid_transform_space.h
# namespace pcl
# namespace recognition
# class RotationSpaceCell
		# public:
        # class Entry
        # {
        # 	public:
        #     Entry () : num_transforms_ (0)
        #     Entry (const Entry& src) : num_transforms_ (src.num_transforms_)
        #     const Entry& operator = (const Entry& src)
        #     inline const Entry& addRigidTransform (const float axis_angle[3], const float translation[3])
        #     inline void computeAverageRigidTransform (float *rigid_transform = NULL)
        #     inline const float* getAxisAngle () const
        #     inline const float* getTranslation () const
        #     inline int getNumberOfTransforms () const
        # };// class Entry
		# 
		# 
      	# public:
        # RotationSpaceCell (){}
        # virtual ~RotationSpaceCell ()
        # 
        # inline std::map<const ModelLibrary::Model*,Entry>& getEntries ()
		# 
        # inline const RotationSpaceCell::Entry* getEntry (const ModelLibrary::Model* model) const
		# 
        # inline const RotationSpaceCell::Entry& addRigidTransform (const ModelLibrary::Model* model, const float axis_angle[3], const float translation[3])
    # }; // class RotationSpaceCell
	# 
    # class RotationSpaceCellCreator
    # {
    #   public:
    #     RotationSpaceCellCreator (){}
    #     virtual ~RotationSpaceCellCreator (){}
	# 
    #     RotationSpaceCell* create (const SimpleOctree<RotationSpaceCell, RotationSpaceCellCreator, float>::Node* )
    #     {
    #       return (new RotationSpaceCell ());
    #     }
    # };
	# 
    # typedef SimpleOctree<RotationSpaceCell, RotationSpaceCellCreator, float> CellOctree;
	# 
	# 
    # /** \brief This is a class for a discrete representation of the rotation space based on the axis-angle representation.
    #   * This class is not supposed to be very general. That's why it is dependent on the class ModelLibrary.
    #   * \author Chavdar Papazov
    #   * \ingroup recognition
    #   */
    # class PCL_EXPORTS RotationSpace
    # {
    	# public:
        # /** \brief We use the axis-angle representation for rotations. The axis is encoded in the vector
        #   * and the angle is its magnitude. This is represented in an octree with bounds [-pi, pi]^3. */
        # RotationSpace (float discretization)
        # 
        # inline void setCenter (const float* c)
        # inline const float* getCenter () const { return center_;}
        # inline bool getTransformWithMostVotes (const ModelLibrary::Model* model, float rigid_transform[12]) const
        # inline bool addRigidTransform (const ModelLibrary::Model* model, const float axis_angle[3], const float translation[3])
    # };// class RotationSpace
	
    # class RotationSpaceCreator
    	# public:
        # RotationSpaceCreator() : counter_ (0)
        # virtual ~RotationSpaceCreator(){}
        # RotationSpace* create(const SimpleOctree<RotationSpace, RotationSpaceCreator, float>::Node* leaf)
        # void setDiscretization (float value){ discretization_ = value;}
        # int getNumberOfRotationSpaces () const { return (counter_);}
        # const std::list<RotationSpace*>& getRotationSpaces () const { return (rotation_spaces_);}
		# 
        # std::list<RotationSpace*>& getRotationSpaces (){ return (rotation_spaces_);}
		# 
        # void reset ()
	# 
    # typedef SimpleOctree<RotationSpace, RotationSpaceCreator, float> RotationSpaceOctree;

    # class PCL_EXPORTS RigidTransformSpace
    	# public:
        # RigidTransformSpace (){}
        # virtual ~RigidTransformSpace (){ this->clear ();}
        inline void build (const float* pos_bounds, float translation_cell_size, float rotation_cell_size)
        inline void clear ()
        inline std::list<RotationSpace*>& getRotationSpaces ()
        inline const std::list<RotationSpace*>& getRotationSpaces () const
        inline int getNumberOfOccupiedRotationSpaces ()
        inline bool addRigidTransform (const ModelLibrary::Model* model, const float position[3], const float rigid_transform[12])

###

# simple_octree.h
# namespace pcl
# namespace recognition
# template<typename NodeData, typename NodeDataCreator, typename Scalar = float>
# class PCL_EXPORTS SimpleOctree
	# public:
    #     class Node
    #     	public:
    #         Node ();
	# 		virtual~ Node ();
    #         inline void setCenter (const Scalar *c);
    #         inline void setBounds (const Scalar *b);
    #         inline const Scalar* getCenter () const { return center_;}
    #         inline const Scalar* getBounds () const { return bounds_;}
    #         inline void getBounds (Scalar b[6]) const { memcpy (b, bounds_, 6*sizeof (Scalar));}
    #         inline Node* getChild (int id) { return &children_[id];}
    #         inline Node* getChildren () { return children_;}
    #         inline void setData (const NodeData& src){ *data_ = src;}
    #         inline NodeData& getData (){ return *data_;}
    #         inline const NodeData& getData () const { return *data_;}
    #         inline Node* getParent (){ return parent_;}
    #         inline float getRadius () const { return radius_;}
    #         inline bool hasData (){ return static_cast<bool> (data_);}
    #         inline bool hasChildren (){ return static_cast<bool> (children_);}
    #         inline const std::set<Node*>& getNeighbors () const { return (full_leaf_neighbors_);}
    #         inline void deleteChildren ();
    #         inline void deleteData ();
    #         friend class SimpleOctree;
    #     };
    # 
    # 
    	# SimpleOctree ();
    	# virtual ~SimpleOctree ();
        # void clear ();
		# 
        # /** \brief Creates an empty octree with bounds at least as large as the ones provided as input and with leaf
        #   * size equal to 'voxel_size'. */
        # void build (const Scalar* bounds, Scalar voxel_size, NodeDataCreator* node_data_creator);
		# 
        # /** \brief Creates the leaf containing p = (x, y, z) and returns a pointer to it, however, only if p lies within
        #   * the octree bounds! A more general version which allows p to be out of bounds is not implemented yet. The method
        #   * returns NULL if p is not within the root bounds. If the leaf containing p already exists nothing happens and
        #   * method just returns a pointer to the leaf. Note that for a new created leaf, the method also creates its data
        #   * object. */
        # inline Node* createLeaf (Scalar x, Scalar y, Scalar z);
		# 
        # /** \brief Since the leaves are aligned in a rectilinear grid, each leaf has a unique id. The method returns the full
        #   * leaf, i.e., the one having a data object, with id [i, j, k] or NULL is no such leaf exists. */
        # inline Node* getFullLeaf (int i, int j, int k);
		# 
        # /** \brief Returns a pointer to the full leaf, i.e., one having a data pbject, containing p = (x, y, z) or NULL if no such leaf exists. */
        # inline Node* getFullLeaf (Scalar x, Scalar y, Scalar z);
		# 
        # inline std::vector<Node*>& getFullLeaves () { return full_leaves_;}
		# 
        # inline const std::vector<Node*>& getFullLeaves () const { return full_leaves_;}
		# 
        # inline Node* getRoot (){ return root_;}
		# 
        # inline const Scalar* getBounds () const { return (bounds_);}
		# 
        # inline void getBounds (Scalar b[6]) const { memcpy (b, bounds_, 6*sizeof (Scalar));}
		# 
        # inline Scalar getVoxelSize () const { return voxel_size_;}
		
###

# sparse_quantized_multi_mod_template.h
# namespace pcl
# 
# /** \brief Feature that defines a position and quantized value in a specific modality. 
# * \author Stefan Holzer
# */
# struct QuantizedMultiModFeature
    # /** \brief Constructor. */
    # QuantizedMultiModFeature () : x (0), y (0), modality_index (0), quantized_value (0) {}
	# 
    # /** \brief x-position. */
    # int x;
    # /** \brief y-position. */
    # int y;
    # /** \brief the index of the corresponding modality. */
    # size_t modality_index;
    # /** \brief the quantized value attached to the feature. */
    # unsigned char quantized_value;
	# 
    # /** \brief Compares whether two features are the same. 
    #   * \param[in] base the feature to compare to.
    #   */
    # bool compareForEquality (const QuantizedMultiModFeature & base)
    # 
    # /** \brief Serializes the object to the specified stream.
    #   * \param[out] stream the stream the object will be serialized to. */
    # void serialize (std::ostream & stream) const
    # 
    # /** \brief Deserializes the object from the specified stream.
    #   * \param[in] stream the stream the object will be deserialized from. */
    # void deserialize (std::istream & stream)
    # 
  	# /** \brief A multi-modality template constructed from a set of quantized multi-modality features.
    #  * \author Stefan Holzer 
    #  */
  	# struct SparseQuantizedMultiModTemplate
  		# /** \brief Constructor. */
    	# SparseQuantizedMultiModTemplate () : features (), region () {}
		# 
    	# /** \brief The storage for the multi-modality features. */
    	# std::vector<QuantizedMultiModFeature> features;
		# 
    	# /** \brief The region assigned to the template. */
    	# RegionXY region;
		# 
    	# /** \brief Serializes the object to the specified stream.
      	#  * \param[out] stream the stream the object will be serialized to. */
    	# void serialize (std::ostream & stream) const
    	# 
    	# /** \brief Deserializes the object from the specified stream.
      	#  * \param[in] stream the stream the object will be deserialized from. */
    	# void deserialize (std::istream & stream)


###

# surface_normal_modality.h
# namespace pcl
# /** \brief Map that stores orientations.
#  * \author Stefan Holzer
#  */
# struct PCL_EXPORTS LINEMOD_OrientationMap
    	# public:
      	# /** \brief Constructor. */
      	# inline LINEMOD_OrientationMap () : width_ (0), height_ (0), map_ () {}
      	# /** \brief Destructor. */
      	# inline ~LINEMOD_OrientationMap () {}
		# 
      	# /** \brief Returns the width of the modality data map. */
      	# inline size_t getWidth () const
		# 
      	# /** \brief Returns the height of the modality data map. */
      	# inline size_t getHeight () const
		# 
      	# /** \brief Resizes the map to the specific width and height and initializes 
        # 	*        all new elements with the specified value.
        # 	* \param[in] width the width of the resized map.
        # 	* \param[in] height the height of the resized map.
        # 	* \param[in] value the value all new elements will be initialized with.
        # */
      	# inline void resize (const size_t width, const size_t height, const float value)
		# 
      	# /** \brief Operator to access elements of the map. 
        # 	* \param[in] col_index the column index of the element to access.
        # 	* \param[in] row_index the row index of the element to access.
        # */
      	# inline float & operator() (const size_t col_index, const size_t row_index)
		# 
      	# /** \brief Operator to access elements of the map. 
        # 	* \param[in] col_index the column index of the element to access.
        # 	* \param[in] row_index the row index of the element to access.
        # */
      	# inline const float &operator() (const size_t col_index, const size_t row_index) const
		# 
		# /** \brief Look-up-table for fast surface normal quantization.
		# * \author Stefan Holzer
		# */
		# struct QuantizedNormalLookUpTable
		# {
		#     /** \brief The range of the LUT in x-direction. */
		#     int range_x;
		#     /** \brief The range of the LUT in y-direction. */
		#     int range_y;
		#     /** \brief The range of the LUT in z-direction. */
		#     int range_z;
		# 
		#     /** \brief The offset in x-direction. */
		#     int offset_x;
		#     /** \brief The offset in y-direction. */
		#     int offset_y;
		#     /** \brief The offset in z-direction. */
		#     int offset_z;
		# 
		#     /** \brief The size of the LUT in x-direction. */
		#     int size_x;
		#     /** \brief The size of the LUT in y-direction. */
		#     int size_y;
		#     /** \brief The size of the LUT in z-direction. */
		#     int size_z;
		# 
		#     /** \brief The LUT data. */
		#     unsigned char * lut;
		# 
		#     /** \brief Constructor. */
		#     QuantizedNormalLookUpTable () : 
		#       range_x (-1), range_y (-1), range_z (-1), 
		#       offset_x (-1), offset_y (-1), offset_z (-1), 
		#       size_x (-1), size_y (-1), size_z (-1), lut (NULL) 
		#     {}
		# 
		#     /** \brief Destructor. */
		#     ~QuantizedNormalLookUpTable () 
		#     { 
		#       if (lut != NULL) 
		#         delete[] lut; 
		#     }
		# 
    	# /** \brief Initializes the LUT.
      	# * \param[in] range_x_arg the range of the LUT in x-direction.
      	# * \param[in] range_y_arg the range of the LUT in y-direction.
      	# * \param[in] range_z_arg the range of the LUT in z-direction.
      	# */
    	# void initializeLUT (const int range_x_arg, const int range_y_arg, const int range_z_arg)
		# 
    	# /** \brief Operator to access an element in the LUT.
      	# * \param[in] x the x-component of the normal.
      	# * \param[in] y the y-component of the normal.
      	# * \param[in] z the z-component of the normal. 
      	# */
    	# inline unsigned char operator() (const float x, const float y, const float z) const
		# 
    	# /** \brief Operator to access an element in the LUT.
      	# * \param[in] index the index of the element. 
      	# */
    	# inline unsigned char operator() (const int index) const


# /** \brief Modality based on surface normals.
# * \author Stefan Holzer
# */
# template <typename PointInT>
# class SurfaceNormalModality : public QuantizableModality, public PCLBase<PointInT>
        # protected:
        # using PCLBase<PointInT>::input_;
        # 
        # /** \brief Candidate for a feature (used in feature extraction methods). */
        # struct Candidate
        #     /** \brief Constructor. */
        #   Candidate () : normal (), distance (0.0f), bin_index (0), x (0), y (0) {}
        #   
        #   /** \brief Normal. */
        #   Normal normal;
        #   /** \brief Distance to the next different quantized value. */
        #   float distance;
        #   
        #   /** \brief Quantized value. */
        #   unsigned char bin_index;
        #     
        #   /** \brief x-position of the feature. */
        #   size_t x;
        #   /** \brief y-position of the feature. */
        #   size_t y;   
        #   /** \brief Compares two candidates based on their distance to the next different quantized value. 
        #       * \param[in] rhs the candidate to compare with. 
        #       */
        #   bool operator< (const Candidate & rhs) const
        # 
        # public:
        #   typedef typename pcl::PointCloud<PointInT> PointCloudIn;
        # 
        # /** \brief Constructor. */
        # SurfaceNormalModality ();
        # /** \brief Destructor. */
        # virtual ~SurfaceNormalModality ();
        # 
        # /** \brief Sets the spreading size.
        #  * \param[in] spreading_size the spreading size.
        #  */
        # inline void setSpreadingSize (const size_t spreading_size)
        # 
        # /** \brief Enables/disables the use of extracting a variable number of features.
        #  * \param[in] enabled specifies whether extraction of a variable number of features will be enabled/disabled.
        #  */
        # inline void setVariableFeatureNr (const bool enabled)
        # 
        # /** \brief Returns the surface normals. */
        # inline pcl::PointCloud<pcl::Normal> &getSurfaceNormals ()
        # 
        # /** \brief Returns the surface normals. */
        # inline const pcl::PointCloud<pcl::Normal> &getSurfaceNormals () const
        # 
        # /** \brief Returns a reference to the internal quantized map. */
        # inline QuantizedMap &getQuantizedMap () 
        # 
        # /** \brief Returns a reference to the internal spreaded quantized map. */
        # inline QuantizedMap &getSpreadedQuantizedMap () 
        # 
        # /** \brief Returns a reference to the orientation map. */
        # inline LINEMOD_OrientationMap &getOrientationMap ()
        # 
        # /** \brief Extracts features from this modality within the specified mask.
        #   * \param[in] mask defines the areas where features are searched in. 
        #   * \param[in] nr_features defines the number of features to be extracted 
        #   *            (might be less if not sufficient information is present in the modality).
        #   * \param[in] modality_index the index which is stored in the extracted features.
        #   * \param[out] features the destination for the extracted features.
        # */
        # void extractFeatures (
        #           const MaskMap & mask, size_t nr_features, size_t modality_index,
        #             std::vector<QuantizedMultiModFeature> & features) const;
        # 
        # /** \brief Extracts all possible features from the modality within the specified mask.
        #   * \param[in] mask defines the areas where features are searched in. 
        #   * \param[in] nr_features IGNORED (TODO: remove this parameter).
        #   * \param[in] modality_index the index which is stored in the extracted features.
        #   * \param[out] features the destination for the extracted features.
        # */
        # void extractAllFeatures (
        #           const MaskMap & mask, size_t nr_features, size_t modality_index,
        #             std::vector<QuantizedMultiModFeature> & features) const;
        # 
        # /** \brief Provide a pointer to the input dataset (overwrites the PCLBase::setInputCloud method)
        #  * \param[in] cloud the const boost shared pointer to a PointCloud message
        #  */
        # virtual void setInputCloud (const typename PointCloudIn::ConstPtr & cloud) 
        # 
        # /** \brief Processes the input data (smoothing, computing gradients, quantizing, filtering, spreading). */
        # virtual void processInputData ();
        # 
        # /** \brief Processes the input data assuming that everything up to filtering is already done/available 
        #  *        (so only spreading is performed). */
        # virtual void processInputDataFromFiltered ();


# template <typename PointInT> pcl::SurfaceNormalModality<PointInT>::SurfaceNormalModality ()

# template <typename PointInT> pcl::SurfaceNormalModality<PointInT>::~SurfaceNormalModality ()

# template <typename PointInT> void pcl::SurfaceNormalModality<PointInT>::processInputData ()

# template <typename PointInT> void pcl::SurfaceNormalModality<PointInT>::processInputDataFromFiltered ()

# template <typename PointInT> void pcl::SurfaceNormalModality<PointInT>::computeSurfaceNormals ()

# template <typename PointInT> void pcl::SurfaceNormalModality<PointInT>::computeAndQuantizeSurfaceNormals ()

# static void accumBilateral(long delta, long i, long j, long * A, long * b, int threshold)

# /**
#  * \brief Compute quantized normal image from depth image.
#  * Implements section 2.6 "Extension to Dense Depth Sensors."
#  * \todo Should also need camera model, or at least focal lengths? Replace distance_threshold with mask?
#  */
# template <typename PointInT> void pcl::SurfaceNormalModality<PointInT>::computeAndQuantizeSurfaceNormals2 ()

# template <typename PointInT> void pcl::SurfaceNormalModality<PointInT>::extractFeatures (const MaskMap & mask,
#                                                        const size_t nr_features,
#                                                        const size_t modality_index,
#                                                        std::vector<QuantizedMultiModFeature> & features) const

# template <typename PointInT> void pcl::SurfaceNormalModality<PointInT>::extractAllFeatures (
#     	const MaskMap & mask, const size_t, const size_t modality_index, std::vector<QuantizedMultiModFeature> & features) const

# template <typename PointInT> void pcl::SurfaceNormalModality<PointInT>::quantizeSurfaceNormals ()

# pcl::SurfaceNormalModality<PointInT>::filterQuantizedSurfaceNormals ()

# template <typename PointInT> void
# pcl::SurfaceNormalModality<PointInT>::computeDistanceMap (const MaskMap & input, DistanceMap & output) const


# trimmed_icp.h
# namespace pcl
# namespace recognition
# template<typename PointT, typename Scalar>
# class PCL_EXPORTS TrimmedICP : public pcl::registration::TransformationEstimationSVD<PointT, PointT, Scalar>
# {
cdef extern from "pcl/Recognition/trimmed_icp.h" namespace "pcl::registration":
    cdef cppclass TrimmedICP[PointT, Scalar](pcl::registration::TransformationEstimationSVD[PointT, PointT, Scalar])
        TrimmedICP ()
      	# public:
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # typedef typename Eigen::Matrix<Scalar, 4, 4> Matrix4;
      	# public:
        # TrimmedICP () : new_to_old_energy_ratio_ (0.99f)
		# 
        # /** \brief Call this method before calling align().
        #   * \param[in] target is target point cloud. The method builds a kd-tree based on 'target' for performing fast closest point search.
        #   *            The source point cloud will be registered to 'target' (see align() method).
        #   * */
        # inline void init (const PointCloudConstPtr& target)
		# 
        # /** \brief The method performs trimmed ICP, i.e., it rigidly registers the source to the target (passed to the init() method).
        #   * \param[in] source_points is the point cloud to be registered to the target.
        #   * \param[in] num_source_points_to_use gives the number of closest source points taken into account for registration. By closest
        #   * source points we mean the source points closest to the target. These points are computed anew at each iteration.
        #   * \param[in,out] guess_and_result is the estimated rigid transform. IMPORTANT: this matrix is also taken as the initial guess
        #   * for the alignment. If there is no guess, set the matrix to identity!
        #   * */
        # inline void align (const PointCloud& source_points, int num_source_points_to_use, Matrix4& guess_and_result) const
		# 
        # inline void setNewToOldEnergyRatio (float ratio)

#endif /* TRIMMED_ICP_H_ */
###

# voxel_structure.h
# namespace pcl
# namespace recognition
# 
# /** \brief This class is a box in R3 built of voxels ordered in a regular rectangular grid. Each voxel is of type T. */
# template<class T, typename REAL = float>
# class VoxelStructure
cdef extern from "pcl/Recognition/voxel_structure.h" namespace "pcl::recognition":
    cdef cppclass VoxelStructure[T, float]
        VoxelStructure ()
        # public:
        # inline VoxelStructure (): voxels_(NULL){}
        # inline virtual ~VoxelStructure (){ this->clear();}
        # 
        # /** \brief Call this method before using an instance of this class. Parameter meaning is obvious. */
        # inline void build (const REAL bounds[6], int num_of_voxels[3]);
        # 
        # /** \brief Release the memory allocated for the voxels. */
        # inline void clear (){ if ( voxels_ ){ delete[] voxels_; voxels_ = NULL;}}
        # 
        # /** \brief Returns a pointer to the voxel which contains p or NULL if p is not inside the structure. */
        # inline T* getVoxel (const REAL p[3]);
        # 
        # /** \brief Returns a pointer to the voxel with integer id (x,y,z) or NULL if (x,y,z) is out of bounds. */
        # inline T* getVoxel (int x, int y, int z) const;
        # 
        # /** \brief Returns the linear voxel array. */
        # const inline T* getVoxels () const
        # 
        # /** \brief Returns the linear voxel array. */
        # inline T* getVoxels ()
        # 
        # /** \brief Converts a linear id to a 3D id, i.e., computes the integer 3D coordinates of a voxel from its position in the voxel array.
        #     *
        #     * \param[in]  linear_id the position of the voxel in the internal voxel array.
        #     * \param[out] id3d an array of 3 integers for the integer coordinates of the voxel. */
        # inline void compute3dId (int linear_id, int id3d[3]) const
        # 
        # /** \brief Returns the number of voxels in x, y and z direction. */
        # inline const int* getNumberOfVoxelsXYZ() const
		# 
        # /** \brief Computes the center of the voxel with given integer coordinates.
        #     *
        #     * \param[in]  id3 the integer coordinates along the x, y and z axis.
        #     * \param[out] center */
        # inline void computeVoxelCenter (const int id3[3], REAL center[3]) const
		# 
        # /** \brief Returns the total number of voxels. */
        # inline int getNumberOfVoxels() const
		# 
        # /** \brief Returns the bounds of the voxel structure, which is pointer to the internal array of 6 doubles: (min_x, max_x, min_y, max_y, min_z, max_z). */
        # inline const float* getBounds() const
		# 
        # /** \brief Copies the bounds of the voxel structure to 'b'. */
        # inline void getBounds(REAL b[6]) const
		# 
        # /** \brief Returns the voxel spacing in x, y and z direction. That's the same as the voxel size along each axis. */
        # const REAL* getVoxelSpacing() const
		# 
        # /** \brief Saves pointers to the voxels which are neighbors of the voxels which contains 'p'. The containing voxel is returned too.
        #     * 'neighs' has to be an array of pointers with space for at least 27 pointers (27 = 3^3 which is the max number of neighbors). The */
        # inline int getNeighbors (const REAL* p, T **neighs) const;

###
