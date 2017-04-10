# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features_180 as pclftr


cdef class RangeImageBorderExtractor:
    """
    RangeImageBorderExtractor class for 
    """
    cdef pclftr.RangeImageBorderExtractor_t *me
    
    def __cinit__(self):
        self.me = new pclftr.RangeImageBorderExtractor_t()
    
    def __dealloc__(self):
        del self.me
    
    def set_RangeImage(self, RangeImage rangeImage):
        self.me.setRangeImage(range_image)
    
    def ClearData (self):
        clearData ()
    
    # def GetAnglesImageForBorderDirections ()
    #   data = self.me.getAnglesImageForBorderDirections()
    #   return data
    
    # def GetAnglesImageForSurfaceChangeDirections ()
    #   data = self.me.getAnglesImageForSurfaceChangeDirections ()
    #   return data
    
    # def Compute ()
    #   output = pcl.PointCloudOut()
    #   self.me.compute (output)
    #   return output
    
    
    # Parameters& getParameters ()
    # def GetParameters ()
    #     return self.me.getParameters ()

    # 
    # def HasRangeImage ()
    #     # cdef param = self.me.hasRangeImage ()
    #     return self.me.hasRangeImage ()
    # 
    # # 
    # def GetRangeImage()
    #     const pcl_r_img.RangeImage 
    #     self.me.getRangeImage ()
    # 
    # def GetBorderScoresLeft ()
    #     float[] data = self.me.getBorderScoresLeft ()   
    #     return data
    # 
    # def GetBorderScoresRight ()
    #     float[] data = self.me.getBorderScoresRight ()  
    #     return data
    # 
    # def GetBorderScoresTop ()
    #     float[] data = self.me.getBorderScoresTop ()  
    #     return data
    # 
    # def GetBorderScoresBottom ()
    #     float[] data = self.me.getBorderScoresBottom ()  
    #     return data



###

