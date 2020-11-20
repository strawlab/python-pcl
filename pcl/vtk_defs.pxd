# vtkObjectBase
# class VTKCOMMONCORE_EXPORT vtkObjectBase
cdef extern from "vtkObjectBase.h" nogil:
    cdef cppclass vtkObjectBase:
        vtkObjectBase()
        # public:
        # /**
        # * Return the class name as a string.
        # */
        # const char* GetClassName() const;
        # // Define possible mangled names.
        # const char* GetClassNameA() const;
        # const char* GetClassNameW() const;
        # 
        # /**
        # * Return 1 if this class type is the same type of (or a subclass of)
        # * the named class. Returns 0 otherwise. This method works in
        # * combination with vtkTypeMacro found in vtkSetGet.h.
        # */
        # static vtkTypeBool IsTypeOf(const char *name);
        # 
        # /**
        # * Return 1 if this class is the same type of (or a subclass of)
        # * the named class. Returns 0 otherwise. This method works in
        # * combination with vtkTypeMacro found in vtkSetGet.h.
        # */
        # virtual vtkTypeBool IsA(const char *name);
        # 
        # /**
        # * Delete a VTK object.  This method should always be used to delete
        # * an object when the New() method was used to create it. Using the
        # * C++ delete method will not work with reference counting.
        # */
        # virtual void Delete();
        # 
        # /**
        # * Delete a reference to this object.  This version will not invoke
        # * garbage collection and can potentially leak the object if it is
        # * part of a reference loop.  Use this method only when it is known
        # * that the object has another reference and would not be collected
        # * if a full garbage collection check were done.
        # */
        # virtual void FastDelete();
        # 
        # /**
        # * Create an object with Debug turned off, modified time initialized
        # * to zero, and reference counting on.
        # */
        # static vtkObjectBase *New()
        # 
        # // Called by implementations of vtkObject::New(). Centralized location for
        # // vtkDebugLeaks registration:
        # void InitializeObjectBase();
        # 
        # #ifdef _WIN32
        # // avoid dll boundary problems
        # void* operator new( size_t tSize );
        # void operator delete( void* p );
        # #endif
        # 
        # /**
        # * Print an object to an ostream. This is the method to call
        # * when you wish to see print the internal state of an object.
        # */
        # void Print(ostream& os);
        # 
        # //@{
        # /**
        # * Methods invoked by print to print information about the object
        # * including superclasses. Typically not called by the user (use
        # * Print() instead) but used in the hierarchical print process to
        # * combine the output of several classes.
        # */
        # virtual void PrintSelf(ostream& os, vtkIndent indent);
        # virtual void PrintHeader(ostream& os, vtkIndent indent);
        # virtual void PrintTrailer(ostream& os, vtkIndent indent);
        # //@}
        # 
        # /**
        # * Increase the reference count (mark as used by another object).
        # */
        # virtual void Register(vtkObjectBase* o);
        # 
        # /**
        # * Decrease the reference count (release by another object). This
        # * has the same effect as invoking Delete() (i.e., it reduces the
        # * reference count by 1).
        # */
        # virtual void UnRegister(vtkObjectBase* o);
        # 
        # """Return the current reference count of this object.
        # """
        int GetReferenceCount()
        
        # """Sets the reference count. (This is very dangerous, use with care.)
        # """
        void SetReferenceCount(int);


###

# point_cloud_handlers.h
# template <typename PointT>
# class VTKCOMMONCORE_EXPORT vtkSmartPointerBase
cdef extern from "vtkSmartPointerBase.h" nogil:
    cdef cppclass vtkSmartPointerBase:
        vtkSmartPointerBase()
        # vtkSmartPointerBase(vtkObjectBase* r);
        # vtkSmartPointerBase(const vtkSmartPointerBase& r);
        # ~vtkSmartPointerBase();

        # vtkSmartPointerBase& operator=(vtkObjectBase* r);
        # vtkSmartPointerBase& operator=(const vtkSmartPointerBase& r);

        vtkObjectBase* GetPointer()
        # void Report(vtkGarbageCollector* collector, const char* desc);


###


# //----------------------------------------------------------------------------
# #define VTK_SMART_POINTER_BASE_DEFINE_OPERATOR(op) \
#   inline bool \
#   operator op (const vtkSmartPointerBase& l, const vtkSmartPointerBase& r) \
#   { \
#     return (static_cast<void*>(l.GetPointer()) op \
#             static_cast<void*>(r.GetPointer())); \
#   } \
#   inline bool \
#   operator op (vtkObjectBase* l, const vtkSmartPointerBase& r) \
#   { \
#     return (static_cast<void*>(l) op static_cast<void*>(r.GetPointer())); \
#   } \
#   inline bool \
#   operator op (const vtkSmartPointerBase& l, vtkObjectBase* r) \
#   { \
#     return (static_cast<void*>(l.GetPointer()) op static_cast<void*>(r)); \
#   }
# /**
#  * Compare smart pointer values.
#  */
# VTK_SMART_POINTER_BASE_DEFINE_OPERATOR(==)
# VTK_SMART_POINTER_BASE_DEFINE_OPERATOR(!=)
# VTK_SMART_POINTER_BASE_DEFINE_OPERATOR(<)
# VTK_SMART_POINTER_BASE_DEFINE_OPERATOR(<=)
# VTK_SMART_POINTER_BASE_DEFINE_OPERATOR(>)
# VTK_SMART_POINTER_BASE_DEFINE_OPERATOR(>=)
# 
# #undef VTK_SMART_POINTER_BASE_DEFINE_OPERATOR
# 
# /**
#  * Streaming operator to print smart pointer like regular pointers.
#  */
# VTKCOMMONCORE_EXPORT ostream& operator << (ostream& os,
#                                         const vtkSmartPointerBase& p);
# 
# // VTK-HeaderTest-Exclude: vtkSmartPointerBase.h
# 
###


# template <class T>
# class vtkSmartPointer: public vtkSmartPointerBase
cdef extern from "vtkSmartPointer.h" nogil:
    cdef cppclass vtkSmartPointer[T](vtkSmartPointerBase):
        # static T* CheckType(T* t) { return t; }
        vtkSmartPointer()
        # vtkSmartPointer(T* r): vtkSmartPointerBase(r) {}

        # template <class U>
        # vtkSmartPointer(const vtkSmartPointer<U>& r):
        #     vtkSmartPointerBase(CheckType(r.GetPointer())) {}

        # vtkSmartPointer& operator=(T* r)

        # template <class U>
        # vtkSmartPointer& operator=(const vtkSmartPointer<U>& r)

        # T* GetPointer() const
        # T* Get() const
        # operator T* () const
        # T& operator*() const
        # T* operator->() const
        # void TakeReference(T* t)
        # static vtkSmartPointer<T> New()
        # static vtkSmartPointer<T> NewInstance(T* t)
        # static vtkSmartPointer<T> Take(T* t)


# public:
# VTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(==)
# VTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(!=)
# VTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(<)
# VTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(<=)
# VTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(>)
# VTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(>=)
# # undef VTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND
# 
# #define VTK_SMART_POINTER_DEFINE_OPERATOR(op) \
# template <class T> \
# inline bool \
#
# template <class T> \
# inline bool operator op (T* l, const vtkSmartPointer<T>& r) \
# 
# template <class T> \
# inline bool operator op (const vtkSmartPointer<T>& l, T* r) \
# 
# # Compare smart pointer values.
# VTK_SMART_POINTER_DEFINE_OPERATOR(==)
# VTK_SMART_POINTER_DEFINE_OPERATOR(!=)
# VTK_SMART_POINTER_DEFINE_OPERATOR(<)
# VTK_SMART_POINTER_DEFINE_OPERATOR(<=)
# VTK_SMART_POINTER_DEFINE_OPERATOR(>)
# VTK_SMART_POINTER_DEFINE_OPERATOR(>=)
# # Streaming operator to print smart pointer like regular pointers.
# template <class T>
# inline ostream& operator << (ostream& os, const vtkSmartPointer<T>& p)
# VTK-HeaderTest-Exclude: vtkSmartPointer.h
###


# class vtkRenderer : public vtkViewport
cdef extern from "vtkRenderer.h" nogil:
    # cdef cppclass vtkRenderer(vtkViewport)
    cdef cppclass vtkRenderer:
        # vtkTypeMacro(vtkRenderer,vtkViewport);
        # void PrintSelf(ostream& os, vtkIndent indent) VTK_OVERRIDE;
        # static vtkRenderer *New();
        vtkRenderer()

        # void AddActor(vtkProp *p);
        # void AddVolume(vtkProp *p);
        # void RemoveActor(vtkProp *p);
        # void RemoveVolume(vtkProp *p);
        # void AddLight(vtkLight *);
        # void RemoveLight(vtkLight *);
        # void RemoveAllLights();
        # vtkLightCollection *GetLights();
        # void SetLightCollection(vtkLightCollection *lights);
        # void CreateLight(void);
        # virtual vtkLight *MakeLight();
        # vtkGetMacro(TwoSidedLighting,int);
        # vtkSetMacro(TwoSidedLighting,int);
        # vtkBooleanMacro(TwoSidedLighting,int);
        # vtkSetMacro(LightFollowCamera,int);
        # vtkGetMacro(LightFollowCamera,int);
        # vtkBooleanMacro(LightFollowCamera,int);
        # vtkGetMacro(AutomaticLightCreation,int);
        # vtkSetMacro(AutomaticLightCreation,int);
        # vtkBooleanMacro(AutomaticLightCreation,int);
        # virtual int UpdateLightsGeometryToFollowCamera(void);
        # vtkVolumeCollection *GetVolumes();
        # vtkActorCollection *GetActors();
        # void SetActiveCamera(vtkCamera *);
        # vtkCamera *GetActiveCamera();
        # virtual vtkCamera *MakeCamera();
        # vtkSetMacro(Erase, int);
        # vtkGetMacro(Erase, int);
        # vtkBooleanMacro(Erase, int);
        # vtkSetMacro(Draw, int);
        # vtkGetMacro(Draw, int);
        # vtkBooleanMacro(Draw, int);
        # int CaptureGL2PSSpecialProp(vtkProp *);
        # void SetGL2PSSpecialPropCollection(vtkPropCollection *);
        # void AddCuller(vtkCuller *);
        # void RemoveCuller(vtkCuller *);
        # vtkCullerCollection *GetCullers();
        # vtkSetVector3Macro(Ambient,double);
        # vtkGetVectorMacro(Ambient,double,3);
        # vtkSetMacro(AllocatedRenderTime,double);
        # virtual double GetAllocatedRenderTime();
        # virtual double GetTimeFactor();
        # virtual void Render();
        # virtual void DeviceRender() =0;
        # virtual void DeviceRenderOpaqueGeometry();
        # virtual void DeviceRenderTranslucentPolygonalGeometry();
        # virtual void ClearLights(void) {};
        # virtual void Clear() {}
        # int VisibleActorCount();
        # int VisibleVolumeCount();
        # void ComputeVisiblePropBounds( double bounds[6] );
        # double *ComputeVisiblePropBounds();
        # virtual void ResetCameraClippingRange();
        # virtual void ResetCameraClippingRange( double bounds[6] );
        # virtual void ResetCameraClippingRange( double xmin, double xmax,
        #                                        double ymin, double ymax,
        #                                        double zmin, double zmax);
        # vtkSetClampMacro(NearClippingPlaneTolerance,double,0,0.99);
        # vtkGetMacro(NearClippingPlaneTolerance,double);
        # vtkSetClampMacro(ClippingRangeExpansion,double,0,0.99);
        # vtkGetMacro(ClippingRangeExpansion,double);
        # virtual void ResetCamera();
        # virtual void ResetCamera(double bounds[6]);
        # virtual void ResetCamera(double xmin, double xmax, double ymin, double ymax,
        #                          double zmin, double zmax);
        # void SetRenderWindow(vtkRenderWindow *);
        # vtkRenderWindow *GetRenderWindow() {return this->RenderWindow;};
        # vtkWindow *GetVTKWindow() VTK_OVERRIDE;
        # vtkSetMacro(BackingStore,int);
        # vtkGetMacro(BackingStore,int);
        # vtkBooleanMacro(BackingStore,int);
        # vtkSetMacro(Interactive,int);
        # vtkGetMacro(Interactive,int);
        # vtkBooleanMacro(Interactive,int);
        # virtual void SetLayer(int layer);
        # vtkGetMacro(Layer, int);
        # vtkGetMacro(PreserveColorBuffer, int);
        # vtkSetMacro(PreserveColorBuffer, int);
        # vtkBooleanMacro(PreserveColorBuffer, int);
        # vtkSetMacro(PreserveDepthBuffer, int);
        # vtkGetMacro(PreserveDepthBuffer, int);
        # vtkBooleanMacro(PreserveDepthBuffer, int);
        # int  Transparent();
        # void WorldToView() VTK_OVERRIDE;
        # void ViewToWorld() VTK_OVERRIDE;
        # void ViewToWorld(double &wx, double &wy, double &wz) VTK_OVERRIDE;
        # void WorldToView(double &wx, double &wy, double &wz) VTK_OVERRIDE;
        # double GetZ (int x, int y);
        # vtkMTimeType GetMTime() VTK_OVERRIDE;
        # vtkGetMacro( LastRenderTimeInSeconds, double );
        # vtkGetMacro( NumberOfPropsRendered, int );
        # vtkAssemblyPath* PickProp(double selectionX, double selectionY) VTK_OVERRIDE
        # vtkAssemblyPath* PickProp(double selectionX1, double selectionY1,
        #                           double selectionX2, double selectionY2) VTK_OVERRIDE;
        # virtual void StereoMidpoint() { return; };
        # double GetTiledAspectRatio();
        # int IsActiveCameraCreated()
        # vtkSetMacro(UseDepthPeeling,int);
        # vtkGetMacro(UseDepthPeeling,int);
        # vtkBooleanMacro(UseDepthPeeling,int);
        # vtkSetMacro(UseDepthPeelingForVolumes, bool)
        # vtkGetMacro(UseDepthPeelingForVolumes, bool)
        # vtkBooleanMacro(UseDepthPeelingForVolumes, bool)
        # vtkSetClampMacro(OcclusionRatio,double,0.0,0.5);
        # vtkGetMacro(OcclusionRatio,double);
        # vtkSetMacro(MaximumNumberOfPeels,int);
        # vtkGetMacro(MaximumNumberOfPeels,int);
        # vtkGetMacro(LastRenderingUsedDepthPeeling,int);
        # void SetDelegate(vtkRendererDelegate *d);
        # vtkGetObjectMacro(Delegate,vtkRendererDelegate);
        # vtkGetObjectMacro(Selector, vtkHardwareSelector);
        # virtual void SetBackgroundTexture(vtkTexture*);
        # vtkGetObjectMacro(BackgroundTexture, vtkTexture);
        # vtkSetMacro(TexturedBackground,bool);
        # vtkGetMacro(TexturedBackground,bool);
        # vtkBooleanMacro(TexturedBackground,bool);
        # virtual void ReleaseGraphicsResources(vtkWindow *);
        # vtkSetMacro(UseFXAA, bool)
        # vtkGetMacro(UseFXAA, bool)
        # vtkBooleanMacro(UseFXAA, bool)
        # vtkGetObjectMacro(FXAAOptions, vtkFXAAOptions)
        # virtual void SetFXAAOptions(vtkFXAAOptions*);
        # vtkSetMacro(UseShadows,int);
        # vtkGetMacro(UseShadows,int);
        # vtkBooleanMacro(UseShadows,int);
        # vtkSetMacro(UseHiddenLineRemoval, int)
        # vtkGetMacro(UseHiddenLineRemoval, int)
        # vtkBooleanMacro(UseHiddenLineRemoval, int)
        # void SetPass(vtkRenderPass *p);
        # vtkGetObjectMacro(Pass, vtkRenderPass);
        # vtkGetObjectMacro(Information, vtkInformation);
        # virtual void SetInformation(vtkInformation*);


# inline vtkLightCollection *vtkRenderer::GetLights()
#
# # Get the list of cullers for this renderer.
# inline vtkCullerCollection *vtkRenderer::GetCullers(){return this->Cullers;}
#
###

# "vtkRenderer.h"
# lets define the different types of stereo
# #define VTK_STEREO_CRYSTAL_EYES 1
# #define VTK_STEREO_RED_BLUE     2
# #define VTK_STEREO_INTERLACED   3
# #define VTK_STEREO_LEFT         4
# #define VTK_STEREO_RIGHT        5
# #define VTK_STEREO_DRESDEN      6
# #define VTK_STEREO_ANAGLYPH     7
# #define VTK_STEREO_CHECKERBOARD 8
# #define VTK_STEREO_SPLITVIEWPORT_HORIZONTAL 9
# #define VTK_STEREO_FAKE 10
# 
# #define VTK_CURSOR_DEFAULT   0
# #define VTK_CURSOR_ARROW     1
# #define VTK_CURSOR_SIZENE    2
# #define VTK_CURSOR_SIZENW    3
# #define VTK_CURSOR_SIZESW    4
# #define VTK_CURSOR_SIZESE    5
# #define VTK_CURSOR_SIZENS    6
# #define VTK_CURSOR_SIZEWE    7
# #define VTK_CURSOR_SIZEALL   8
# #define VTK_CURSOR_HAND      9
# #define VTK_CURSOR_CROSSHAIR 10

# class vtkRenderer : public vtkViewport
cdef extern from "vtkRenderWindow.h" nogil:
    # cdef cppclass vtkRenderWindow(vtkWindow):
    cdef cppclass vtkRenderWindow:
        vtkRenderWindow()
        # vtkTypeMacro(vtkRenderWindow,vtkWindow);
        # void PrintSelf(ostream& os, vtkIndent indent) VTK_OVERRIDE;
        # * Construct an instance of  vtkRenderWindow with its screen size
        # * set to 300x300, borders turned on, positioned at (0,0), double
        # * buffering turned on.
        # static vtkRenderWindow *New();
        # * Add a renderer to the list of renderers.
        # virtual void AddRenderer(vtkRenderer *);
        # 
        # /**
        # * Remove a renderer from the list of renderers.
        # */
        # void RemoveRenderer(vtkRenderer *);
        # 
        # /**
        # * Query if a renderer is in the list of renderers.
        # */
        # int HasRenderer(vtkRenderer *);
        # 
        # /**
        # * What rendering library has the user requested
        # */
        # static const char *GetRenderLibrary();
        # 
        # /**
        # * What rendering backend has the user requested
        # */
        # virtual const char *GetRenderingBackend();
        # 
        # /**
        # * Return the collection of renderers in the render window.
        # */
        # vtkRendererCollection *GetRenderers() {return this->Renderers;};
        # 
        # /**
        # * The GL2PS exporter must handle certain props in a special way (e.g. text).
        # * This method performs a render and captures all "GL2PS-special" props in
        # * the specified collection. The collection will contain a
        # * vtkPropCollection for each vtkRenderer in this->GetRenderers(), each
        # * containing the special props rendered by the corresponding renderer.
        # */
        # void CaptureGL2PSSpecialProps(vtkCollection *specialProps);
        # 
        # //@{
        # /**
        # * Returns true if the render process is capturing text actors.
        # */
        # vtkGetMacro(CapturingGL2PSSpecialProps, int);
        # //@}
        # 
        # /**
        # * Ask each renderer owned by this RenderWindow to render its image and
        # * synchronize this process.
        # */
        # void Render() VTK_OVERRIDE;
        # 
        # /**
        # * Initialize the rendering process.
        # */
        # virtual void Start() = 0;
        # 
        # /**
        # * Finalize the rendering process.
        # */
        # virtual void Finalize() = 0;
        # 
        # /**
        # * A termination method performed at the end of the rendering process
        # * to do things like swapping buffers (if necessary) or similar actions.
        # */
        # virtual void Frame() = 0;
        # 
        # /**
        # * Block the thread until the actual rendering is finished().
        # * Useful for measurement only.
        # */
        # virtual void WaitForCompletion()=0;
        # 
        # /**
        # * Performed at the end of the rendering process to generate image.
        # * This is typically done right before swapping buffers.
        # */
        # virtual void CopyResultFrame();
        # 
        # /**
        # * Create an interactor to control renderers in this window. We need
        # * to know what type of interactor to create, because we might be in
        # * X Windows or MS Windows.
        # */
        # virtual vtkRenderWindowInteractor *MakeRenderWindowInteractor();
        # 
        # //@{
        # /**
        # * Hide or Show the mouse cursor, it is nice to be able to hide the
        # * default cursor if you want VTK to display a 3D cursor instead.
        # * Set cursor position in window (note that (0,0) is the lower left
        # * corner).
        # */
        # virtual void HideCursor() = 0;
        # virtual void ShowCursor() = 0;
        # virtual void SetCursorPosition(int , int ) {}
        # //@}
        # 
        # //@{
        # /**
        # * Change the shape of the cursor.
        # */
        # vtkSetMacro(CurrentCursor,int);
        # vtkGetMacro(CurrentCursor,int);
        # //@}
        # 
        # //@{
        # /**
        # * Turn on/off rendering full screen window size.
        # */
        # virtual void SetFullScreen(int) = 0;
        # vtkGetMacro(FullScreen,int);
        # vtkBooleanMacro(FullScreen,int);
        # //@}
        # 
        # //@{
        # /**
        # * Turn on/off window manager borders. Typically, you shouldn't turn the
        # * borders off, because that bypasses the window manager and can cause
        # * undesirable behavior.
        # */
        # vtkSetMacro(Borders,int);
        # vtkGetMacro(Borders,int);
        # vtkBooleanMacro(Borders,int);
        # //@}
        # 
        # //@{
        # /**
        # * Prescribe that the window be created in a stereo-capable mode. This
        # * method must be called before the window is realized. Default is off.
        # */
        # vtkGetMacro(StereoCapableWindow,int);
        # vtkBooleanMacro(StereoCapableWindow,int);
        # virtual void SetStereoCapableWindow(int capable);
        # //@}
        # 
        # //@{
        # /**
        # * Turn on/off stereo rendering.
        # */
        # vtkGetMacro(StereoRender,int);
        # void SetStereoRender(int stereo);
        # vtkBooleanMacro(StereoRender,int);
        # //@}
        # 
        # //@{
        # /**
        # * Turn on/off the use of alpha bitplanes.
        # */
        # vtkSetMacro(AlphaBitPlanes, int);
        # vtkGetMacro(AlphaBitPlanes, int);
        # vtkBooleanMacro(AlphaBitPlanes, int);
        # //@}
        # 
        # //@{
        # /**
        # * Turn on/off point smoothing. Default is off.
        # * This must be applied before the first Render.
        # */
        # vtkSetMacro(PointSmoothing,int);
        # vtkGetMacro(PointSmoothing,int);
        # vtkBooleanMacro(PointSmoothing,int);
        # //@}
        # 
        # //@{
        # /**
        # * Turn on/off line smoothing. Default is off.
        # * This must be applied before the first Render.
        # */
        # vtkSetMacro(LineSmoothing,int);
        # vtkGetMacro(LineSmoothing,int);
        # vtkBooleanMacro(LineSmoothing,int);
        # //@}
        # 
        # //@{
        # /**
        # * Turn on/off polygon smoothing. Default is off.
        # * This must be applied before the first Render.
        # */
        # vtkSetMacro(PolygonSmoothing,int);
        # vtkGetMacro(PolygonSmoothing,int);
        # vtkBooleanMacro(PolygonSmoothing,int);
        # //@}
        # 
        # //@{
        # /**
        # * Set/Get what type of stereo rendering to use.  CrystalEyes
        # * mode uses frame-sequential capabilities available in OpenGL
        # * to drive LCD shutter glasses and stereo projectors.  RedBlue
        # * mode is a simple type of stereo for use with red-blue glasses.
        # * Anaglyph mode is a superset of RedBlue mode, but the color
        # * output channels can be configured using the AnaglyphColorMask
        # * and the color of the original image can be (somewhat) maintained
        # * using AnaglyphColorSaturation;  the default colors for Anaglyph
        # * mode is red-cyan.  Interlaced stereo mode produces a composite
        # * image where horizontal lines alternate between left and right
        # * views.  StereoLeft and StereoRight modes choose one or the other
        # * stereo view.  Dresden mode is yet another stereoscopic
        # * interleaving. Fake simply causes the window to render twice without
        # * actually swapping the camera from left eye to right eye. This is useful in
        # * certain applications that want to emulate the rendering passes without
        # * actually rendering in stereo mode.
        # */
        # vtkGetMacro(StereoType,int);
        # vtkSetMacro(StereoType,int);
        # void SetStereoTypeToCrystalEyes()
        # {this->SetStereoType(VTK_STEREO_CRYSTAL_EYES);}
        # void SetStereoTypeToRedBlue()
        # {this->SetStereoType(VTK_STEREO_RED_BLUE);}
        # void SetStereoTypeToInterlaced()
        # {this->SetStereoType(VTK_STEREO_INTERLACED);}
        # void SetStereoTypeToLeft()
        # {this->SetStereoType(VTK_STEREO_LEFT);}
        # void SetStereoTypeToRight()
        # {this->SetStereoType(VTK_STEREO_RIGHT);}
        # void SetStereoTypeToDresden()
        # {this->SetStereoType(VTK_STEREO_DRESDEN);}
        # void SetStereoTypeToAnaglyph()
        # {this->SetStereoType(VTK_STEREO_ANAGLYPH);}
        # void SetStereoTypeToCheckerboard()
        # {this->SetStereoType(VTK_STEREO_CHECKERBOARD);}
        # void SetStereoTypeToSplitViewportHorizontal()
        # {this->SetStereoType(VTK_STEREO_SPLITVIEWPORT_HORIZONTAL);}
        # void SetStereoTypeToFake()
        # {this->SetStereoType(VTK_STEREO_FAKE);}
        # //@}
        # 
        # const char *GetStereoTypeAsString();
        # 
        # /**
        # * Update the system, if needed, due to stereo rendering. For some stereo
        # * methods, subclasses might need to switch some hardware settings here.
        # */
        # virtual void StereoUpdate();
        # 
        # /**
        # * Intermediate method performs operations required between the rendering
        # * of the left and right eye.
        # */
        # virtual void StereoMidpoint();
        # 
        # /**
        # * Handles work required once both views have been rendered when using
        # * stereo rendering.
        # */
        # virtual void StereoRenderComplete();
        # 
        # //@{
        # /**
        # * Set/get the anaglyph color saturation factor.  This number ranges from
        # * 0.0 to 1.0:  0.0 means that no color from the original object is
        # * maintained, 1.0 means all of the color is maintained.  The default
        # * value is 0.65.  Too much saturation can produce uncomfortable 3D
        # * viewing because anaglyphs also use color to encode 3D.
        # */
        # vtkSetClampMacro(AnaglyphColorSaturation,float, 0.0f, 1.0f);
        # vtkGetMacro(AnaglyphColorSaturation,float);
        # //@}
        # 
        # //@{
        # /**
        # * Set/get the anaglyph color mask values.  These two numbers are bits
        # * mask that control which color channels of the original stereo
        # * images are used to produce the final anaglyph image.  The first
        # * value is the color mask for the left view, the second the mask
        # * for the right view.  If a bit in the mask is on for a particular
        # * color for a view, that color is passed on to the final view; if
        # * it is not set, that channel for that view is ignored.
        # * The bits are arranged as r, g, and b, so r = 4, g = 2, and b = 1.
        # * By default, the first value (the left view) is set to 4, and the
        # * second value is set to 3.  That means that the red output channel
        # * comes from the left view, and the green and blue values come from
        # * the right view.
        # */
        # vtkSetVector2Macro(AnaglyphColorMask,int);
        # vtkGetVectorMacro(AnaglyphColorMask,int,2);
        # //@}
        # 
        # /**
        # * Remap the rendering window. This probably only works on UNIX right now.
        # * It is useful for changing properties that can't normally be changed
        # * once the window is up.
        # */
        # virtual void WindowRemap() = 0;
        # 
        # //@{
        # /**
        # * Turn on/off buffer swapping between images.
        # */
        # vtkSetMacro(SwapBuffers,int);
        # vtkGetMacro(SwapBuffers,int);
        # vtkBooleanMacro(SwapBuffers,int);
        # //@}
        # 
        # //@{
        # /**
        # * Set/Get the pixel data of an image, transmitted as RGBRGBRGB. The
        # * front argument indicates if the front buffer should be used or the back
        # * buffer. It is the caller's responsibility to delete the resulting
        # * array. It is very important to realize that the memory in this array
        # * is organized from the bottom of the window to the top. The origin
        # * of the screen is in the lower left corner. The y axis increases as
        # * you go up the screen. So the storage of pixels is from left to right
        # * and from bottom to top.
        # * (x,y) is any corner of the rectangle. (x2,y2) is its opposite corner on
        # * the diagonal.
        # */
        # virtual int SetPixelData(int x, int y, int x2, int y2, unsigned char *data,
        #                        int front) = 0;
        # virtual int SetPixelData(int x, int y, int x2, int y2,
        #                        vtkUnsignedCharArray *data, int front) = 0;
        # //@}
        # 
        # //@{
        # /**
        # * Same as Get/SetPixelData except that the image also contains an alpha
        # * component. The image is transmitted as RGBARGBARGBA... each of which is a
        # * float value. The "blend" parameter controls whether the SetRGBAPixelData
        # * method blends the data with the previous contents of the frame buffer
        # * or completely replaces the frame buffer data.
        # */
        # virtual float *GetRGBAPixelData(int x, int y, int x2, int y2, int front) = 0;
        # virtual int GetRGBAPixelData(int x, int y, int x2, int y2, int front,
        #                            vtkFloatArray *data) = 0;
        # virtual int SetRGBAPixelData(int x, int y, int x2, int y2, float *,
        #                            int front, int blend=0) = 0;
        # virtual int SetRGBAPixelData(int, int, int, int, vtkFloatArray*,
        #                            int, int blend=0) = 0;
        # virtual void ReleaseRGBAPixelData(float *data)=0;
        # virtual unsigned char *GetRGBACharPixelData(int x, int y, int x2, int y2,
        #                                           int front) = 0;
        # virtual int GetRGBACharPixelData(int x, int y, int x2, int y2, int front,
        #                                vtkUnsignedCharArray *data) = 0;
        # virtual int SetRGBACharPixelData(int x,int y, int x2, int y2,
        #                                unsigned char *data, int front,
        #                                int blend=0) = 0;
        # virtual int SetRGBACharPixelData(int x, int y, int x2, int y2,
        #                                vtkUnsignedCharArray *data, int front,
        #                                int blend=0) = 0;
        # //@}
        # 
        # //@{
        # /**
        # * Set/Get the zbuffer data from the frame buffer.
        # * (x,y) is any corner of the rectangle. (x2,y2) is its opposite corner on
        # * the diagonal.
        # */
        # virtual float *GetZbufferData(int x, int y, int x2, int y2) = 0;
        # virtual int GetZbufferData(int x, int y, int x2, int y2, float *z) = 0;
        # virtual int GetZbufferData(int x, int y, int x2, int y2,
        #                          vtkFloatArray *z) = 0;
        # virtual int SetZbufferData(int x, int y, int x2, int y2, float *z) = 0;
        # virtual int SetZbufferData(int x, int y, int x2, int y2,
        #                          vtkFloatArray *z) = 0;
        # float GetZbufferDataAtPoint(int x, int y)
        # {
        # float value;
        # this->GetZbufferData(x, y, x, y, &value);
        # return value;
        # }
        # //@}
        # 
        # //@{
        # /**
        # * Set the number of frames for doing antialiasing. The default is
        # * zero. Typically five or six will yield reasonable results without
        # * taking too long.
        # */
        # vtkGetMacro(AAFrames,int);
        # vtkSetMacro(AAFrames,int);
        # //@}
        # 
        # //@{
        # /**
        # * Set the number of frames for doing focal depth. The default is zero.
        # * Depending on how your scene is organized you can get away with as
        # * few as four frames for focal depth or you might need thirty.
        # * One thing to note is that if you are using focal depth frames,
        # * then you will not need many (if any) frames for antialiasing.
        # */
        # vtkGetMacro(FDFrames,int);
        # virtual void SetFDFrames (int fdFrames);
        # //@}
        # 
        # //@{
        # /**
        # * Turn on/off using constant offsets for focal depth rendering.
        # * The default is off. When constants offsets are used, re-rendering
        # * the same scene using the same camera yields the same image; otherwise
        # * offsets are random numbers at each rendering that yields
        # * slightly different images.
        # */
        # vtkGetMacro(UseConstantFDOffsets,int);
        # vtkSetMacro(UseConstantFDOffsets,int);
        # //@}
        # 
        # //@{
        # /**
        # * Set the number of sub frames for doing motion blur. The default is zero.
        # * Once this is set greater than one, you will no longer see a new frame
        # * for every Render().  If you set this to five, you will need to do
        # * five Render() invocations before seeing the result. This isn't
        # * very impressive unless something is changing between the Renders.
        # * Changing this value may reset the current subframe count.
        # */
        # vtkGetMacro(SubFrames,int);
        # virtual void SetSubFrames(int subFrames);
        # //@}
        # 
        # //@{
        # /**
        # * This flag is set if the window hasn't rendered since it was created
        # */
        # vtkGetMacro(NeverRendered,int);
        # //@}
        # 
        # //@{
        # /**
        # * This is a flag that can be set to interrupt a rendering that is in
        # * progress.
        # */
        # vtkGetMacro(AbortRender,int);
        # vtkSetMacro(AbortRender,int);
        # vtkGetMacro(InAbortCheck,int);
        # vtkSetMacro(InAbortCheck,int);
        # virtual int CheckAbortStatus();
        # //@}
        # 
        # vtkGetMacro(IsPicking,int);
        # vtkSetMacro(IsPicking,int);
        # vtkBooleanMacro(IsPicking,int);
        # 
        # /**
        # * Check to see if a mouse button has been pressed.  All other events
        # * are ignored by this method.  Ideally, you want to abort the render
        # * on any event which causes the DesiredUpdateRate to switch from
        # * a high-quality rate to a more interactive rate.
        # */
        # virtual int GetEventPending() = 0;
        # 
        # /**
        # * Are we rendering at the moment
        # */
        # virtual int  CheckInRenderStatus() { return this->InRender; }
        # 
        # /**
        # * Clear status (after an exception was thrown for example)
        # */
        # virtual void ClearInRenderStatus() { this->InRender = 0; }
        # 
        # //@{
        # /**
        # * Set/Get the desired update rate. This is used with
        # * the vtkLODActor class. When using level of detail actors you
        # * need to specify what update rate you require. The LODActors then
        # * will pick the correct resolution to meet your desired update rate
        # * in frames per second. A value of zero indicates that they can use
        # * all the time they want to.
        # */
        # virtual void SetDesiredUpdateRate(double);
        # vtkGetMacro(DesiredUpdateRate,double);
        # //@}
        # 
        # //@{
        # /**
        # * Get the number of layers for renderers.  Each renderer should have
        # * its layer set individually.  Some algorithms iterate through all layers,
        # * so it is not wise to set the number of layers to be exorbitantly large
        # * (say bigger than 100).
        # */
        # vtkGetMacro(NumberOfLayers, int);
        # vtkSetClampMacro(NumberOfLayers, int, 1, VTK_INT_MAX);
        # //@}
        # 
        # //@{
        # /**
        # * Get the interactor associated with this render window
        # */
        # vtkGetObjectMacro(Interactor,vtkRenderWindowInteractor);
        # //@}
        # 
        # /**
        # * Set the interactor to the render window
        # */
        # void SetInteractor(vtkRenderWindowInteractor *);
        # 
        # /**
        # * This Method detects loops of RenderWindow<->Interactor,
        # * so objects are freed properly.
        # */
        # void UnRegister(vtkObjectBase *o) VTK_OVERRIDE;
        # 
        # //@{
        # /**
        # * Dummy stubs for vtkWindow API.
        # */
        # void SetDisplayId(void *) VTK_OVERRIDE = 0;
        # void SetWindowId(void *)  VTK_OVERRIDE = 0;
        # virtual void SetNextWindowId(void *) = 0;
        # void SetParentId(void *)  VTK_OVERRIDE = 0;
        # void *GetGenericDisplayId() VTK_OVERRIDE = 0;
        # void *GetGenericWindowId() VTK_OVERRIDE = 0;
        # void *GetGenericParentId() VTK_OVERRIDE = 0;
        # void *GetGenericContext() VTK_OVERRIDE = 0;
        # void *GetGenericDrawable() VTK_OVERRIDE = 0;
        # void SetWindowInfo(char *) VTK_OVERRIDE = 0;
        # virtual void SetNextWindowInfo(char *) = 0;
        # void SetParentInfo(char *) VTK_OVERRIDE = 0;
        # //@}
        # 
        # /**
        # * Initialize the render window from the information associated
        # * with the currently activated OpenGL context.
        # */
        # virtual bool InitializeFromCurrentContext() { return false; };
        # 
        # /**
        # * Attempt to make this window the current graphics context for the calling
        # * thread.
        # */
        # void MakeCurrent() VTK_OVERRIDE = 0;
        # 
        # /**
        # * Tells if this window is the current graphics context for the calling
        # * thread.
        # */
        # virtual bool IsCurrent()=0;
        # 
        # /**
        # * Test if the window has a valid drawable. This is
        # * currently only an issue on Mac OS X Cocoa where rendering
        # * to an invalid drawable results in all OpenGL calls to fail
        # * with "invalid framebuffer operation".
        # */
        # virtual bool IsDrawable(){ return true; }
        # 
        # /**
        # * If called, allow MakeCurrent() to skip cache-check when called.
        # * MakeCurrent() reverts to original behavior of cache-checking
        # * on the next render.
        # */
        # virtual void SetForceMakeCurrent() {}
        # 
        # /**
        # * Get report of capabilities for the render window
        # */
        # virtual const char *ReportCapabilities() { return "Not Implemented";};
        # 
        # /**
        # * Does this render window support OpenGL? 0-false, 1-true
        # */
        # virtual int SupportsOpenGL() { return 0;};
        # 
        # /**
        # * Is this render window using hardware acceleration? 0-false, 1-true
        # */
        # virtual int IsDirect() { return 0;};
        # 
        # /**
        # * This method should be defined by the subclass. How many bits of
        # * precision are there in the zbuffer?
        # */
        # virtual int GetDepthBufferSize() = 0;
        # 
        # /**
        # * Get the size of the color buffer.
        # * Returns 0 if not able to determine otherwise sets R G B and A into buffer.
        # */
        # virtual int GetColorBufferSizes(int *rgba) = 0;
        # 
        # //@{
        # /**
        # * Get the vtkPainterDeviceAdapter which can be used to paint on
        # * this render window.  Note the old OpenGL backend requires this
        # * method.
        # */
        # vtkGetObjectMacro(PainterDeviceAdapter, vtkPainterDeviceAdapter);
        # //@}
        # 
        # //@{
        # /**
        # * Set / Get the number of multisamples to use for hardware antialiasing.
        # */
        # vtkSetMacro(MultiSamples,int);
        # vtkGetMacro(MultiSamples,int);
        # //@}
        # 
        # //@{
        # /**
        # * Set / Get the availability of the stencil buffer.
        # */
        # vtkSetMacro(StencilCapable, int);
        # vtkGetMacro(StencilCapable, int);
        # vtkBooleanMacro(StencilCapable, int);
        # //@}
        # 
        # //@{
        # /**
        # * If there are several graphics card installed on a system,
        # * this index can be used to specify which card you want to render to.
        # * the default is 0. This may not work on all derived render window and
        # * it may need to be set before the first render.
        # */
        # vtkSetMacro(DeviceIndex,int);
        # vtkGetMacro(DeviceIndex,int);
        # //@}
        # /**
        # * Returns the number of devices (graphics cards) on a system.
        # * This may not work on all derived render windows.
        # */
        # virtual int GetNumberOfDevices()
        # {
        # return 0;
        # }
        # 
        # /**
        # * Create and bind offscreen rendering buffers without destroying the current
        # * OpenGL context. This allows to temporary switch to offscreen rendering
        # * (ie. to make a screenshot even if the window is hidden).
        # * Return if the creation was successful (1) or not (0).
        # * Note: This function requires that the device supports OpenGL framebuffer extension.
        # * The function has no effect if OffScreenRendering is ON.
        # */
        # virtual int SetUseOffScreenBuffers(bool) { return 0; }
        # virtual bool GetUseOffScreenBuffers() { return false; }


###

# vtkRenderWindowInteractor.h

# class VTKRENDERINGCORE_EXPORT vtkRenderWindowInteractor : public vtkObject
cdef extern from "vtkRenderWindowInteractor.h" nogil:
    # cdef cppclass vtkRenderWindowInteractor(vtkObject):
    cdef cppclass vtkRenderWindowInteractor:
        vtkRenderWindowInteractor()
        # friend class vtkInteractorEventRecorder;
        # 
        # public:
        # static vtkRenderWindowInteractor *New();
        # vtkTypeMacro(vtkRenderWindowInteractor,vtkObject);
        # void PrintSelf(ostream& os, vtkIndent indent) VTK_OVERRIDE;
        # 
        # //@{
        # /**
        # * Prepare for handling events and set the Enabled flag to true.
        # * This will be called automatically by Start() if the interactor
        # * is not initialized, but it can be called manually if you need
        # * to perform any operations between initialization and the start
        # * of the event loop.
        # */
        # virtual void Initialize();
        # void ReInitialize() {  this->Initialized = 0; this->Enabled = 0;
        #                     this->Initialize(); }
        # //@}
        # 
        # /**
        # * This Method detects loops of RenderWindow-Interactor,
        # * so objects are freed properly.
        # */
        # void UnRegister(vtkObjectBase *o) VTK_OVERRIDE;
        # 
        # /**
        # * Start the event loop. This is provided so that you do not have to
        # * implement your own event loop. You still can use your own
        # * event loop if you want.
        # */
        # virtual void Start();
        # 
        # /**
        # * Enable/Disable interactions.  By default interactors are enabled when
        # * initialized.  Initialize() must be called prior to enabling/disabling
        # * interaction. These methods are used when a window/widget is being
        # * shared by multiple renderers and interactors.  This allows a "modal"
        # * display where one interactor is active when its data is to be displayed
        # * and all other interactors associated with the widget are disabled
        # * when their data is not displayed.
        # */
        # virtual void Enable() { this->Enabled = 1; this->Modified();}
        # virtual void Disable() { this->Enabled = 0; this->Modified();}
        # vtkGetMacro(Enabled, int);
        # 
        # //@{
        # /**
        # * Enable/Disable whether vtkRenderWindowInteractor::Render() calls
        # * this->RenderWindow->Render().
        # */
        # vtkBooleanMacro(EnableRender, bool);
        # vtkSetMacro(EnableRender, bool);
        # vtkGetMacro(EnableRender, bool);
        # //@}
        # 
        # //@{
        # /**
        # * Set/Get the rendering window being controlled by this object.
        # */
        # void SetRenderWindow(vtkRenderWindow *aren);
        # vtkGetObjectMacro(RenderWindow,vtkRenderWindow);
        # //@}
        # 
        # /**
        # * Event loop notification member for window size change.
        # * Window size is measured in pixels.
        # */
        # virtual void UpdateSize(int x,int y);
        # 
        # /**
        # * This class provides two groups of methods for manipulating timers.  The
        # * first group (CreateTimer(timerType) and DestroyTimer()) implicitly use
        # * an internal timer id (and are present for backward compatibility). The
        # * second group (CreateRepeatingTimer(long),CreateOneShotTimer(long),
        # * ResetTimer(int),DestroyTimer(int)) use timer ids so multiple timers can
        # * be independently managed. In the first group, the CreateTimer() method
        # * takes an argument indicating whether the timer is created the first time
        # * (timerType==VTKI_TIMER_FIRST) or whether it is being reset
        # * (timerType==VTKI_TIMER_UPDATE). (In initial implementations of VTK this
        # * was how one shot and repeating timers were managed.) In the second
        # * group, the create methods take a timer duration argument (in
        # * milliseconds) and return a timer id. Thus the ResetTimer(timerId) and
        # * DestroyTimer(timerId) methods take this timer id and operate on the
        # * timer as appropriate. Methods are also available for determining
        # */
        # virtual int CreateTimer(int timerType); //first group, for backward compatibility
        # virtual int DestroyTimer(); //first group, for backward compatibility
        # 
        # /**
        # * Create a repeating timer, with the specified duration (in milliseconds).
        # * \return the timer id.
        # */
        # int CreateRepeatingTimer(unsigned long duration);
        # 
        # /**
        # * Create a one shot timer, with the specified duretion (in milliseconds).
        # * \return the timer id.
        # */
        # int CreateOneShotTimer(unsigned long duration);
        # 
        # /**
        # * Query whether the specified timerId is a one shot timer.
        # * \return 1 if the timer is a one shot timer.
        # */
        # int IsOneShotTimer(int timerId);
        # 
        # /**
        # * Get the duration (in milliseconds) for the specified timerId.
        # */
        # unsigned long GetTimerDuration(int timerId);
        # 
        # /**
        # * Reset the specified timer.
        # */
        # int ResetTimer(int timerId);
        # 
        # /**
        # * Destroy the timer specified by timerId.
        # * \return 1 if the timer was destroyed.
        # */
        # int DestroyTimer(int timerId);
        # 
        # /**
        # * Get the VTK timer ID that corresponds to the supplied platform ID.
        # */
        # virtual int GetVTKTimerId(int platformTimerId);
        # 
        # // Moved into the public section of the class so that classless timer procs
        # // can access these enum members without being "friends"...
        # enum {OneShotTimer=1,RepeatingTimer};
        # 
        # //@{
        # /**
        # * Specify the default timer interval (in milliseconds). (This is used in
        # * conjunction with the timer methods described previously, e.g.,
        # * CreateTimer() uses this value; and CreateRepeatingTimer(duration) and
        # * CreateOneShotTimer(duration) use the default value if the parameter
        # * "duration" is less than or equal to zero.) Care must be taken when
        # * adjusting the timer interval from the default value of 10
        # * milliseconds--it may adversely affect the interactors.
        # */
        # vtkSetClampMacro(TimerDuration,unsigned long,1,100000);
        # vtkGetMacro(TimerDuration,unsigned long);
        # //@}
        # 
        # //@{
        # /**
        # * These methods are used to communicate information about the currently
        # * firing CreateTimerEvent or DestroyTimerEvent. The caller of
        # * CreateTimerEvent sets up TimerEventId, TimerEventType and
        # * TimerEventDuration. The observer of CreateTimerEvent should set up an
        # * appropriate platform specific timer based on those values and set the
        # * TimerEventPlatformId before returning. The caller of DestroyTimerEvent
        # * sets up TimerEventPlatformId. The observer of DestroyTimerEvent should
        # * simply destroy the platform specific timer created by CreateTimerEvent.
        # * See vtkGenericRenderWindowInteractor's InternalCreateTimer and
        # * InternalDestroyTimer for an example.
        # */
        # vtkSetMacro(TimerEventId, int);
        # vtkGetMacro(TimerEventId, int);
        # vtkSetMacro(TimerEventType, int);
        # vtkGetMacro(TimerEventType, int);
        # vtkSetMacro(TimerEventDuration, int);
        # vtkGetMacro(TimerEventDuration, int);
        # vtkSetMacro(TimerEventPlatformId, int);
        # vtkGetMacro(TimerEventPlatformId, int);
        # //@}
        # 
        # /**
        # * This function is called on 'q','e' keypress if exitmethod is not
        # * specified and should be overridden by platform dependent subclasses
        # * to provide a termination procedure if one is required.
        # */
        # virtual void TerminateApp(void) {}
        # 
        # //@{
        # /**
        # * External switching between joystick/trackball/new? modes. Initial value
        # * is a vtkInteractorStyleSwitch object.
        # */
        # virtual void SetInteractorStyle(vtkInteractorObserver *);
        # vtkGetObjectMacro(InteractorStyle,vtkInteractorObserver);
        # //@}
        # 
        # //@{
        # /**
        # * Turn on/off the automatic repositioning of lights as the camera moves.
        # * Default is On.
        # */
        # vtkSetMacro(LightFollowCamera,int);
        # vtkGetMacro(LightFollowCamera,int);
        # vtkBooleanMacro(LightFollowCamera,int);
        # //@}
        # 
        # //@{
        # /**
        # * Set/Get the desired update rate. This is used by vtkLODActor's to tell
        # * them how quickly they need to render.  This update is in effect only
        # * when the camera is being rotated, or zoomed.  When the interactor is
        # * still, the StillUpdateRate is used instead.
        # * The default is 15.
        # */
        # vtkSetClampMacro(DesiredUpdateRate,double,0.0001,VTK_FLOAT_MAX);
        # vtkGetMacro(DesiredUpdateRate,double);
        # //@}
        # 
        # //@{
        # /**
        # * Set/Get the desired update rate when movement has stopped.
        # * For the non-still update rate, see the SetDesiredUpdateRate method.
        # * The default is 0.0001
        # */
        # vtkSetClampMacro(StillUpdateRate,double,0.0001,VTK_FLOAT_MAX);
        # vtkGetMacro(StillUpdateRate,double);
        # //@}
        # 
        # //@{
        # /**
        # * See whether interactor has been initialized yet.
        # * Default is 0.
        # */
        # vtkGetMacro(Initialized,int);
        # //@}
        # 
        # //@{
        # /**
        # * Set/Get the object used to perform pick operations. In order to
        # * pick instances of vtkProp, the picker must be a subclass of
        # * vtkAbstractPropPicker, meaning that it can identify a particular
        # * instance of vtkProp.
        # */
        # virtual void SetPicker(vtkAbstractPicker*);
        # vtkGetObjectMacro(Picker,vtkAbstractPicker);
        # //@}
        # 
        # /**
        # * Create default picker. Used to create one when none is specified.
        # * Default is an instance of vtkPropPicker.
        # */
        # virtual vtkAbstractPropPicker *CreateDefaultPicker();
        # 
        # //@{
        # /**
        # * Set the picking manager.
        # * Set/Get the object used to perform operations through the interactor
        # * By default, a valid but disabled picking manager is instantiated.
        # */
        # virtual void SetPickingManager(vtkPickingManager*);
        # vtkGetObjectMacro(PickingManager,vtkPickingManager);
        # //@}
        # 
        # //@{
        # /**
        # * These methods correspond to the the Exit, User and Pick
        # * callbacks. They allow for the Style to invoke them.
        # */
        # virtual void ExitCallback();
        # virtual void UserCallback();
        # virtual void StartPickCallback();
        # virtual void EndPickCallback();
        # //@}
        # 
        # /**
        # * Get the current position of the mouse.
        # */
        # virtual void GetMousePosition(int *x, int *y) { *x = 0 ; *y = 0; }
        # 
        # //@{
        # /**
        # * Hide or show the mouse cursor, it is nice to be able to hide the
        # * default cursor if you want VTK to display a 3D cursor instead.
        # */
        # void HideCursor();
        # void ShowCursor();
        # //@}
        # 
        # /**
        # * Render the scene. Just pass the render call on to the
        # * associated vtkRenderWindow.
        # */
        # virtual void Render();
        # 
        # //@{
        # /**
        # * Given a position x, move the current camera's focal point to x.
        # * The movement is animated over the number of frames specified in
        # * NumberOfFlyFrames. The LOD desired frame rate is used.
        # */
        # void FlyTo(vtkRenderer *ren, double x, double y, double z);
        # void FlyTo(vtkRenderer *ren, double *x)
        # {this->FlyTo(ren, x[0], x[1], x[2]);}
        # void FlyToImage(vtkRenderer *ren, double x, double y);
        # void FlyToImage(vtkRenderer *ren, double *x)
        # {this->FlyToImage(ren, x[0], x[1]);}
        # //@}
        # 
        # //@{
        # /**
        # * Set the number of frames to fly to when FlyTo is invoked.
        # */
        # vtkSetClampMacro(NumberOfFlyFrames,int,1,VTK_INT_MAX);
        # vtkGetMacro(NumberOfFlyFrames,int);
        # //@}
        # 
        # //@{
        # /**
        # * Set the total Dolly value to use when flying to (FlyTo()) a
        # * specified point. Negative values fly away from the point.
        # */
        # vtkSetMacro(Dolly,double);
        # vtkGetMacro(Dolly,double);
        # //@}
        # 
        # //@{
        # /**
        # * Set/Get information about the current event.
        # * The current x,y position is in the EventPosition, and the previous
        # * event position is in LastEventPosition, updated automatically each
        # * time EventPosition is set using its Set() method. Mouse positions
        # * are measured in pixels.
        # * The other information is about key board input.
        # */
        # vtkGetVector2Macro(EventPosition,int);
        # vtkGetVector2Macro(LastEventPosition,int);
        # vtkSetVector2Macro(LastEventPosition,int);
        # virtual void SetEventPosition(int x, int y)
        # {
        # vtkDebugMacro(<< this->GetClassName() << " (" << this
        #               << "): setting EventPosition to (" << x << "," << y << ")");
        # if (this->EventPosition[0] != x || this->EventPosition[1] != y ||
        #     this->LastEventPosition[0] != x || this->LastEventPosition[1] != y)
        # {
        #   this->LastEventPosition[0] = this->EventPosition[0];
        #   this->LastEventPosition[1] = this->EventPosition[1];
        #   this->EventPosition[0] = x;
        #   this->EventPosition[1] = y;
        #   this->Modified();
        # }
        # }
        # virtual void SetEventPosition(int pos[2])
        # {
        # this->SetEventPosition(pos[0], pos[1]);
        # }
        # virtual void SetEventPositionFlipY(int x, int y)
        # {
        # this->SetEventPosition(x, this->Size[1] - y - 1);
        # }
        # virtual void SetEventPositionFlipY(int pos[2])
        # {
        # this->SetEventPositionFlipY(pos[0], pos[1]);
        # }
        # //@}
        # 
        # virtual int *GetEventPositions(int pointerIndex)
        # {
        # if (pointerIndex >= VTKI_MAX_POINTERS)
        # {
        #   return NULL;
        # }
        # return this->EventPositions[pointerIndex];
        # }
        # virtual int *GetLastEventPositions(int pointerIndex)
        # {
        # if (pointerIndex >= VTKI_MAX_POINTERS)
        # {
        #   return NULL;
        # }
        # return this->LastEventPositions[pointerIndex];
        # }
        # virtual void SetEventPosition(int x, int y, int pointerIndex)
        # {
        # if (pointerIndex < 0 || pointerIndex >= VTKI_MAX_POINTERS)
        # {
        #   return;
        # }
        # if (pointerIndex == 0)
        # {
        #   this->LastEventPosition[0] = this->EventPosition[0];
        #   this->LastEventPosition[1] = this->EventPosition[1];
        #   this->EventPosition[0] = x;
        #   this->EventPosition[1] = y;
        # }
        # vtkDebugMacro(<< this->GetClassName() << " (" << this
        #               << "): setting EventPosition to (" << x << "," << y << ") for pointerIndex number " << pointerIndex);
        # if (this->EventPositions[pointerIndex][0] != x || this->EventPositions[pointerIndex][1] != y ||
        #     this->LastEventPositions[pointerIndex][0] != x || this->LastEventPositions[pointerIndex][1] != y)
        # {
        #   this->LastEventPositions[pointerIndex][0] = this->EventPositions[pointerIndex][0];
        #   this->LastEventPositions[pointerIndex][1] = this->EventPositions[pointerIndex][1];
        #   this->EventPositions[pointerIndex][0] = x;
        #   this->EventPositions[pointerIndex][1] = y;
        #   this->Modified();
        # }
        # }
        # virtual void SetEventPosition(int pos[2], int pointerIndex)
        # {
        # this->SetEventPosition(pos[0], pos[1], pointerIndex);
        # }
        # virtual void SetEventPositionFlipY(int x, int y, int pointerIndex)
        # {
        # this->SetEventPosition(x, this->Size[1] - y - 1, pointerIndex);
        # }
        # virtual void SetEventPositionFlipY(int pos[2], int pointerIndex)
        # {
        # this->SetEventPositionFlipY(pos[0], pos[1], pointerIndex);
        # }
        # 
        # //@{
        # /**
        # * Set/get whether alt modifier key was pressed.
        # */
        # vtkSetMacro(AltKey, int);
        # vtkGetMacro(AltKey, int);
        # //@}
        # 
        # //@{
        # /**
        # * Set/get whether control modifier key was pressed.
        # */
        # vtkSetMacro(ControlKey, int);
        # vtkGetMacro(ControlKey, int);
        # //@}
        # 
        # //@{
        # /**
        # * Set/get whether shift modifier key was pressed.
        # */
        # vtkSetMacro(ShiftKey, int);
        # vtkGetMacro(ShiftKey, int);
        # //@}
        # 
        # //@{
        # /**
        # * Set/get the key code for the key that was pressed.
        # */
        # vtkSetMacro(KeyCode, char);
        # vtkGetMacro(KeyCode, char);
        # //@}
        # 
        # //@{
        # /**
        # * Set/get the repear count for the key or mouse event. This specifies how
        # * many times a key has been pressed.
        # */
        # vtkSetMacro(RepeatCount, int);
        # vtkGetMacro(RepeatCount, int);
        # //@}
        # 
        # //@{
        # /**
        # * Set/get the key symbol for the key that was pressed. This is the key
        # * symbol as defined by the relevant X headers. On X based platforms this
        # * corresponds to the installed X sevrer, whereas on other platforms the
        # * native key codes are translated into a string representation.
        # */
        # vtkSetStringMacro(KeySym);
        # vtkGetStringMacro(KeySym);
        # //@}
        # 
        # //@{
        # /**
        # * Set/get the index of the most recent pointer to have an event
        # */
        # vtkSetMacro(PointerIndex, int);
        # vtkGetMacro(PointerIndex, int);
        # //@}
        # 
        # //@{
        # /**
        # * Set/get the rotation for the gesture in degrees, update LastRotation
        # */
        # void SetRotation(double val);
        # vtkGetMacro(Rotation, double);
        # vtkGetMacro(LastRotation, double);
        # //@}
        # 
        # //@{
        # /**
        # * Set/get the scale for the gesture, updates LastScale
        # */
        # void SetScale(double val);
        # vtkGetMacro(Scale, double);
        # vtkGetMacro(LastScale, double);
        # //@}
        # 
        # //@{
        # /**
        # * Set/get the tranlation for pan/swipe gestures, update LastTranslation
        # */
        # void SetTranslation(double val[2]);
        # vtkGetVector2Macro(Translation, double);
        # vtkGetVector2Macro(LastTranslation, double);
        # //@}
        # 
        # //@{
        # /**
        # * Set all the event information in one call.
        # */
        # void SetEventInformation(int x,
        #                        int y,
        #                        int ctrl,
        #                        int shift,
        #                        char keycode,
        #                        int repeatcount,
        #                        const char* keysym,
        #                        int pointerIndex)
        # {
        #   this->SetEventPosition(x,y,pointerIndex);
        #   this->ControlKey = ctrl;
        #   this->ShiftKey = shift;
        #   this->KeyCode = keycode;
        #   this->RepeatCount = repeatcount;
        #   this->PointerIndex = pointerIndex;
        #   if(keysym)
        #   {
        #     this->SetKeySym(keysym);
        #   }
        #   this->Modified();
        # }
        # void SetEventInformation(int x, int y,
        #                        int ctrl=0, int shift=0,
        #                        char keycode=0,
        #                        int repeatcount=0,
        #                        const char* keysym=0)
        # {
        #   this->SetEventInformation(x,y,ctrl,shift,keycode,repeatcount,keysym,0);
        # }
        # //@}
        # 
        # //@{
        # /**
        # * Calls SetEventInformation, but flips the Y based on the current Size[1]
        # * value (i.e. y = this->Size[1] - y - 1).
        # */
        # void SetEventInformationFlipY(int x, int y,
        #                             int ctrl, int shift,
        #                             char keycode,
        #                             int repeatcount,
        #                             const char* keysym,
        #                             int pointerIndex)
        # {
        #   this->SetEventInformation(x,
        #                             this->Size[1] - y - 1,
        #                             ctrl,
        #                             shift,
        #                             keycode,
        #                             repeatcount,
        #                             keysym,
        #                             pointerIndex);
        # }
        # void SetEventInformationFlipY(int x, int y,
        #                        int ctrl=0, int shift=0,
        #                        char keycode=0,
        #                        int repeatcount=0,
        #                        const char* keysym=0)
        # {
        #   this->SetEventInformationFlipY(x,y,ctrl,shift,keycode,repeatcount,keysym,0);
        # }
        # //@}
        # 
        # //@{
        # /**
        # * Set all the keyboard-related event information in one call.
        # */
        # void SetKeyEventInformation(int ctrl=0,
        #                           int shift=0,
        #                           char keycode=0,
        #                           int repeatcount=0,
        #                           const char* keysym=0)
        # {
        #   this->ControlKey = ctrl;
        #   this->ShiftKey = shift;
        #   this->KeyCode = keycode;
        #   this->RepeatCount = repeatcount;
        #   if(keysym)
        #   {
        #     this->SetKeySym(keysym);
        #   }
        #   this->Modified();
        # }
        # //@}
        # 
        # //@{
        # /**
        # * This methods sets the Size ivar of the interactor without
        # * actually changing the size of the window. Normally
        # * application programmers would use UpdateSize if anything.
        # * This is useful for letting someone else change the size of
        # * the rendering window and just letting the interactor
        # * know about the change.
        # * The current event width/height (if any) is in EventSize
        # * (Expose event, for example).
        # * Window size is measured in pixels.
        # */
        # vtkSetVector2Macro(Size,int);
        # vtkGetVector2Macro(Size,int);
        # vtkSetVector2Macro(EventSize,int);
        # vtkGetVector2Macro(EventSize,int);
        # //@}
        # 
        # /**
        # * When an event occurs, we must determine which Renderer the event
        # * occurred within, since one RenderWindow may contain multiple
        # * renderers.
        # */
        # virtual vtkRenderer *FindPokedRenderer(int,int);
        # 
        # /**
        # * Return the object used to mediate between vtkInteractorObservers
        # * contending for resources. Multiple interactor observers will often
        # * request different resources (e.g., cursor shape); the mediator uses a
        # * strategy to provide the resource based on priority of the observer plus
        # * the particular request (default versus non-default cursor shape).
        # */
        # vtkObserverMediator *GetObserverMediator();
        # 
        # //@{
        # /**
        # * Use a 3DConnexion device. Initial value is false.
        # * If VTK is not build with the TDx option, this is no-op.
        # * If VTK is build with the TDx option, and a device is not connected,
        # * a warning is emitted.
        # * It is must be called before the first Render to be effective, otherwise
        # * it is ignored.
        # */
        # vtkSetMacro(UseTDx,bool);
        # vtkGetMacro(UseTDx,bool);
        # //@}
        # 
        # //@{
        # /**
        # * Fire various events. SetEventInformation should be called just prior
        # * to calling any of these methods. These methods will Invoke the
        # * corresponding vtk event.
        # */
        # virtual void MouseMoveEvent();
        # virtual void RightButtonPressEvent();
        # virtual void RightButtonReleaseEvent();
        # virtual void LeftButtonPressEvent();
        # virtual void LeftButtonReleaseEvent();
        # virtual void MiddleButtonPressEvent();
        # virtual void MiddleButtonReleaseEvent();
        # virtual void MouseWheelForwardEvent();
        # virtual void MouseWheelBackwardEvent();
        # virtual void ExposeEvent();
        # virtual void ConfigureEvent();
        # virtual void EnterEvent();
        # virtual void LeaveEvent();
        # virtual void KeyPressEvent();
        # virtual void KeyReleaseEvent();
        # virtual void CharEvent();
        # virtual void ExitEvent();
        # virtual void FourthButtonPressEvent();
        # virtual void FourthButtonReleaseEvent();
        # virtual void FifthButtonPressEvent();
        # virtual void FifthButtonReleaseEvent();
        # //@}
        # 
        # //@{
        # /**
        # * Fire various gesture based events.  These methods will Invoke the
        # * corresponding vtk event.
        # */
        # virtual void StartPinchEvent();
        # virtual void PinchEvent();
        # virtual void EndPinchEvent();
        # virtual void StartRotateEvent();
        # virtual void RotateEvent();
        # virtual void EndRotateEvent();
        # virtual void StartPanEvent();
        # virtual void PanEvent();
        # virtual void EndPanEvent();
        # virtual void TapEvent();
        # virtual void LongTapEvent();
        # virtual void SwipeEvent();
        # //@}
        # 
        # //@{
        # /**
        # * Convert multitouch events into gestures. When this is on
        # * (its default) multitouch events received by this interactor
        # * will be converted into gestures by VTK. If turned off the
        # * raw multitouch events will be passed down.
        # */
        # vtkSetMacro(RecognizeGestures,bool);
        # vtkGetMacro(RecognizeGestures,bool);
        # //@}
        # 
        # //@{
        # /**
        # * When handling gestures you can query this value to
        # * determine how many pointers are down for the gesture
        # * this is useful for pan gestures for example
        # */
        # vtkGetMacro(PointersDownCount,int);
        # //@}
        # 
        # //@{
        # /**
        # * Most multitouch systems use persistent contact/pointer ids to
        # * track events/motion during multitouch events. We keep an array
        # * that maps these system dependent contact ids to our pointer index
        # * These functions return -1 if the ID is not found or if there
        # * is no more room for contacts
        # */
        # void ClearContact(size_t contactID);
        # int GetPointerIndexForContact(size_t contactID);
        # int GetPointerIndexForExistingContact(size_t contactID);
        # bool IsPointerIndexSet(int i);
        # void ClearPointerIndex(int i);
        # //@}


###