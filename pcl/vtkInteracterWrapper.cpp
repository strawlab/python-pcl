#include "vtkInteracterWrapper.h"

// Set up the QVTK window.
// void wrapped_from_pclvis_to_vtk(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, QVTKRenderWindowInteractor& qvtkWidget, const std::string& id = "PCL Viewer", bool isVisible=false)
void wrapped_from_pclvis_to_vtk(pcl::visualization::PCLVisualizer* viewer, vtkSmartPointer<vtkRenderWindowInteractor>& qvtkWidget, const std::string& id, bool isVisible)
{
    // Set up the QVTK window. 
    // viewer.reset(new pcl::visualization::PCLVisualizer(id, isVisible));
    qvtkWidget->SetRenderWindow(viewer->getRenderWindow());
    // qvtkWidget->Initialize();
    // qvtkWidget->Start();
    // old vtk?
    // viewer->setupInteractor(qvtkWidget->GetInteractor(), qvtkWidget->GetRenderWindow());
    // qvtkWidget->update(); 
}

// void wrapped_from_pclvis_to_vtk2(PyObject* viewer, PyObject* qvtkWidget, const std::string& id = "PCL Viewer", bool isVisible=false)
void wrapped_from_pclvis_to_vtk2(pcl::visualization::PCLVisualizer* viewer, PyObject* qvtkWidget)
{
    // PyObject* Convert C++ Pointer objects
    // pcl::visualization::PCLVisualizer* tmp_viewer = viewer;
    // vtkSmartPointer<vtkRenderWindowInteractor> tmp_qvtkWidget = PyVTKObject_GetObject(qvtkWidget);
    // vtkSmartPointer<vtkRenderWindowInteractor> tmp_qvtkWidget = (vtkRenderWindowInteractor*)PyVTKObject_GetObject(qvtkWidget);
    printf("0.\n");
    if (qvtkWidget == NULL) { printf("Error : qvtkWidget is NULL.\n"); return ; }
    vtkObjectBase* vtkBasePointer = PyVTKObject_GetObject(qvtkWidget);
    if (vtkBasePointer == NULL) { printf("Error : vtkObjectBase class is NULL.\n"); return ; }
    // vtkRenderWindowInteractor* tmp_qvtkWidget = (vtkRenderWindowInteractor*)vtkBasePointer;
    vtkRenderWindow* tmp_qvtkWidget = (vtkRenderWindow*)vtkBasePointer;
    // cast error
    // vtkSmartPointer<vtkRenderWindow> tmp_qvtkWidget = (vtkSmartPointer<vtkRenderWindow>)vtkBasePointer;
    // if (tmp_qvtkWidget == NULL) { printf("Error : from vtkObjectBase to vtkRenderWindowInteractor is not cast.\n"); return ; }
    if (tmp_qvtkWidget == NULL) { printf("Error : from vtkObjectBase to vtkRenderWindowInteractor is not cast.\n"); return ; }
    // Set up the QVTK window. 
    printf("1.\n");
    // viewer.reset(new pcl::visualization::PCLVisualizer(id, isVisible));
    // viewer->reset(new pcl::visualization::PCLVisualizer("PCL Viewer", false));
    // vtkRenderWindowInteractor
    vtkRendererCollection* aa = viewer->getRenderWindow()->GetRenderers();
    if(aa == NULL) { printf("Error : aa is NULL.\n"); return ; }
    // vtkRenderer* pp = aa->GetNextItem();
    vtkRenderer* pp = aa->GetFirstRenderer();
    if(pp == NULL) { printf("Error : pp is NULL.\n"); return ; }
    tmp_qvtkWidget->AddRenderer(pp);
    // vtkRenderWindowInteractor
    // tmp_qvtkWidget->SetRenderWindow(viewer->getRenderWindow());
    // vtkRenderWindow
    // cast error
    // tmp_qvtkWidget->AddRenderer(viewer->getRenderWindow());
    // tmp_qvtkWidget->Initialize();
    // tmp_qvtkWidget->Start();
    // old vtk?
    // viewer->setupInteractor(tmp_qvtkWidget->GetInteractor(), tmp_qvtkWidget->GetRenderWindow());
    // viewer->setupInteractor(tmp_qvtkWidget->GetRenderWindow());
    // tmp_qvtkWidget->update(); 
    printf("2.\n");
}

// https://github.com/Kitware/VTK/blob/master/Wrapping/PythonCore/PyVTKObject.cxx
// https://postd.cc/python-internals-pyobject/
// http://www.dzeta.jp/~junjis/code_reading/index.php?Python%2Fbuild_class%E5%89%8D%E7%B7%A8%EF%BC%88%E3%81%A8%E3%81%84%E3%81%86%E3%82%88%E3%82%8APyTypeObject%EF%BC%89
PyObject* wrapped_from_pclvis_to_vtk3(pcl::visualization::PCLVisualizer* viewer)
{
    PyObject* retVal = NULL;
    vtkRenderWindow* aa = viewer->getRenderWindow();
    if(aa == NULL) { printf("Error : aa is NULL.\n"); return retVal; }
    // Generate vtk.vtkRenderWindow object
    PyTypeObject *vtkclass = NULL;
    // PyTypeObject bb;
    // bb.---
    // bb.===
    // :
    /*
    static PyTypeObject VtkRenderWindowType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        tp_name      : "vtk.vtkRenderWindow",
        tp_basicsize : (Py_ssize_t) sizeof(RiscvPeObject),
        tp_itemsize  : 0,
        tp_dealloc   : (destructor) RiscvPeDealloc,
        tp_flags     : Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        tp_doc       : "vtk render objects",
        tp_methods   : riscv_chip_methods,
        tp_members   : NULL,
        tp_init      : (initproc) InitRiscvChip,
        tp_new       : MakeRiscvChip
    };
    */
    // PyMethodDef riscv_chip_methods[] = {
    //  { "py_add"     , (PyCFunction)HelloAdd            , METH_VARARGS, "Example ADD"                    },
    //  { "simulate"   , (PyCFunction)SimRiscvChip        , METH_VARARGS, "Simulate RiscvChip"             },
    //  { "load_bin"   , (PyCFunction)LoadBinaryRiscvChip , METH_VARARGS, "Load Binary file"               },
    //  { NULL         , NULL                             ,            0, NULL                             } /* Sentinel */
    // };

    // New Create
    PyObject *pydict = NULL;
    vtkObjectBase* vtkBasePointer = (vtkObjectBase*)aa;
    retVal = PyVTKObject_FromPointer(vtkclass, pydict, vtkBasePointer);
    return retVal;
}