#ifndef _VTKINTERACTERWRAPPER_H_
#define _VTKINTERACTERWRAPPER_H_

// Point Cloud Library visualization
#include <pcl/visualization/pcl_visualizer.h>

// Visualization Toolkit (VTK)
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindowInteractor.h>
// vtk+qt
// #include <QVTKRenderWindowInteractor.h>

// VTK + PythonCore
// conda vtk only?
#include "PyVTKObject.h"

// CPython(PyObject*)
#include "Python.h"

// void wrapped_from_pclvis_to_vtk(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, QVTKRenderWindowInteractor& qvtkWidget, const std::string& id = "PCL Viewer", bool isVisible=false);
void wrapped_from_pclvis_to_vtk(pcl::visualization::PCLVisualizer* viewer, vtkSmartPointer<vtkRenderWindowInteractor>& qvtkWidget, const std::string& id = "PCL Viewer", bool isVisible=false);
// void wrapped_from_pclvis_to_vtk2(PyObject* viewer, PyObject* qvtkWidget, const std::string& id = "PCL Viewer", bool isVisible=false);
// void wrapped_from_pclvis_to_vtk2(pcl::visualization::PCLVisualizer* viewer, PyObject* qvtkWidget, const std::string& id = "PCL Viewer", bool isVisible=false);
void wrapped_from_pclvis_to_vtk2(pcl::visualization::PCLVisualizer* viewer, PyObject* qvtkWidget);
PyObject* wrapped_from_pclvis_to_vtk3(pcl::visualization::PCLVisualizer* viewer);

#endif // _VTKINTERACTERWRAPPER_H_

