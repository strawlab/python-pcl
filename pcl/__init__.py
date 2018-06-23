# XXX do a more specific import!

from ctypes.util import find_library

from ._pcl import *
# vtkSmartPointer.h error (Linux)
# from .pcl_visualization import *
# from .pcl_grabber import *


import sys


def load(path, format=None):
    """Load pointcloud from path.

    Currently supports PCD and PLY files.
    Format should be "pcd", "ply", "obj", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    p = PointCloud()
    try:
        loader = getattr(p, "_from_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if loader(_encode(path)):
        raise IOError("error while loading pointcloud from %r (format=%r)"
                      % (path, format))
    return p


def load_XYZI(path, format=None):
    """Load pointcloud from path.
    Currently supports PCD and PLY files.
    Format should be "pcd", "ply", "obj", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    p = PointCloud_PointXYZI()
    try:
        loader = getattr(p, "_from_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if loader(_encode(path)):
        raise IOError("error while loading pointcloud from %r (format=%r)"
                      % (path, format))
    return p


def load_XYZRGB(path, format=None):
    """
    Load pointcloud from path.
    Currently supports PCD and PLY files.
    Format should be "pcd", "ply", "obj", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    p = PointCloud_PointXYZRGB()
    try:
        loader = getattr(p, "_from_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if loader(_encode(path)):
        raise IOError("error while loading pointcloud from %r (format=%r)"
                      % (path, format))
    return p


def load_XYZRGBA(path, format=None):
    """
    Load pointcloud from path.
    Currently supports PCD and PLY files.
    Format should be "pcd", "ply", "obj", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    p = PointCloud_PointXYZRGBA()
    try:
        loader = getattr(p, "_from_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if loader(_encode(path)):
        raise IOError("error while loading pointcloud from %r (format=%r)"
                      % (path, format))
    return p


def load_PointWithViewpoint(path, format=None):
    """
    Load pointcloud from path.
    Currently supports PCD and PLY files.
    Format should be "pcd", "ply", "obj", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    p = PointCloud_PointWithViewpoint()
    try:
        loader = getattr(p, "_from_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if loader(_encode(path)):
        raise IOError("error while loading pointcloud from %r (format=%r)"
                      % (path, format))
    return p


def save(cloud, path, format=None, binary=False):
    """Save pointcloud to file.
    Format should be "pcd", "ply", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    try:
        dumper = getattr(cloud, "_to_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if dumper(_encode(path), binary):
        raise IOError("error while saving pointcloud to %r (format=%r)"
                      % (path, format))


def save_XYZRGBA(cloud, path, format=None, binary=False):
    """Save pointcloud to file.
    Format should be "pcd", "ply", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    try:
        dumper = getattr(cloud, "_to_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if dumper(_encode(path), binary):
        raise IOError("error while saving pointcloud to %r (format=%r)"
                      % (path, format))


def save_PointNormal(cloud, path, format=None, binary=False):
    """
    Save pointcloud to file.
    Format should be "pcd", "ply", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    try:
        dumper = getattr(cloud, "_to_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if dumper(_encode(path), binary):
        raise IOError("error while saving pointcloud to %r (format=%r)"
                      % (path, format))


def _encode(path):
    # Encode path for use in C++.
    if isinstance(path, bytes):
        return path
    else:
        return path.encode(sys.getfilesystemencoding())


def _infer_format(path, format):
    if format is not None:
        return format.lower()

    for candidate in ["pcd", "ply", "obj"]:
        if path.endswith("." + candidate):
            return candidate

    raise ValueError("Could not determine file format from pathname %s" % path)
