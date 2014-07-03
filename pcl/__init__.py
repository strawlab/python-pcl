# XXX do a more specific import!
from ._pcl import *


def load(path, format=None):
    """Load pointcloud from path.

    Currently supports PCD and PLY files.

    Format should be "pcd", "ply", or None to infer from the pathname.
    """

    if format is None:
        for candidate in ["pcd", "ply"]:
            if path.endswith("." + candidate):
                format = candidate
                break
        else:
            raise ValueError("Could not determine file format from pathname %s"
                             % path)

    p = PointCloud()
    try:
        loader = getattr(p, "_from_%s_file" % format.lower())
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if loader(path):
        raise IOError("error while loading pointcloud from %r (format=%r)"
                      % (path, format))
    return p
