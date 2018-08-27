"""distutils._msvccompiler

Contains MSVCCompiler, an implementation of the abstract CCompiler class
for Microsoft Visual Studio 2015.

The module is compatible with VS 2015 and later. You can find legacy support
for older versions in distutils.msvc9compiler and distutils.msvccompiler.
"""

# Written by Perry Stoll
# hacked by Robin Becker and Thomas Heller to do a better job of
#   finding DevStudio (through the registry)
# ported to VS 2005 and VS 2008 by Christian Heimes
# ported to VS 2015 by Steve Dower

import os
import shutil
import stat
import subprocess

from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
    CompileError, LibError, LinkError
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform

from itertools import count


def _find_vc2015():
    # import winreg
    # Python 2.7
    import _winreg

    try:
        key = _winreg.OpenKeyEx(
            _winreg.HKEY_LOCAL_MACHINE,
            r"Software\Microsoft\VisualStudio\SxS\VC7",
            0,
            _winreg.KEY_READ | _winreg.KEY_WOW64_32KEY
        )
    except OSError:
        log.debug("Visual C++ is not registered")
        return None, None

    best_version = 0
    best_dir = None
    with key:
        for i in count():
            try:
                v, vc_dir, vt = _winreg.EnumValue(key, i)
            except OSError:
                break
            if v and vt == _winreg.REG_SZ and os.path.isdir(vc_dir):
                try:
                    version = int(float(v))
                except (ValueError, TypeError):
                    continue
                if version >= 14 and version > best_version:
                    best_version, best_dir = version, vc_dir
    return best_version, best_dir


def _find_vc2017():
    # import _distutils_findvs
    import threading

    best_version = None   # tuple for full version comparisons
    best_dir = None

    # We need to call findall() on its own thread because it will
    # initialize COM.
    # all_packages = []
    # def _getall():
    #     all_packages.extend(_distutils_findvs.findall())
    # t = threading.Thread(target=_getall)
    # t.start()
    # t.join()

    # for name, version_str, path, packages in all_packages:
    #     if 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' in packages:
    #         vc_dir = os.path.join(path, 'VC', 'Auxiliary', 'Build')
    #         if not os.path.isdir(vc_dir):
    #             continue
    #         try:
    #             version = tuple(int(i) for i in version_str.split('.'))
    #         except (ValueError, TypeError):
    #             continue
    #         if version > best_version:
    #             best_version, best_dir = version, vc_dir
    # try:
    #     best_version = best_version[0]
    # except IndexError:
    #     best_version = None

    # Default Path
    # All Package(Communnity)
    # C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat
    # Console Build Only
    # C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat
    # temporary solution : Default Install Path
    vcvarsall_path = [
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\BuildTools",
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community",
        # Not Check
        # "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Professional",
        # "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Enterprise",
    ]
    for path in vcvarsall_path:
        # path = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\BuildTools"
        vc_dir = os.path.join(path, 'VC', 'Auxiliary', 'Build')
        if os.path.isdir(vc_dir):
            version = 0
            best_version, best_dir = version, vc_dir
            break
    return best_version, best_dir


def _find_vcvarsall(plat_spec):
    best_version, best_dir = _find_vc2017()
    vcruntime = None
    vcruntime_plat = 'x64' if 'amd64' in plat_spec else 'x86'
    if best_version:
        vcredist = os.path.join(best_dir, "..", "..", "Redist", "MSVC", "**",
                                "Microsoft.VC141.CRT", "vcruntime140.dll")
        try:
            import glob
            vcruntime = glob.glob(vcredist, recursive=True)[-1]
        except (ImportError, OSError, LookupError):
            vcruntime = None

    if not best_version:
        best_version, best_dir = _find_vc2015()
        if best_version:
            vcruntime = os.path.join(best_dir, 'redist', vcruntime_plat,
                                     "Microsoft.VC140.CRT", "vcruntime140.dll")

    if not best_version:
        log.debug("No suitable Visual C++ version found")
        return None, None

    vcvarsall = os.path.join(best_dir, "vcvarsall.bat")
    if not os.path.isfile(vcvarsall):
        log.debug("%s cannot be found", vcvarsall)
        return None, None

    if not vcruntime or not os.path.isfile(vcruntime):
        log.debug("%s cannot be found", vcruntime)
        vcruntime = None

    return vcvarsall, vcruntime


def _get_vc_env(plat_spec):
    if os.getenv("DISTUTILS_USE_SDK"):
        return {
            key.lower(): value
            for key, value in os.environ.items()
        }

    vcvarsall, vcruntime = _find_vcvarsall(plat_spec)
    if not vcvarsall:
        raise DistutilsPlatformError("Unable to find vcvarsall.bat")

    try:
        out = subprocess.check_output(
            'cmd /u /c "{}" {} && set'.format(vcvarsall, plat_spec),
            stderr=subprocess.STDOUT,
        ).decode('utf-16le', errors='replace')
    except subprocess.CalledProcessError as exc:
        log.error(exc.output)
        raise DistutilsPlatformError("Error executing {}"
                                     .format(exc.cmd))

    env = {
        key.lower(): value
        for key, _, value in
        (line.partition('=') for line in out.splitlines())
        if key and value
    }

    if vcruntime:
        env['py_vcruntime_redist'] = vcruntime
    return env


def _find_exe(exe, paths=None):
    """Return path to an MSVC executable program.

    Tries to find the program in several places: first, one of the
    MSVC program search paths from the registry; next, the directories
    in the PATH environment variable.  If any of those work, return an
    absolute path that is known to exist.  If none of them work, just
    return the original program name, 'exe'.
    """
    if not paths:
        paths = os.getenv('path').split(os.pathsep)
    for p in paths:
        fn = os.path.join(os.path.abspath(p), exe)
        if os.path.isfile(fn):
            return fn
    return exe


# A map keyed by get_platform() return values to values accepted by
# 'vcvarsall.bat'. Always cross-compile from x86 to work with the
# lighter-weight MSVC installs that do not include native 64-bit tools.
PLAT_TO_VCVARS = {
    'win32': 'x86',
    'win-amd64': 'x86_amd64',
}

# A set containing the DLLs that are guaranteed to be available for
# all micro versions of this Python version. Known extension
# dependencies that are not in this set will be copied to the output
# path.
_BUNDLED_DLLS = frozenset(['vcruntime140.dll'])
