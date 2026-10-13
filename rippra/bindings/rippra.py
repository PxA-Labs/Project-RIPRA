"""
rippra/bindings/rippra.py - Python ctypes bindings for librippra (C reconstructor)

Usage:
    from rippra_bindings import Rippra
    r = Rippra()
    r.load_config("config/system.conf")
    r.calibrate(flat_frame, width, height)
    r.process_frame(frame, width, height)
    # -> (dx, dy, coefficients)

Requires:
    - bin/rippra.dll (Windows) or bin/librippra.so (Linux)
    - numpy
"""

import ctypes
import operator
import os
import numpy as np


# DM safety status values.  Keep these in sync with rippra_api.h.
DM_OK = 0
DM_SATURATED = 1
DM_PARKED = 2
DM_ERROR = -1


def _copy_dm_vector(values, name):
    """Return an owned, aligned, writable C-contiguous DM vector.

    DM entry points write their output buffers.  Copying here means callers may
    pass read-only, strided, or otherwise non-ctypes-compatible NumPy arrays
    without having their input arrays changed.
    """
    try:
        array = np.asarray(values)
    except Exception as exc:
        raise TypeError(f"{name} must be a one-dimensional numeric vector") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    try:
        copied = np.array(array, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional numeric vector") from exc

    # np.array(..., copy=True, order="C") normally guarantees these flags;
    # retain the explicit check because ctypes must never receive a buffer
    # that it cannot safely write.
    if not (copied.flags.c_contiguous and copied.flags.aligned and copied.flags.writeable):
        copied = np.require(copied, dtype=np.float64,
                            requirements=["C", "A", "W", "O"])
    return copied


def _checked_nnodes(nnodes, vector, name):
    """Validate a positive actuator count and its vector length."""
    try:
        count = operator.index(nnodes)
    except TypeError as exc:
        raise TypeError("nnodes must be an integer") from exc
    if count <= 0:
        raise ValueError("nnodes must be positive")
    if vector.size != count:
        raise ValueError(f"{name} length ({vector.size}) must equal nnodes ({count})")
    return count

# ---- C struct definitions (must match rippra_api.h) ---------------------

class RippraConfig(ctypes.Structure):
    _fields_ = [
        ("camera_pixsize",     ctypes.c_double),
        ("frame_width",        ctypes.c_int),
        ("frame_height",       ctypes.c_int),
        ("totlenses",          ctypes.c_int),
        ("flength",            ctypes.c_double),
        ("pitch",              ctypes.c_double),
        ("sa_radius",          ctypes.c_double),
        ("pupil_radius",       ctypes.c_double),
        ("wavelength",         ctypes.c_double),
        ("thresh_binary",      ctypes.c_double),
        ("centroid_percent",   ctypes.c_double),
        ("coarse_grid_radius", ctypes.c_int),
        ("zernike_nmax",       ctypes.c_int),
        ("dm_nact_x",          ctypes.c_int),
        ("dm_nact_y",          ctypes.c_int),
        ("coupling",           ctypes.c_double),
        ("dm_max_stroke",      ctypes.c_double),
        ("dm_park_threshold",  ctypes.c_double),
    ]


class Rippra:
    """Python wrapper around librippra C shared library."""

    ABI_MAJOR = 3
    DM_OK = DM_OK
    DM_SATURATED = DM_SATURATED
    DM_PARKED = DM_PARKED
    DM_ERROR = DM_ERROR

    def __init__(self, lib_path=None):
        if lib_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if os.name == 'nt':
                lib_path = os.path.join(base, "bin", "rippra.dll")
            else:
                lib_path = os.path.join(base, "bin", "librippra.so")
            if not os.path.exists(lib_path):
                # Also check cwd/bin
                lib_path = os.path.join("bin", "rippra.dll" if os.name == 'nt' else "librippra.so")
        self._lib = ctypes.CDLL(lib_path)
        self._check_abi_version()
        self._cal = None  # opaque calibration pointer
        self._nspots = 0
        self._setup_functions()
        # DM safety helpers are useful without a configuration file.  The
        # returned struct is ABI-safe because the major-version check above
        # ran before its struct-return ctypes signature was installed.
        self._cfg = self.default_config()

    def _check_abi_version(self):
        """Check the ABI before registering any struct-return signatures."""
        try:
            version_function = self._lib.rippra_version
        except AttributeError as exc:
            raise RuntimeError(
                "librippra has no version function; rebuild the shared library "
                "from this checkout for Python binding ABI 3.x"
            ) from exc
        version_function.restype = ctypes.c_char_p
        raw_version = version_function()
        if not raw_version:
            raise RuntimeError(
                "librippra returned no version; rebuild the shared library "
                "from this checkout for Python binding ABI 3.x"
            )
        try:
            version = raw_version.decode("ascii")
            major = int(version.split(".", 1)[0])
        except (UnicodeDecodeError, ValueError, IndexError) as exc:
            raise RuntimeError(
                f"unrecognized librippra version {raw_version!r}; rebuild the "
                "shared library from this checkout"
            ) from exc
        if major != self.ABI_MAJOR:
            raise RuntimeError(
                f"librippra ABI {version} is incompatible with the Python "
                f"bindings (requires ABI {self.ABI_MAJOR}.x; the config struct "
                "changed). Rebuild the shared library from this checkout and "
                "pass its path to Rippra(lib_path=...)."
            )
        self._version = version

    def _setup_functions(self):
        lib = self._lib

        # rippra_default_config
        lib.rippra_default_config.restype = RippraConfig

        # rippra_config_load
        lib.rippra_config_load.argtypes = [ctypes.POINTER(RippraConfig), ctypes.c_char_p]
        lib.rippra_config_load.restype = ctypes.c_int

        # rippra_calibrate
        lib.rippra_calibrate.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(RippraConfig),
        ]
        lib.rippra_calibrate.restype = ctypes.c_void_p

        # rippra_calibration_free
        lib.rippra_calibration_free.argtypes = [ctypes.c_void_p]
        lib.rippra_calibration_free.restype = None

        # rippra_calibration_nspots
        lib.rippra_calibration_nspots.argtypes = [ctypes.c_void_p]
        lib.rippra_calibration_nspots.restype = ctypes.c_int

        # rippra_calibration_nnodes
        lib.rippra_calibration_nnodes.argtypes = [ctypes.c_void_p]
        lib.rippra_calibration_nnodes.restype = ctypes.c_int

        # rippra_calibration_ref_centroids
        lib.rippra_calibration_ref_centroids.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
        ]
        lib.rippra_calibration_ref_centroids.restype = None

        # rippra_centroid
        lib.rippra_centroid.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
        ]
        lib.rippra_centroid.restype = ctypes.c_int

        # rippra_reconstruct_zonal
        lib.rippra_reconstruct_zonal.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.POINTER(RippraConfig),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
        ]
        lib.rippra_reconstruct_zonal.restype = ctypes.c_int

        # rippra_reconstruct_modal
        lib.rippra_reconstruct_modal.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.POINTER(RippraConfig),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
        ]
        lib.rippra_reconstruct_modal.restype = ctypes.c_int

        # rippra_process_frame
        lib.rippra_process_frame.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(RippraConfig),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
        ]
        lib.rippra_process_frame.restype = ctypes.c_int

        # rippra_compute_r0
        lib.rippra_compute_r0.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(RippraConfig),
        ]
        lib.rippra_compute_r0.restype = ctypes.c_double

        # rippra_compute_tau0
        lib.rippra_compute_tau0.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_double,
        ]
        lib.rippra_compute_tau0.restype = ctypes.c_double

        # rippra_dm_map
        lib.rippra_dm_map.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(RippraConfig),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
        ]
        lib.rippra_dm_map.restype = ctypes.c_int

        # rippra_dm_map_ex (ABI 3): map and report DM safety status.
        try:
            lib.rippra_dm_map_ex.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.POINTER(RippraConfig),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.rippra_dm_map_ex.restype = ctypes.c_int
        except AttributeError as exc:
            raise RuntimeError(
                "librippra reports ABI 3.x but lacks rippra_dm_map_ex; rebuild "
                "the shared library from this checkout"
            ) from exc

        # rippra_dm_saturate
        lib.rippra_dm_saturate.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.POINTER(RippraConfig),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.rippra_dm_saturate.restype = ctypes.c_int

        # rippra_dm_park
        lib.rippra_dm_park.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int,
        ]
        lib.rippra_dm_park.restype = ctypes.c_int

    @property
    def version(self):
        return self._version

    def default_config(self):
        return self._lib.rippra_default_config()

    def load_config(self, path):
        cfg = RippraConfig()
        ret = self._lib.rippra_config_load(ctypes.byref(cfg), path.encode())
        if ret != 0:
            raise RuntimeError(f"rippra_config_load failed with code {ret}")
        self._cfg = cfg
        return cfg

    def calibrate(self, flat_frame, width, height):
        flat = np.ascontiguousarray(flat_frame, dtype=np.float64)
        cal = self._lib.rippra_calibrate(flat, width, height, ctypes.byref(self._cfg))
        if not cal:
            raise RuntimeError("rippra_calibrate failed")
        if self._cal:
            self._lib.rippra_calibration_free(self._cal)
        self._cal = cal
        self._nspots = self._lib.rippra_calibration_nspots(self._cal)
        return self._nspots

    @property
    def nspots(self):
        return self._nspots

    def ref_centroids(self):
        cx = np.zeros(self._nspots, dtype=np.float64)
        cy = np.zeros(self._nspots, dtype=np.float64)
        self._lib.rippra_calibration_ref_centroids(self._cal, cx, cy)
        return cx, cy

    def centroid(self, frame, width, height, mask=None):
        f = np.ascontiguousarray(frame, dtype=np.float64)
        dx = np.zeros(self._nspots, dtype=np.float64)
        dy = np.zeros(self._nspots, dtype=np.float64)
        if mask is None:
            mask_arr = np.ones(self._nspots, dtype=np.int32)
        else:
            mask_arr = np.ascontiguousarray(mask, dtype=np.int32)
        ret = self._lib.rippra_centroid(self._cal, f, width, height, dx, dy, mask_arr)
        if ret != 0:
            raise RuntimeError(f"rippra_centroid failed with code {ret}")
        return dx, dy

    def reconstruct_zonal(self, dx, dy, mask=None):
        dx_a = np.ascontiguousarray(dx, dtype=np.float64)
        dy_a = np.ascontiguousarray(dy, dtype=np.float64)
        if mask is None:
            mask_arr = np.ones(self._nspots, dtype=np.int32)
        else:
            mask_arr = np.ascontiguousarray(mask, dtype=np.int32)
        nnodes = self._nspots  # approx
        phase = np.zeros(nnodes, dtype=np.float64)
        ret = self._lib.rippra_reconstruct_zonal(
            self._cal, dx_a, dy_a, mask_arr, ctypes.byref(self._cfg), phase)
        if ret != 0:
            raise RuntimeError(f"rippra_reconstruct_zonal failed with code {ret}")
        return phase

    def reconstruct_modal(self, dx, dy, mask=None):
        dx_a = np.ascontiguousarray(dx, dtype=np.float64)
        dy_a = np.ascontiguousarray(dy, dtype=np.float64)
        if mask is None:
            mask_arr = np.ones(self._nspots, dtype=np.int32)
        else:
            mask_arr = np.ascontiguousarray(mask, dtype=np.int32)
        nmodes = (self._cfg.zernike_nmax + 1) * (self._cfg.zernike_nmax + 2) // 2 - 1
        coeffs = np.zeros(nmodes, dtype=np.float64)
        ret = self._lib.rippra_reconstruct_modal(
            self._cal, dx_a, dy_a, mask_arr, ctypes.byref(self._cfg), coeffs)
        if ret != 0:
            raise RuntimeError(f"rippra_reconstruct_modal failed with code {ret}")
        return coeffs

    def process_frame(self, frame, width, height, mask=None):
        f = np.ascontiguousarray(frame, dtype=np.float64)
        nspots = self._nspots
        nmodes = (self._cfg.zernike_nmax + 1) * (self._cfg.zernike_nmax + 2) // 2 - 1
        dx = np.zeros(nspots, dtype=np.float64)
        dy = np.zeros(nspots, dtype=np.float64)
        if mask is None:
            mask_arr = np.ones(nspots, dtype=np.int32)
        else:
            mask_arr = np.ascontiguousarray(mask, dtype=np.int32)
        coeffs = np.zeros(nmodes, dtype=np.float64)
        ret = self._lib.rippra_process_frame(
            self._cal, f, width, height,
            ctypes.byref(self._cfg), dx, dy, mask_arr, coeffs)
        if ret != 0:
            raise RuntimeError(f"rippra_process_frame failed with code {ret}")
        return dx, dy, coeffs

    def compute_r0(self, dx_series, dy_series, nframes, nspots):
        dx = np.ascontiguousarray(dx_series, dtype=np.float64)
        dy = np.ascontiguousarray(dy_series, dtype=np.float64)
        return self._lib.rippra_compute_r0(
            dx, dy, nframes, nspots, ctypes.byref(self._cfg))

    def compute_tau0(self, dx_series, dy_series, nframes, nspots, frame_rate):
        dx = np.ascontiguousarray(dx_series, dtype=np.float64)
        dy = np.ascontiguousarray(dy_series, dtype=np.float64)
        return self._lib.rippra_compute_tau0(dx, dy, nframes, nspots, frame_rate)

    def dm_map(self, target_phase, nnodes, return_status=False):
        """Map phase to DM commands.

        The default return value remains the command ``ndarray``.  With
        ``return_status=True`` this returns ``(commands, status)`` using the
        DM status constants above.  The mapping always receives an owned,
        writable command buffer; the phase input is never modified.
        """
        phase = _copy_dm_vector(target_phase, "target_phase")
        count = _checked_nnodes(nnodes, phase, "target_phase")
        if self._cal is None:
            raise RuntimeError("dm_map requires a calibrated Rippra instance")
        commands = np.zeros(count, dtype=np.float64, order="C")
        status = ctypes.c_int(DM_ERROR)
        if return_status:
            ret = self._lib.rippra_dm_map_ex(
                phase, count, self._cal, ctypes.byref(self._cfg), commands,
                ctypes.byref(status))
        else:
            # Preserve the original default API and return type.  ABI 3's
            # map implementation still applies the configured safety layer;
            # callers wanting its status use return_status=True above.
            ret = self._lib.rippra_dm_map(
                phase, count, self._cal, ctypes.byref(self._cfg), commands)
        if ret != 0:
            raise RuntimeError(f"rippra_dm_map failed with code {ret}")
        return (commands, status.value) if return_status else commands

    def dm_saturate(self, commands):
        """Apply configured DM bounds and return ``(commands, count, status)``.

        ``commands`` is copied before entering C, so the caller's array is
        never mutated.  ``count`` is the number of over-bound/nonfinite
        entries, or ``-1`` for an invalid C argument/configuration.  In the
        latter case ``status`` is ``DM_ERROR``.
        """
        cmds = _copy_dm_vector(commands, "commands")
        status = ctypes.c_int(DM_ERROR)
        count = self._lib.rippra_dm_saturate(
            cmds, int(cmds.size), ctypes.byref(self._cfg), ctypes.byref(status))
        if count < 0:
            # Older/invalid implementations may fail before writing the
            # status output.  Keep the Python contract deterministic.
            status.value = DM_ERROR
        return cmds, int(count), int(status.value)

    def dm_park(self, commands):
        """Return a zeroed copy of a non-empty one-dimensional DM vector."""
        cmds = _copy_dm_vector(commands, "commands")
        ret = self._lib.rippra_dm_park(cmds, int(cmds.size))
        if ret != 0:
            raise RuntimeError(f"rippra_dm_park failed with code {ret}")
        return cmds

    def close(self):
        if self._cal:
            self._lib.rippra_calibration_free(self._cal)
            self._cal = None

    def __del__(self):
        if hasattr(self, "_cal"):
            self.close()
