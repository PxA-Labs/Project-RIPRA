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
import os
import numpy as np

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
    ]


class Rippra:
    """Python wrapper around librippra C shared library."""

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
        self._cal = None  # opaque calibration pointer
        self._nspots = 0
        self._setup_functions()

    def _setup_functions(self):
        lib = self._lib

        # rippra_version
        lib.rippra_version.restype = ctypes.c_char_p

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
        ]
        lib.rippra_centroid.restype = ctypes.c_int

        # rippra_reconstruct_zonal
        lib.rippra_reconstruct_zonal.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.POINTER(RippraConfig),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
        ]
        lib.rippra_reconstruct_zonal.restype = ctypes.c_int

        # rippra_reconstruct_modal
        lib.rippra_reconstruct_modal.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
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

    @property
    def version(self):
        return self._lib.rippra_version().decode()

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

    def centroid(self, frame, width, height):
        f = np.ascontiguousarray(frame, dtype=np.float64)
        dx = np.zeros(self._nspots, dtype=np.float64)
        dy = np.zeros(self._nspots, dtype=np.float64)
        ret = self._lib.rippra_centroid(self._cal, f, width, height, dx, dy)
        if ret != 0:
            raise RuntimeError(f"rippra_centroid failed with code {ret}")
        return dx, dy

    def reconstruct_zonal(self, dx, dy):
        dx_a = np.ascontiguousarray(dx, dtype=np.float64)
        dy_a = np.ascontiguousarray(dy, dtype=np.float64)
        nnodes = self._nspots  # approx
        phase = np.zeros(nnodes, dtype=np.float64)
        ret = self._lib.rippra_reconstruct_zonal(
            self._cal, dx_a, dy_a, ctypes.byref(self._cfg), phase)
        if ret != 0:
            raise RuntimeError(f"rippra_reconstruct_zonal failed with code {ret}")
        return phase

    def reconstruct_modal(self, dx, dy):
        dx_a = np.ascontiguousarray(dx, dtype=np.float64)
        dy_a = np.ascontiguousarray(dy, dtype=np.float64)
        nmodes = (self._cfg.zernike_nmax + 1) * (self._cfg.zernike_nmax + 2) // 2 - 1
        coeffs = np.zeros(nmodes, dtype=np.float64)
        ret = self._lib.rippra_reconstruct_modal(
            self._cal, dx_a, dy_a, ctypes.byref(self._cfg), coeffs)
        if ret != 0:
            raise RuntimeError(f"rippra_reconstruct_modal failed with code {ret}")
        return coeffs

    def process_frame(self, frame, width, height):
        f = np.ascontiguousarray(frame, dtype=np.float64)
        nspots = self._nspots
        nmodes = (self._cfg.zernike_nmax + 1) * (self._cfg.zernike_nmax + 2) // 2 - 1
        dx = np.zeros(nspots, dtype=np.float64)
        dy = np.zeros(nspots, dtype=np.float64)
        coeffs = np.zeros(nmodes, dtype=np.float64)
        ret = self._lib.rippra_process_frame(
            self._cal, f, width, height,
            ctypes.byref(self._cfg), dx, dy, coeffs)
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

    def dm_map(self, target_phase, nnodes):
        phase = np.ascontiguousarray(target_phase, dtype=np.float64)
        commands = np.zeros(nnodes, dtype=np.float64)
        ret = self._lib.rippra_dm_map(
            phase, nnodes, self._cal, ctypes.byref(self._cfg), commands)
        if ret != 0:
            raise RuntimeError(f"rippra_dm_map failed with code {ret}")
        return commands

    def close(self):
        if self._cal:
            self._lib.rippra_calibration_free(self._cal)
            self._cal = None

    def __del__(self):
        self.close()
