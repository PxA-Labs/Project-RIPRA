"""Focused integration tests for the Python DM safety bindings.

Set ``RIPRA_TEST_LIBRARY`` to the ABI-3 shared library to run these tests.
With no library path the whole integration class is skipped.  A supplied path
is never silently skipped: load, ABI, and C API failures should be visible.
"""

import os
import unittest

LIBRARY_PATH = os.environ.get("RIPRA_TEST_LIBRARY")

if LIBRARY_PATH:
    import numpy as np

    from rippra.bindings.rippra import (
        DM_ERROR,
        DM_OK,
        DM_PARKED,
        DM_SATURATED,
        Rippra,
    )


@unittest.skipUnless(LIBRARY_PATH, "RIPRA_TEST_LIBRARY is not set")
class TestDMSafetyBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rippra = Rippra(LIBRARY_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.rippra.close()

    def setUp(self):
        # Keep each test independent while exercising the binding's default
        # config path rather than requiring a configuration file.
        self.rippra._cfg = self.rippra.default_config()

    def test_park_returns_zeroed_copy(self):
        source = np.array([1.25, -2.5, 0.0, 9.0], dtype=np.float64)
        source.setflags(write=False)
        parked = self.rippra.dm_park(source)

        np.testing.assert_array_equal(parked, np.zeros(source.size))
        np.testing.assert_array_equal(source, [1.25, -2.5, 0.0, 9.0])
        self.assertIsNot(parked, source)

    def test_saturate_clamps_without_mutating_input(self):
        self.rippra._cfg.dm_max_stroke = 2.0
        self.rippra._cfg.dm_park_threshold = 0.5
        source = np.array([1.0, -3.0, 0.5, 1.8], dtype=np.float64)

        commands, count, status = self.rippra.dm_saturate(source)

        self.assertEqual(count, 1)
        self.assertEqual(status, DM_SATURATED)
        np.testing.assert_allclose(commands, [1.0, -2.0, 0.5, 1.8])
        np.testing.assert_array_equal(source, [1.0, -3.0, 0.5, 1.8])

    def test_nonfinite_command_parks_when_limiting_disabled(self):
        self.rippra._cfg.dm_max_stroke = 0.0
        source = np.array([4.0, np.nan, -7.0], dtype=np.float64)

        commands, count, status = self.rippra.dm_saturate(source)

        self.assertEqual(count, 1)
        self.assertEqual(status, DM_PARKED)
        np.testing.assert_array_equal(commands, np.zeros(source.size))
        np.testing.assert_array_equal(source, [4.0, np.nan, -7.0])

    def test_finite_nonpositive_limit_disables_limiting(self):
        self.rippra._cfg.dm_max_stroke = -1.0
        source = np.array([100.0, -500.0, 20.0], dtype=np.float64)

        commands, count, status = self.rippra.dm_saturate(source)

        self.assertEqual(count, 0)
        self.assertEqual(status, DM_OK)
        np.testing.assert_array_equal(commands, source)

    def test_nonfinite_enabled_config_parks_and_reports_error(self):
        self.rippra._cfg.dm_max_stroke = np.nan
        source = np.array([1.0, -1.0], dtype=np.float64)

        commands, count, status = self.rippra.dm_saturate(source)

        self.assertEqual(count, -1)
        self.assertEqual(status, DM_ERROR)
        np.testing.assert_array_equal(commands, np.zeros(source.size))

    def test_park_threshold_is_clamped_by_c_layer(self):
        self.rippra._cfg.dm_max_stroke = 1.0
        source = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.rippra._cfg.dm_park_threshold = 2.0
        commands, count, status = self.rippra.dm_saturate(source)
        self.assertEqual((count, status), (1, DM_SATURATED))
        np.testing.assert_allclose(commands, [1.0, 0.0, 0.0, 0.0])

        self.rippra._cfg.dm_park_threshold = -1.0
        commands, count, status = self.rippra.dm_saturate(source)
        self.assertEqual((count, status), (1, DM_PARKED))
        np.testing.assert_array_equal(commands, np.zeros(source.size))

    def test_dm_vectors_reject_empty_and_non_one_dimensional_inputs(self):
        invalid = [np.array([]), np.zeros((2, 2))]
        for vector in invalid:
            with self.subTest(shape=vector.shape):
                with self.assertRaises(ValueError):
                    self.rippra.dm_park(vector)
                with self.assertRaises(ValueError):
                    self.rippra.dm_saturate(vector)

        with self.assertRaises(ValueError):
            self.rippra.dm_map(np.zeros((2, 2)), 4)

    def test_configured_map_supports_status_and_default_ndarray_return(self):
        # This is the same physical setup used by the synthetic C pipeline:
        # a 137-spot circular pupil on a 648x492 detector.
        from rippra.ml.synthetic_shwfs import lenslet_positions, render_frame

        cfg = self.rippra._cfg
        width, height = cfg.frame_width, cfg.frame_height
        pitch_px = cfg.pitch / cfg.camera_pixsize
        pupil_radius_px = cfg.pupil_radius / cfg.camera_pixsize
        positions = lenslet_positions(
            width, height, pitch_px, pupil_radius_px, max_lenslets=200
        )
        flat = render_frame(positions, width, height)

        self.assertGreater(self.rippra.calibrate(flat, width, height), 0)
        nnodes = self.rippra._lib.rippra_calibration_nnodes(self.rippra._cal)
        self.assertGreater(nnodes, 0)
        target_phase = np.zeros(nnodes, dtype=np.float64)

        default_commands = self.rippra.dm_map(target_phase, nnodes)
        status_commands, status = self.rippra.dm_map(
            target_phase, nnodes, return_status=True
        )

        self.assertIsInstance(default_commands, np.ndarray)
        self.assertEqual(default_commands.shape, (nnodes,))
        self.assertEqual(status_commands.shape, (nnodes,))
        self.assertEqual(status, DM_OK)
        np.testing.assert_allclose(default_commands, 0.0)
        np.testing.assert_allclose(status_commands, 0.0)


if __name__ == "__main__":
    unittest.main()
