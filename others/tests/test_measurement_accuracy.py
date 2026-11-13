"""Geometry regressions independent of personal reference measurements."""

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from measurement_extraction import (find_chest_height, measure_circumference, ramanujan_perimeter,
                                    section_perimeter_mm, SECTION_PERIMETER_Q,
                                    shoulder_surface_endpoints, side_arm_intersects_height)
from profile_carve import body_depth_profile
from waist_correction import correct_waist
from mediapipe_silhouette import recover_missing_feet


class MeasurementGeometryTests(unittest.TestCase):
    def test_arm_segment_crosses_slice_even_when_joints_are_far_away(self):
        pose = {"left_shoulder": [100, 100], "left_elbow": [100, 400], "left_wrist": [100, 800]}
        self.assertTrue(side_arm_intersects_height(pose, 600, 0, 1, band_mm=90))
        self.assertFalse(side_arm_intersects_height(pose, 950, 0, 1, band_mm=90))

    def test_missing_arm_pose_does_not_assert_an_unoccluded_slice(self):
        self.assertTrue(side_arm_intersects_height({"left_shoulder": [100, 100]}, 600, 0, 1))

    def test_vertical_arm_normals_point_outward_on_both_shoulders(self):
        mask = np.zeros((250, 300), np.uint8)
        mask[50:220, 80:221] = 1
        pose = {"left_shoulder": [90, 70], "right_shoulder": [210, 70],
                "left_elbow": [90, 170], "right_elbow": [210, 170]}
        endpoints = shoulder_surface_endpoints(mask, pose, 1.)
        self.assertLess(endpoints[0, 0], 90)
        self.assertGreater(endpoints[1, 0], 210)

    def test_missing_feet_refinement_recovers_foreground_without_changing_upper_body(self):
        image = np.full((260, 180, 3), (40, 90, 40), np.uint8)
        cv2.rectangle(image, (60, 20), (120, 230), (230, 230, 230), -1)
        mask = np.zeros((260, 180), np.uint8)
        mask[20:211, 60:121] = 1
        landmarks = [SimpleNamespace(x=0., y=0., visibility=0.) for _ in range(33)]
        for idx, x, y in [(25, 80, 140), (27, 80, 215), (29, 80, 225), (31, 85, 228)]:
            landmarks[idx] = SimpleNamespace(x=x/180, y=y/260, visibility=.99)
        refined, flags = recover_missing_feet(image, mask, landmarks)
        self.assertTrue(flags)
        self.assertTrue(refined[225, 80])
        np.testing.assert_array_equal(refined[:120], mask[:120])
        self.assertFalse(refined[:, :40].any())

    def test_low_confidence_foot_cannot_change_height_calibration(self):
        image = np.zeros((260, 180, 3), np.uint8)
        mask = np.zeros((260, 180), np.uint8)
        mask[20:211, 60:121] = 1
        landmarks = [SimpleNamespace(x=.5, y=.95, visibility=.1) for _ in range(33)]
        refined, flags = recover_missing_feet(image, mask, landmarks)
        np.testing.assert_array_equal(refined, mask)
        self.assertEqual(flags, [])

    def test_chest_does_not_measure_fused_sleeves(self):
        mask = np.zeros((230, 500), np.uint8)
        mask[:, 200:301] = 1
        mask[:100, 50:451] = 1  # Sleeves merge with torso above the armpits.
        mask[100:, 50:101] = 1
        mask[100:, 400:451] = 1
        y = find_chest_height(mask, 40, 180, 0, 1, 250)
        self.assertGreater(y, 110)  # The entire local 10mm band must be clean.
        self.assertLess(y, 180)
        cols = np.flatnonzero(mask[int(y)])
        central = cols[(cols >= 200) & (cols <= 300)]
        self.assertEqual(len(central), 101)

    def test_fitted_ellipse_stays_inside_observed_section(self):
        occupied = np.zeros((240, 1, 160), dtype=bool)
        occupied[20:220, :, 20:140] = True
        hull = {"occupied": occupied, "x_mm": np.arange(240.), "y_mm": np.array([0.]),
                "z_mm": np.arange(160.), "voxel_mm": 1.}
        measured = measure_circumference(hull, 0)
        points = np.array(measured["boundary_points_mm"])
        t = np.linspace(0, 2*np.pi, 2000)
        angle = np.radians(measured["rotation_deg"])
        local = np.array([measured["semi_major_mm"]*np.cos(t), measured["semi_minor_mm"]*np.sin(t)])
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ellipse = (rot @ local).T + measured["center_mm"]
        self.assertTrue(np.all(ellipse.min(axis=0) >= points.min(axis=0)-1e-6))
        self.assertTrue(np.all(ellipse.max(axis=0) <= points.max(axis=0)+1e-6))

    def test_circle_circumference_preserves_known_scale(self):
        x, z = np.meshgrid(np.arange(-90., 91.), np.arange(-90., 91.), indexing="ij")
        occupied = (x*x+z*z <= 70**2)[:, None, :]
        hull = {"occupied": occupied, "x_mm": np.arange(-90., 91.), "y_mm": np.array([0.]),
                "z_mm": np.arange(-90., 91.), "voxel_mm": 1.}
        measured = measure_circumference(hull, 0)
        self.assertLess(abs(measured["circumference_mm"]/(2*np.pi*70)-1), 0.02)

    @staticmethod
    def _waist_case(narrowest_y, dip_rows=None):
        mask = np.zeros((280, 220), np.uint8)
        for y in range(280):
            half = 30 if dip_rows and dip_rows[0] <= y < dip_rows[1] else int(40+0.02*(y-narrowest_y)**2)
            mask[y, max(0, 110-min(half, 105)):min(220, 111+min(half, 105))] = 1
        hull = {"occupied": np.ones((220, 280, 40), bool), "x_mm": np.arange(-110., 110.),
                "y_mm": np.arange(280.), "z_mm": np.arange(-20., 20.),
                "voxel_mm": 1., "front_scale_mm_per_px": 1.}
        return correct_waist(mask, hull, 40, 240)

    def test_waist_is_narrowest_row_in_anatomical_band_not_fixed_fraction(self):
        # Band is 160-200mm here; the fixed fraction would sit at 178mm.
        correction = self._waist_case(165)
        self.assertLess(abs(correction["waist_y_mm"]-165), 8)

    def test_waist_ignores_narrow_dip_high_in_torso(self):
        # Elbows crowding the outline produce dips like this 46-56% of the
        # way down; one at 25% must not become the waist.
        correction = self._waist_case(180, dip_rows=(85, 100))
        self.assertGreater(correction["waist_y_mm"], 160)

    def test_section_perimeter_uses_population_q_between_geometric_bounds(self):
        width, depth = 330., 224.
        lower, upper = 2*np.hypot(width, depth), 2*(width+depth)
        chest = section_perimeter_mm(width, depth, "chest")
        self.assertAlmostEqual(chest, lower+SECTION_PERIMETER_Q["chest"]*(upper-lower))
        # A trunk section is squarer than the ellipse inscribed in W x D.
        self.assertGreater(chest, ramanujan_perimeter(width/2, depth/2))
        self.assertAlmostEqual(section_perimeter_mm(2*width, 2*depth, "waist"),
                               2*section_perimeter_mm(width, depth, "waist"))

    def test_depth_profile_keeps_clean_back_edge_beside_forward_arm(self):
        mask = np.zeros((400, 300), np.uint8)
        for y in range(20, 390):
            back = 88 if 95 <= y < 135 else 100  # Scapulae.
            mask[y, back:161] = 1
        mask[100:125, 100:290] = 1  # Arm held forward in profile.
        rows, lo, hi, bad = body_depth_profile(mask, scale_mm_per_px=2.)
        i = int(np.flatnonzero(rows == 115)[0])
        self.assertLess(lo[i], 92)  # Measured back edge, not interpolated to 100.
        self.assertLess(hi[i], 175)  # Arm-contaminated front edge is bridged.
        self.assertTrue(bad[i])

    def test_depth_filter_uses_physical_length_at_different_resolutions(self):
        mask = np.zeros((300, 180), np.uint8)
        for y in range(300):
            half = int(20+8*np.sin(y/300*np.pi))
            mask[y, 90-half:91+half] = 1
        mask[100:125, 20:161] = 1  # Local arm intrusion.
        rows, lo, hi, _ = body_depth_profile(mask, scale_mm_per_px=2.)
        larger = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        rows2, lo2, hi2, _ = body_depth_profile(larger, scale_mm_per_px=1.)
        depth = (hi-lo)*2
        depth2 = np.interp(rows*2, rows2, hi2-lo2)
        self.assertLess(np.mean(abs(depth-depth2)), 2.)

    def test_surface_shoulder_endpoints_are_scale_equivariant(self):
        mask = np.zeros((250, 300), np.uint8)
        cv2.rectangle(mask, (85, 50), (215, 220), 1, -1)
        cv2.line(mask, (90, 70), (40, 150), 1, 30)
        cv2.line(mask, (210, 70), (260, 150), 1, 30)
        landmarks = {"left_shoulder": [90, 70], "right_shoulder": [210, 70],
                     "left_elbow": [40, 150], "right_elbow": [260, 150]}
        endpoints = shoulder_surface_endpoints(mask, landmarks, 2.)
        larger = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        endpoints2 = shoulder_surface_endpoints(larger, {k:np.array(v)*2 for k,v in landmarks.items()}, 1.)
        np.testing.assert_allclose(endpoints*2, endpoints2, atol=2.)


if __name__ == "__main__":
    unittest.main()
