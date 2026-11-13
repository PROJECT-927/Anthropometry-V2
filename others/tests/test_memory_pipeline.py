"""Geometry equivalence and compute admission tests for memory optimizations."""

import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import ndimage

from compute_queue import ComputeQueue
from export_hull_viewer import extract_surface_points
from profile_carve import _close_chessboard
from visual_hull import _carve_grid, _sample_mask, keep_largest_3d_component, remove_small_components
from voxel_memory import check_grid_size, component_counts


class GeometryTests(unittest.TestCase):
    def test_carve_matches_full_meshgrid_with_rounding_and_out_of_bounds(self):
        rng = np.random.default_rng(17)
        for step in (0.5, 2.0, 3.7):
            front, side = rng.random((37, 29)) > 0.5, rng.random((41, 23)) > 0.5
            args = dict(x_range=(-40, 40), y_range=(-15, 75), z_range=(-30, 40),
                        voxel_mm=step, scale_f=2.5, top_f=0.5, cx_f=11.5,
                        scale_s=1.7, top_s=2.5, cx_s=10.5)
            actual, xs, ys, zs = _carve_grid(front, side, **args)
            x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
            expected = (_sample_mask(front, args["top_f"] + y/args["scale_f"], args["cx_f"] + x/args["scale_f"]) &
                        _sample_mask(side, args["top_s"] + y/args["scale_s"], args["cx_s"] + z/args["scale_s"]))
            np.testing.assert_array_equal(actual, expected)

    def test_component_counts_and_selection_match_reference(self):
        rng = np.random.default_rng(21)
        for density in (0, 0.03, 0.2, 1):
            vol = rng.random((29, 17, 13)) < density
            labels, n = ndimage.label(vol, structure=np.ones((3, 3, 3)))
            expected = np.bincount(labels.ravel(), minlength=n+1)
            expected[0] = 0
            np.testing.assert_array_equal(component_counts(labels, n, chunk_cells=37), expected)
            keep = vol if n <= 1 else labels == (1 + np.argmax(expected[1:]))
            np.testing.assert_array_equal(keep_largest_3d_component(vol), keep)
            actual, removed = remove_small_components(vol, min_fraction=0.1)
            if n > 1:
                ids = np.flatnonzero(expected >= 0.1 * expected.sum())
                keep = np.isin(labels, ids[ids > 0])
            else:
                keep = vol
            np.testing.assert_array_equal(actual, keep)
            self.assertEqual(removed, int(vol.sum() - keep.sum()))

    def test_closing_matches_previous_distance_transform_borders(self):
        rng = np.random.default_rng(23)
        for shape in ((3, 7, 11), (19, 17, 13)):
            for density in (0, 0.05, 0.3, 0.9, 1):
                for radius in (1, 2, 4):
                    vol = rng.random(shape) < density
                    padded = np.pad(vol, radius, constant_values=True)
                    dilated = ndimage.distance_transform_cdt(~padded, metric="chessboard") <= radius
                    eroded = ndimage.distance_transform_cdt(dilated, metric="chessboard") > radius
                    expected = eroded[radius:-radius, radius:-radius, radius:-radius]
                    np.testing.assert_array_equal(_close_chessboard(vol, radius), expected)

    def test_viewer_seeded_points_match_previous_order(self):
        vol = np.random.default_rng(31).random((25, 35, 17)) > 0.4
        axes = [np.arange(n, dtype=float)*2 for n in vol.shape]
        ix, iy, iz = np.where(vol & ~ndimage.binary_erosion(vol))
        for target in (90, len(ix)+1):
            vx, vy, vz = axes[0][ix], 72-axes[1][iy], axes[2][iz]
            if len(ix) > target:
                sel = np.random.default_rng(42).choice(len(ix), size=target, replace=False)
                vx, vy, vz = vx[sel], vy[sel], vz[sel]
            expected = np.round(np.stack([vx, vy, vz], axis=1)).astype("<i2")
            actual, count = extract_surface_points(vol, *axes, 72, target)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(count, len(ix))

    def test_grid_budget_rejects_before_allocating(self):
        with patch.dict(os.environ, {"MAX_GRID_VOXELS": "100"}):
            with self.assertRaisesRegex(ValueError, "limit 100"):
                check_grid_size((11, 10, 1))

    def test_stl_in_place_field_matches_previous_filter(self):
        vol = np.random.default_rng(8).random((17, 19, 13)) > 0.5
        previous = ndimage.gaussian_filter(np.pad(vol.astype(np.float32), 4), sigma=1)
        field = np.zeros(tuple(size+8 for size in vol.shape), dtype=np.float32)
        field[4:-4, 4:-4, 4:-4] = vol
        ndimage.gaussian_filter(field, sigma=1, output=field)
        np.testing.assert_array_equal(field, previous)


class QueueTests(unittest.TestCase):
    def test_two_users_queue_and_capacity_is_bounded(self):
        queue = ComputeQueue(workers=1, pending=1)
        started, release, second = threading.Event(), threading.Event(), threading.Event()
        def first():
            started.set()
            release.wait(timeout=5)
        try:
            self.assertTrue(queue.submit(first))
            self.assertTrue(started.wait(timeout=3))
            self.assertTrue(queue.submit(second.set))
            self.assertFalse(queue.submit(lambda: None))
            self.assertFalse(second.is_set())
            release.set()
            self.assertTrue(second.wait(timeout=3))
        finally:
            release.set()
            queue.executor.shutdown(wait=True)

    def test_failure_releases_compute_capacity(self):
        queue = ComputeQueue(workers=1, pending=0)
        errors = []
        def fail():
            raise RuntimeError("example failure")
        self.assertTrue(queue.submit(fail, on_error=errors.append))
        queue.executor.shutdown(wait=True)
        self.assertEqual(str(errors[0]), "example failure")
        self.assertTrue(queue.capacity.acquire(blocking=False))
        queue.capacity.release()


class ApiTests(unittest.TestCase):
    def test_stl_uses_shared_queue_and_does_not_submit_twice(self):
        import app
        with tempfile.TemporaryDirectory() as root, patch.object(app, "RUNS_DIR", root), \
                patch.object(app._compute_queue, "submit", return_value=True) as submit:
            job_id = "abcdef123456"
            folder = Path(root)/job_id
            folder.mkdir()
            (folder/app.STL_GRID).touch()
            with app.app.test_client() as client:
                url = "/api/job/" + job_id + "/stl"
                for _ in range(2):
                    response = client.post(url)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.get_json()["state"], "queued")
                self.assertEqual(client.get(url).get_json()["state"], "queued")
                submit.assert_called_once()
            app._stl_jobs.pop(job_id, None)

    def test_uploads_report_queued_and_full_queue_cleans_up(self):
        import app
        with tempfile.TemporaryDirectory() as root, patch.object(app, "RUNS_DIR", root), \
                patch.object(app, "_save_silhouette", return_value=("mask.png", None)), \
                patch.object(app, "_save_landmarks", return_value=("points.json", None)), \
                patch.object(app._compute_queue, "submit", side_effect=[True, True, False]):
            with app.app.test_client() as client:
                ids = []
                for _ in range(2):
                    response = client.post("/api/run", data={"height": "177"})
                    self.assertEqual(response.status_code, 200)
                    job_id = response.get_json()["job_id"]
                    ids.append(job_id)
                    state = client.get("/api/job/" + job_id).get_json()
                    self.assertEqual(state["state"], "queued")
                self.assertNotEqual(*ids)
                response = client.post("/api/run", data={"height": "177"})
                self.assertEqual(response.status_code, 429)
                self.assertEqual(len(list(Path(root).iterdir())), 2)
                for job_id in ids:
                    app._jobs.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()
