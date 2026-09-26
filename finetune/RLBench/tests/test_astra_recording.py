"""Recording-only camera and composite-video contract tests."""

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class FakePolicySensor:
    def __init__(self, name):
        self.name = name
        self.pose = np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=float)

    def get_perspective_angle(self):
        return 60.0

    def get_resolution(self):
        return [8, 8]

    def get_near_clipping_plane(self):
        return 0.01

    def get_far_clipping_plane(self):
        return 5.0

    def get_render_mode(self):
        return 0

    def get_position(self):
        return self.pose[:3].copy()

    def get_orientation(self):
        return np.zeros(3)

    def get_pose(self):
        return self.pose.copy()


class FakeRecordingSensor:
    made = []

    @classmethod
    def create(cls, **kwargs):
        sensor = cls(kwargs)
        cls.made.append(sensor)
        return sensor

    def __init__(self, params):
        self.params = params
        self.width, self.height = params["resolution"]
        self.pose = np.asarray(params["position"] + [0, 0, 0, 1], dtype=float)
        self.handled = 0
        self.removed = False

    def get_resolution(self):
        return [self.width, self.height]

    def get_matrix(self):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)

    def get_intrinsic_matrix(self):
        focal = self.width / (2 * math.tan(math.radians(self.params["view_angle"]) / 2))
        return np.array([[focal, 0, self.width / 2],
                         [0, focal, self.height / 2], [0, 0, 1]], dtype=float)

    def set_pose(self, pose):
        self.pose = np.asarray(pose, dtype=float).copy()

    def get_position(self):
        return self.pose[:3].copy()

    def get_quaternion(self):
        return self.pose[3:7].copy()

    def get_perspective_angle(self):
        return self.params["view_angle"]

    def handle_explicitly(self):
        self.handled += 1

    def capture_rgb(self):
        return np.full((self.height, self.width, 3),
                       (self.handled * 13) % 255, dtype=np.uint8)

    def remove(self):
        self.removed = True


class DisplayCamera:
    width = 1280
    height = 720

    @staticmethod
    def project_xyz(_point):
        return None


class RecordingCameraTests(unittest.TestCase):
    def setUp(self):
        FakeRecordingSensor.made = []

    def test_recording_rig_creates_native_720p_sensors_and_tracks_wrist(self):
        from astra.episode_recorder import RecordingCameraRig

        sources = {
            name: FakePolicySensor(name)
            for name in ("front", "left_shoulder", "right_shoulder", "wrist")
        }
        scene = types.SimpleNamespace(
            _cam_front=sources["front"],
            _cam_over_shoulder_left=sources["left_shoulder"],
            _cam_over_shoulder_right=sources["right_shoulder"],
            _cam_wrist=sources["wrist"],
        )
        pyrep = types.ModuleType("pyrep")
        pyrep.__path__ = []
        objects = types.ModuleType("pyrep.objects")
        objects.__path__ = []
        vision_sensor = types.ModuleType("pyrep.objects.vision_sensor")
        vision_sensor.VisionSensor = FakeRecordingSensor
        with mock.patch.dict(sys.modules, {
            "pyrep": pyrep,
            "pyrep.objects": objects,
            "pyrep.objects.vision_sensor": vision_sensor,
        }):
            rig = RecordingCameraRig(scene, 1280, 720)
            try:
                self.assertEqual(len(FakeRecordingSensor.made), 4)
                for sensor in FakeRecordingSensor.made:
                    self.assertEqual(sensor.params["resolution"], [1280, 720])
                    self.assertTrue(sensor.params["explicit_handling"])
                    self.assertAlmostEqual(sensor.params["view_angle"],
                                           math.degrees(2 * math.atan(
                                               math.tan(math.radians(60) / 2) * 16 / 9
                                           )))
                self.assertEqual(
                    rig.metadata()["front"]["policy_input_resolution_xy"], [8, 8]
                )
                self.assertEqual(
                    rig.metadata()["front"]["recording_source_resolution_xy"],
                    [1280, 720],
                )
                self.assertEqual(
                    rig.metadata()["front"]["recording_output_resolution_xy"],
                    [1280, 720],
                )
                self.assertFalse(rig.metadata()["front"]["pose_dynamic"])
                self.assertTrue(rig.metadata()["wrist"]["pose_dynamic"])
                sources["wrist"].pose = np.array([1, 2, 3, 0, 0, 0, 1.0])
                views = rig.capture_views()
                self.assertEqual(set(views), {
                    "front", "left_shoulder", "right_shoulder", "wrist",
                })
                self.assertEqual(views["front"].shape, (720, 1280, 3))
                np.testing.assert_array_equal(
                    rig.sensors["wrist"].pose, sources["wrist"].pose
                )
                self.assertEqual(rig.sensors["wrist"].handled, 1)
            finally:
                rig.close()
        self.assertTrue(all(sensor.removed for sensor in FakeRecordingSensor.made))

    def test_2560_by_1440_grid_layout_encodes_and_decodes(self):
        import cv2
        from astra.episode_recorder import EpisodeRecorder
        from astra.visualization import render_policy_grid

        names = ("front", "left_shoulder", "right_shoulder", "wrist")
        colors = {
            "front": (255, 0, 0), "left_shoulder": (0, 255, 0),
            "right_shoulder": (0, 0, 255), "wrist": (255, 255, 0),
        }
        views = {name: np.full((720, 1280, 3), color, dtype=np.uint8)
                 for name, color in colors.items()}
        cameras = {name: DisplayCamera() for name in names}
        frame = render_policy_grid(views, cameras, "READY", 0)
        self.assertEqual(frame.shape, (1440, 2560, 3))
        np.testing.assert_array_equal(frame[360, 640], colors["front"])
        np.testing.assert_array_equal(frame[360, 1920], colors["left_shoulder"])
        np.testing.assert_array_equal(frame[1080, 640], colors["right_shoulder"])
        np.testing.assert_array_equal(frame[1080, 1920], colors["wrist"])

        class Rig:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EpisodeRecorder(
                temp_dir, "open_drawer", 0, record_video=True,
                recording_width=1280, recording_height=720, fps=20,
            )
            rig = Rig()
            recorder._camera_rig = rig
            for _ in range(2):
                recorder._video_frame_paths.append(recorder._save_video_frame(frame))
            recorder.write_summary({"success": False, "termination_reason": "test"})
            video_path = recorder.finalize_video({"success": False})
            self.assertIsNotNone(video_path)
            capture = cv2.VideoCapture(str(video_path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 2560)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1440)
                ok, decoded = capture.read()
                self.assertTrue(ok)
                self.assertEqual(decoded.shape, (1440, 2560, 3))
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
                self.assertAlmostEqual(
                    capture.get(cv2.CAP_PROP_FRAME_COUNT) /
                    capture.get(cv2.CAP_PROP_FPS), 0.1, places=2,
                )
            finally:
                capture.release()
                recorder.close()
            self.assertTrue(rig.closed)


if __name__ == "__main__":
    unittest.main()
