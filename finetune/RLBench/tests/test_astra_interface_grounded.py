"""Tests for the versioned, runtime-grounded Astra robot interface profile."""

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from astra.action_adapter import AstraActionAdapter
from astra.codex_policy import (
    CodexAstraPolicy,
    GENERAL_CLOSED_LOOP_V1_FOLLOWUP_TEMPLATE,
    GENERAL_CLOSED_LOOP_V1_INITIAL_TEMPLATE,
    INTERFACE_GROUNDED_V1_FOLLOWUP_TEMPLATE,
    INTERFACE_GROUNDED_V1_INITIAL_TEMPLATE,
    ROBOT_INTERFACE_NOTES_PATH,
)
from astra.schemas import AstraAction, AstraObservation
from astra.visualization import orientation_error_degrees


def matrix_from_xyzw(quaternion):
    x, y, z, w = [float(value) for value in quaternion]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ])


class InterfaceGroundedProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.config = {
            "robot_setup": "panda",
            "action_mode": {
                "combined_class": "MoveArmThenGripper2",
                "arm_class": "EndEffectorPoseViaPlanning2",
                "absolute_mode": True,
                "frame": "world",
                "gripper_class": "Discrete",
            },
        }
        with mock.patch("astra.codex_policy.shutil.which", return_value=None):
            self.policy = CodexAstraPolicy(
                work_root=str(self.root / "policy work"),
                collision_mode="fixed0",
                motion_prompt_profile="interface_grounded_v1",
                position_tolerance_m=0.0125,
                orientation_tolerance_deg=7.25,
                robot_interface_configuration=self.config,
            )
        self.addCleanup(self.policy.close)
        self.policy.set_evaluation_context("interface-test", 1, "custom", 0, 25)
        self.pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        self.images = {
            camera: np.zeros((8, 8, 3), dtype=np.uint8)
            for camera in CodexAstraPolicy.IMAGE_FIELDS
        }

    def prompt(self, instruction, step=0, pose=None, gripper_open=True):
        pose = self.pose if pose is None else pose
        observation = AstraObservation(
            instruction, self.images, pose, gripper_open
        )
        return self.policy._make_prompt(observation, pose, step)

    def test_cli_registers_new_profile_and_keeps_existing_default(self):
        from eval_astra import _build_parser

        parser = _build_parser()
        self.assertEqual(parser.parse_args([]).motion_prompt_profile,
                         "adaptive_small_steps")
        self.assertEqual(parser.parse_args([
            "--motion-prompt-profile", "general_closed_loop_v1"
        ]).motion_prompt_profile, "general_closed_loop_v1")
        self.assertEqual(parser.parse_args([
            "--motion-prompt-profile", "interface_grounded_v1"
        ]).motion_prompt_profile, "interface_grounded_v1")
        self.assertTrue(GENERAL_CLOSED_LOOP_V1_INITIAL_TEMPLATE)
        self.assertTrue(GENERAL_CLOSED_LOOP_V1_FOLLOWUP_TEMPLATE)

    def test_new_profile_templates_are_versioned_and_distinct(self):
        metadata = CodexAstraPolicy.prompt_profile_metadata(
            "interface_grounded_v1"
        )
        self.assertEqual(metadata["prompt_template_version"],
                         "interface_grounded_v1")
        self.assertEqual(set(metadata["prompt_template_sha256"]),
                         {"first_turn", "followup_turn"})
        self.assertNotEqual(INTERFACE_GROUNDED_V1_INITIAL_TEMPLATE,
                            GENERAL_CLOSED_LOOP_V1_INITIAL_TEMPLATE)
        self.assertNotEqual(INTERFACE_GROUNDED_V1_FOLLOWUP_TEMPLATE,
                            GENERAL_CLOSED_LOOP_V1_FOLLOWUP_TEMPLATE)

    def test_actual_first_prompt_contains_notes_and_dynamic_values(self):
        prompt = self.prompt("Complete the supplied task.")
        self.assertTrue(prompt.startswith(
            "You control the robot described in ROBOT INTERFACE."
        ))
        self.assertIn(self.policy._robot_interface_notes, prompt)
        self.assertIn("fixed0: The evaluator supplies ignore_collisions=0", prompt)
        self.assertIn("less than 0.0125 m", prompt)
        self.assertIn("less than 7.25 degrees", prompt)
        self.assertIn("0.01 m or less", prompt)
        self.assertIn("TASK INSTRUCTION: Complete the supplied task.", prompt)
        self.assertIn(
            "Measured gripper_open: true\n"
            "Interpretation: near fully open. This flag alone does not "
            "establish exact aperture or object contact.",
            prompt,
        )
        self.assertIn("1. front\n2. left_shoulder\n3. right_shoulder\n4. wrist", prompt)
        self.assertNotIn("{{", prompt)
        self.assertNotIn("}}", prompt)

    def test_first_prompt_distinguishes_true_false_and_unknown_measurements(self):
        cases = (
            (True, "Measured gripper_open: true\n"
                   "Interpretation: near fully open. This flag alone does not "
                   "establish exact aperture or object contact."),
            (False, "Measured gripper_open: false\n"
                    "Interpretation: not near fully open. This flag does not "
                    "establish complete closure, exact aperture, or whether "
                    "an object is held."),
            (None, "Measured gripper_open: unknown\n"
                   "Interpretation: unavailable; do not infer aperture, "
                   "closure, or object contact."),
        )
        for measured, expected in cases:
            with self.subTest(measured=measured):
                prompt = self.prompt("Perform the supplied task.",
                                     gripper_open=measured)
                state = prompt[prompt.rfind("Measured gripper_open:"):]
                state = state.split("\n\n", 1)[0]
                self.assertEqual(state, expected)
                self.assertNotIn("Measured gripper state: Measured gripper_open:",
                                 prompt)

    def test_task_changes_do_not_change_fixed_interface_or_principles(self):
        prompts = [self.prompt(task) for task in (
            "open the drawer", "press the buttons", "stack the blocks",
        )]
        prefixes = [value.split("\nCURRENT EPISODE\n", 1)[0]
                    for value in prompts]
        self.assertEqual(prefixes[0], prefixes[1])
        self.assertEqual(prefixes[1], prefixes[2])
        for task_word in ("drawer", "button", "block", "grill", "handle"):
            self.assertNotIn(task_word, prefixes[0].lower())
        self.assertNotEqual(prompts[0], prompts[1])

    def test_followup_uses_only_current_turn_values_and_allowlisted_feedback(self):
        measured_feedback = {
            "step_id": 0,
            "action_id": "action000",
            "eef_pose_after": [0.101, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            "requested_displacement_m": [0.01, 0.0, 0.0],
            "actual_displacement_m": [0.001, 0.0, 0.0],
            "gripper_command": 0,
            "gripper_open_before": True,
            "gripper_open_after": None,
            "position_reached": False,
            "pose_reached": False,
        }
        pose = [0.101, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        for current_measurement, expected in (
            (True, "Measured gripper_open: true\n"
                   "Interpretation: near fully open. This flag alone does not "
                   "establish exact aperture or object contact."),
            (False, "Measured gripper_open: false\n"
                    "Interpretation: not near fully open. This flag does not "
                    "establish complete closure, exact aperture, or whether "
                    "an object is held."),
        ):
            with self.subTest(current_measurement=current_measurement):
                self.policy._pending_feedback = measured_feedback
                prompt = self.prompt(
                    "continue from current evidence", step=1, pose=pose,
                    gripper_open=current_measurement,
                )
                self.assertTrue(prompt.startswith(
                    "Continue the same task episode."
                ))
                self.assertIn(json.dumps(measured_feedback), prompt)
                self.assertIn('"gripper_command": 0', prompt)
                self.assertIn('"gripper_open_before": true', prompt)
                self.assertIn('"gripper_open_after": null', prompt)
                state = prompt[prompt.rfind("Measured gripper_open:"):]
                state = state.split("\n\n", 1)[0]
                self.assertEqual(state, expected)
                self.assertIn("interface-test:r1:custom:ep0:obs001", prompt)
                self.assertNotIn("ROBOT INTERFACE", prompt)
                self.assertNotIn("ROBOT INTERFACE NOTES", prompt)
                self.assertEqual(prompt.count("TASK INSTRUCTION:"), 1)

    def test_notes_are_confirmed_task_independent_and_path_free(self):
        notes = self.policy._robot_interface_notes
        for expected in (
            "world frame, in meters", "Panda_tip", "XYZW order",
            "local +Y spans", "local +Z runs", "gripper command 0 closes",
            "does not indicate that an object is held",
            "do not guarantee that a position-and-orientation target is reachable",
        ):
            self.assertIn(expected.lower(), notes.lower())
        for forbidden in (
            "open_drawer", "meat_off_grill", "/remote_userdata/",
            "Panda_gripper_attachProxSensor", "handle coordinate",
        ):
            self.assertNotIn(forbidden.lower(), notes.lower())
        self.assertNotIn("{{", notes)
        self.assertNotIn("}}", notes)
        self.assertNotIn("Panda_tip pose:", notes)

    def test_profile_records_actual_notes_templates_instructions_and_config_hashes(self):
        metadata = self.policy.prompt_artifact_metadata()
        notes = metadata["robot_interface_notes"]
        self.assertEqual(
            metadata["robot_interface_notes_sha256"],
            hashlib.sha256(notes.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            metadata["robot_interface_notes_template_sha256"],
            hashlib.sha256(ROBOT_INTERFACE_NOTES_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(metadata["robot_interface_configuration"], self.config)
        canonical_config = json.dumps(
            self.config, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        self.assertEqual(
            metadata["robot_interface_configuration_sha256"],
            hashlib.sha256(canonical_config.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(metadata["robot_interface_diagnostic_thresholds"], {
            "position_tolerance_m": 0.0125,
            "orientation_tolerance_deg": 7.25,
        })
        self.assertEqual(
            metadata["application_developer_instructions_sha256"],
            hashlib.sha256(metadata[
                "application_developer_instructions"
            ].encode("utf-8")).hexdigest(),
        )
        self.assertTrue(all(len(value) == 64 for value in
                            metadata["prompt_template_sha256"].values()))

    def test_collision_and_arrival_text_tracks_mode_and_configured_thresholds(self):
        for mode, phrase in (
            ("fixed0", "does not guarantee that collisions are checked throughout the action"),
            ("fixed1", "ignore_collisions=1"),
            ("predict", "action schema includes ignore_collisions"),
        ):
            rendered = CodexAstraPolicy.render_robot_interface_notes(
                ROBOT_INTERFACE_NOTES_PATH.read_text(encoding="utf-8"),
                mode, 0.003, 11.5,
            )
            self.assertIn(phrase, rendered)
            self.assertIn("less than 0.003 m", rendered)
            self.assertIn("less than 11.5 degrees", rendered)
        self.assertNotIn("collision-free guarantee", self.policy._robot_interface_notes)
        self.assertNotIn("entire reachable workspace", self.policy._robot_interface_notes)

    def test_missing_or_unknown_note_sources_fail_closed(self):
        template = ROBOT_INTERFACE_NOTES_PATH.read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "required marker"):
            CodexAstraPolicy.render_robot_interface_notes(
                template.replace("{{POSITION_TOLERANCE_M}}", "unknown"),
                "fixed0", 0.01, 5.0,
            )
        with self.assertRaisesRegex(ValueError, "unresolved placeholders"):
            CodexAstraPolicy.render_robot_interface_notes(
                template + "\n{{UNVERIFIED_AXIS}}", "fixed0", 0.01, 5.0
            )
        with mock.patch(
                "astra.codex_policy.ROBOT_INTERFACE_NOTES_PATH",
                self.root / "missing-notes.txt"):
            with mock.patch("astra.codex_policy.shutil.which", return_value=None):
                with self.assertRaisesRegex(ValueError, "notes are unavailable"):
                    CodexAstraPolicy(
                        work_root=str(self.root / "missing-policy"),
                        motion_prompt_profile="interface_grounded_v1",
                    )

    def test_runtime_fingerprint_matches_loaded_dedicated_stack(self):
        from eval_astra import _add_project_paths

        _add_project_paths(allow_shared=False)
        import pyrep
        import rlbench
        import rlbench.environment as rlbench_environment
        from rlbench.action_modes.gripper_action_modes import Discrete
        from utils.rlbench_planning import (
            EndEffectorPoseViaPlanning2, MoveArmThenGripper2,
        )
        from astra.robot_interface import (
            build_runtime_robot_interface_configuration,
            verify_runtime_robot_interface_configuration,
        )

        mode = MoveArmThenGripper2(EndEffectorPoseViaPlanning2(), Discrete())
        configuration = build_runtime_robot_interface_configuration(
            rlbench, pyrep, rlbench_environment, mode
        )
        verified = verify_runtime_robot_interface_configuration(configuration)
        self.assertEqual(verified["robot_interface_verification"], "verified")
        self.assertEqual(verified["robot_setup"], "panda")
        self.assertEqual(verified["rlbench"]["version"], "1.2.0")
        self.assertEqual(verified["pyrep"]["version"], "4.1.0.3")

    def test_runtime_fingerprint_mismatch_is_rejected(self):
        from astra.robot_interface import verify_runtime_robot_interface_configuration

        incorrect = {
            "robot_setup": "panda",
            "action_mode": {
                "combined_class": "MoveArmThenGripper2",
                "arm_class": "EndEffectorPoseViaPlanning2",
                "absolute_mode": True,
                "frame": "world",
                "gripper_class": "Discrete",
            },
            "rlbench": {"version": "different", "git_sha": "different"},
            "pyrep": {"version": "4.1.0.3",
                      "git_sha": "231a1ac6b0a179cff53c1d403d379260b9f05f2f"},
            "source_file_sha256": {},
            "scene_asset": {"sha256": "different"},
        }
        with self.assertRaisesRegex(RuntimeError, "runtime verification failed"):
            verify_runtime_robot_interface_configuration(incorrect)
        self.assertNotIn("robot_interface_verification", incorrect)

    def test_runtime_quaternion_and_finger_axis_fixture_match_notes(self):
        fixture_path = Path(__file__).parent / "fixtures" / "panda_tip_runtime_rotation.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        quaternion = fixture["tip_quaternion_xyzw"]
        measured_matrix = np.asarray(fixture["rotation_matrix_local_to_world"])
        self.assertGreater(float(np.max(np.abs(measured_matrix - np.eye(3)))), 0.1)
        self.assertLess(float(np.max(np.abs(
            matrix_from_xyzw(quaternion) - measured_matrix
        ))), 5e-8)
        self.assertLess(orientation_error_degrees(
            quaternion, [-v for v in quaternion]
        ), 1e-5)
        finger_delta = fixture["open_finger_centroid_delta_in_tip_frame_m"]
        self.assertGreater(finger_delta[1], 0.09)
        self.assertLess(abs(finger_delta[0]), 0.001)
        self.assertLess(abs(finger_delta[2]), 0.001)
        self.assertIn("local +Y spans", self.policy._robot_interface_notes)
        self.assertNotIn(json.dumps(quaternion), self.policy._robot_interface_notes)
        self.assertIn("test fixture only, not a control target", fixture["scope"])

    def test_adapter_keeps_accepted_targets_and_action_space_unchanged(self):
        target = [2.5, -4.0, 7.25]
        adapter = AstraActionAdapter("fixed0")
        for command in (0, 1):
            with self.subTest(command=command):
                action = AstraAction(target, [0.0, 0.0, 0.0, 2.0], command)
                final_action = adapter.adapt(action)
                self.assertEqual(final_action[:3], target)
                self.assertEqual(final_action[3:7], [0.0, 0.0, 0.0, 1.0])
                self.assertEqual(final_action[7:], [float(command), 0.0])
                self.assertEqual(
                    adapter.last_diagnostics["validated_action"]["gripper"],
                    command,
                )
                self.assertNotIn("workspace_clip", adapter.last_diagnostics)

    def test_four_camera_policy_input_and_recording_resolution_contract_unchanged(self):
        from eval_astra import _build_parser
        from utils.peract_utils_rlbench import IMAGE_SIZE

        args = _build_parser().parse_args([
            "--motion-prompt-profile", "interface_grounded_v1",
        ])
        self.assertEqual(IMAGE_SIZE, 128)
        self.assertEqual((args.recording_width, args.recording_height,
                          args.recording_fps), (1280, 720, 20))
        self.assertEqual(CodexAstraPolicy.IMAGE_FIELDS,
                         ("front", "left_shoulder", "right_shoulder", "wrist"))
        self.assertEqual(set(self.images), set(CodexAstraPolicy.IMAGE_FIELDS))


if __name__ == "__main__":
    unittest.main()
