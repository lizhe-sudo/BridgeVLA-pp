"""Pure/fake based contract tests for the Astra evaluator."""

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_DIR))

from astra.action_adapter import AstraActionAdapter
from astra.experiment import (
    EVAL_TASKS,
    SANITY_CHECK_TASKS,
    aggregate_results,
    load_episode_ids,
    read_step_limits,
    resolve_budgets,
    select_episode_ids,
    validate_run_protocol,
)
from astra.schemas import AstraAction, AstraObservation
from astra.sim_paths import configure_project_paths, resolve_sim_roots


class ExperimentProtocolTests(unittest.TestCase):
    def test_repository_budget_snapshot_and_uniform_protocol(self):
        limits = read_step_limits(RL_DIR / "configs" / "eval_step_limit.yml")
        self.assertEqual(limits["place_cups"], 35)
        self.assertEqual(limits["stack_blocks"], 35)
        self.assertEqual(limits["stack_cups"], 25)
        self.assertEqual(resolve_budgets(EVAL_TASKS, "uniform25", limits),
                         {task: 25 for task in EVAL_TASKS})
        self.assertEqual(resolve_budgets(EVAL_TASKS, "repo_step_limits", limits), {
            "place_cups": 35,
            "place_shape_in_shape_sorter": 25,
            "put_groceries_in_cupboard": 25,
            "stack_blocks": 35,
            "stack_cups": 25,
        })

    def test_budget_override_is_debug_cap_and_missing_limit_errors(self):
        limits = {task: 25 for task in EVAL_TASKS}
        self.assertEqual(resolve_budgets([EVAL_TASKS[0]], "uniform25", limits, 3),
                         {EVAL_TASKS[0]: 3})
        with self.assertRaisesRegex(ValueError, "no non-null entry"):
            resolve_budgets(["custom_task"], "repo_step_limits", limits)

    def test_main_sanity_groups_and_formal_protocol(self):
        self.assertEqual(SANITY_CHECK_TASKS, ("meat_off_grill",))
        validate_run_protocol("formal", EVAL_TASKS, 25, 5)
        with self.assertRaisesRegex(ValueError, "complete ordered"):
            validate_run_protocol("formal", EVAL_TASKS[:-1], 25, 5)
        with self.assertRaisesRegex(ValueError, "25 episodes and 5 repeats"):
            validate_run_protocol("formal", EVAL_TASKS, 1, 5)

    def test_episode_list_is_explicit_and_reusable(self):
        self.assertEqual(select_episode_ids(0, 2, [7, 8]), [7, 8])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "episodes.json"
            path.write_text(json.dumps({"episode_ids": [7, 8]}), encoding="utf-8")
            self.assertEqual(load_episode_ids(path), [7, 8])
        with self.assertRaisesRegex(ValueError, "length"):
            select_episode_ids(0, 2, [7])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            select_episode_ids(0, 2, [7, 7])

    def test_subset_never_receives_full_suite_macro_average(self):
        result = aggregate_results(
            [{"task": "meat_off_grill", "repeat_id": 1,
              "success": True, "evaluable": True}],
            {"meat_off_grill": 1}, [1], ["meat_off_grill"],
        )
        self.assertFalse(result["main_suite_complete"])
        self.assertIsNone(result["main_suite_repeat_mean_0_1"])
        self.assertEqual(result["sanity_check_results"][0]["task_group"], "sanity_check")

    def test_repeat_aggregation_keeps_task_and_repeat_labels(self):
        episodes = [
            {"task": task, "repeat_id": repeat, "success": repeat == 1,
             "evaluable": True, "error_class": None}
            for repeat in (1, 2) for task in EVAL_TASKS
        ]
        result = aggregate_results(episodes, {task: 1 for task in EVAL_TASKS},
                                   [1, 2], list(EVAL_TASKS))
        self.assertTrue(result["main_suite_complete"])
        self.assertEqual(len(result["task_results"]), 10)
        self.assertEqual(result["main_suite_macro_by_repeat"], [
            {"repeat_id": 1, "success_rate_0_1": 1.0},
            {"repeat_id": 2, "success_rate_0_1": 0.0},
        ])
        self.assertEqual(result["main_suite_repeat_mean_0_1"], 0.5)
        self.assertEqual(result["main_suite_repeat_sample_stddev_0_1"],
                         2 ** -0.5)


class SimulatorPathTests(unittest.TestCase):
    def make_roots(self, base):
        base = Path(base)
        finetune = base / "finetune"
        rlbench_root = finetune / "bridgevla" / "libs" / "RLBench_peract587"
        pyrep_root = finetune / "bridgevla" / "libs" / "PyRep_stepjam231"
        (rlbench_root / "rlbench").mkdir(parents=True)
        (pyrep_root / "pyrep" / "backend").mkdir(parents=True)
        (pyrep_root / "pyrep" / "backend" / "_sim_cffi_test.so").touch()
        return finetune, rlbench_root, pyrep_root

    def test_dedicated_paths_precede_shared_and_are_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            finetune, rlbench_root, pyrep_root = self.make_roots(temp_dir)
            roots = resolve_sim_roots(finetune, environ={})
            self.assertEqual(roots, {"rlbench": rlbench_root.resolve(),
                                     "pyrep": pyrep_root.resolve()})
            shared_pyrep = finetune / "bridgevla" / "libs" / "PyRep"
            shared_pyrep.mkdir(parents=True)
            previous = list(sys.path)
            try:
                sys.path[:] = [str(shared_pyrep)]
                setup = configure_project_paths(
                    finetune / "RLBench", finetune, environ={},
                    loaded_modules={},
                )
                self.assertEqual(sys.path[0], str(pyrep_root.resolve()))
                self.assertLess(sys.path.index(str(pyrep_root.resolve())),
                                sys.path.index(str(shared_pyrep.resolve())))
                self.assertEqual(setup["pyrep_sim_stack"], str(pyrep_root.resolve()))
            finally:
                sys.path[:] = previous

    def test_missing_or_uncompiled_dedicated_dependency_is_not_silent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            finetune = Path(temp_dir)
            with self.assertRaisesRegex(RuntimeError, "RLBENCH_SIM_STACK"):
                resolve_sim_roots(finetune, environ={})
            rlbench_root = finetune / "custom_rlbench"
            pyrep_root = finetune / "custom_pyrep"
            (rlbench_root / "rlbench").mkdir(parents=True)
            (pyrep_root / "pyrep" / "backend").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "PYREP_SIM_STACK"):
                resolve_sim_roots(finetune, environ={
                    "RLBENCH_SIM_STACK": str(rlbench_root),
                    "PYREP_SIM_STACK": str(pyrep_root),
                })

    def test_shared_stack_requires_explicit_opt_in(self):
        env = {"RLBENCH_SIM_STACK": "", "PYREP_SIM_STACK": ""}
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "explicit"):
                resolve_sim_roots(temp_dir, environ=env, allow_shared=False)
            self.assertEqual(resolve_sim_roots(temp_dir, environ=env,
                                               allow_shared=True),
                             {"rlbench": None, "pyrep": None})

    def test_preloaded_wrong_module_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            finetune, rlbench_root, pyrep_root = self.make_roots(temp_dir)
            wrong = SimpleNamespace(__file__=str(Path(temp_dir) / "wrong" / "rlbench.py"))
            with self.assertRaisesRegex(RuntimeError, "already imported"):
                configure_project_paths(
                    finetune / "RLBench", finetune, environ={},
                    loaded_modules={"rlbench": wrong},
                )


class ActionContractTests(unittest.TestCase):
    def test_fixed0_fixed1_and_predict_collision_actions(self):
        base = AstraAction([0, 0, 0], [0, 0, 0, 1], 1)
        self.assertEqual(AstraActionAdapter("fixed0").adapt(base)[-1], 0.0)
        self.assertEqual(AstraActionAdapter("fixed1").adapt(base)[-1], 1.0)
        with self.assertRaisesRegex(ValueError, "requires ignore_collisions"):
            AstraActionAdapter("predict").adapt(base)
        predicted = AstraAction([0, 0, 0], [0, 0, 0, 1], 1, 1)
        self.assertEqual(AstraActionAdapter("predict").adapt(predicted)[-1], 1.0)
        with self.assertRaisesRegex(ValueError, "do not accept"):
            AstraActionAdapter("fixed0").adapt(predicted)

    def test_nan_inf_zero_quaternion_and_invalid_gripper_rejected(self):
        adapter = AstraActionAdapter()
        for position in ([float("nan"), 0, 0], [float("inf"), 0, 0]):
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                adapter.adapt(AstraAction(position, [0, 0, 0, 1], 0))
        with self.assertRaisesRegex(ValueError, "too small"):
            adapter.adapt(AstraAction([0, 0, 0], [0, 0, 0, 0], 0))
        with self.assertRaisesRegex(ValueError, "gripper must be"):
            adapter.adapt(AstraAction([0, 0, 0], [0, 0, 0, 1], 0.5))

    def test_quaternion_normalization_and_passive_workspace_clip_record(self):
        adapter = AstraActionAdapter("fixed0")
        output = adapter.adapt(AstraAction([2, 0, 0], [0, 0, 0, 2], 1))
        adapter.set_workspace_bounds([0, -1, -1], [1, 1, 1])
        self.assertEqual(output[3:7], [0, 0, 0, 1])
        self.assertEqual(adapter.last_diagnostics["raw_policy_action"]["position"],
                         [2.0, 0.0, 0.0])
        self.assertAlmostEqual(adapter.last_diagnostics["effective_planner_target"][0],
                               0.9999999)
        self.assertTrue(adapter.last_diagnostics["workspace_clip"]["applied"])
        self.assertTrue(adapter.last_diagnostics["quaternion_was_normalized"])
        self.assertEqual(len(output), 9)  # one goal vector, never an action chunk

    def test_policy_schema_tracks_predict_mode(self):
        from astra.codex_policy import CodexAstraPolicy
        fixed = CodexAstraPolicy._schema_for_mode("fixed0")
        predicted = CodexAstraPolicy._schema_for_mode("predict")
        self.assertNotIn("ignore_collisions", fixed["properties"])
        self.assertIn("ignore_collisions", predicted["required"])


class PolicyMetadataTests(unittest.TestCase):
    def test_cli_cwd_context_check_detects_ancestor_instructions(self):
        from astra.codex_policy import CodexAstraPolicy

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cwd = root / "fresh" / "nested"
            cwd.mkdir(parents=True)
            self.assertEqual(CodexAstraPolicy._inherited_context_files(cwd), [])
            instruction = root / "AGENTS.md"
            instruction.write_text("inherited", encoding="utf-8")
            self.assertEqual(
                CodexAstraPolicy._inherited_context_files(cwd), [str(instruction)]
            )

    def test_second_step_early_exception_does_not_reuse_first_call_metadata(self):
        from astra.codex_policy import CodexAstraPolicy

        def fake_run(command, **kwargs):
            if command[-1] == "--version":
                return __import__("subprocess").CompletedProcess(
                    command, 0, "codex-cli fake-1\n", ""
                )
            output_path = command[command.index("--output-last-message") + 1]
            Path(output_path).write_text(json.dumps({
                "position": [0.1, 0.2, 0.3],
                "quaternion": [0, 0, 0, 1],
                "gripper": 1,
            }), encoding="utf-8")
            events = json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 11, "output_tokens": 3,
                "reasoning_output_tokens": 2,
            }}) + "\n"
            return __import__("subprocess").CompletedProcess(command, 0, events, "")

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("astra.codex_policy.shutil.which", return_value="/fake/codex"), \
                 mock.patch("astra.codex_policy.subprocess.run", side_effect=fake_run):
                policy = CodexAstraPolicy(work_root=temp_dir)
                policy.reset("place a cup")
                policy.set_evaluation_context("eval-1", 2, "place_cups", 7)
                images = {name: np.zeros((8, 8, 3), dtype=np.uint8)
                          for name in policy.IMAGE_FIELDS}
                observation = AstraObservation("place a cup", images,
                                               [0, 0, 0, 0, 0, 0, 1], True)
                policy.act(observation)
                first = dict(policy.last_metadata)
                self.assertEqual(first["cli_invocation_count"], 1)
                self.assertTrue(first["codex_context_files_checked"])
                self.assertEqual(first["codex_inherited_context_files"], [])
                self.assertEqual(first["token_usage"]["input_tokens"], 11)
                broken = AstraObservation("place a cup", images,
                                          [float("nan"), 0, 0, 0, 0, 0, 1], True)
                with self.assertRaises(Exception):
                    policy.act(broken)
                second = policy.last_metadata
                self.assertEqual(second["step_id"], 1)
                self.assertEqual(second["cli_invocation_count"], 0)
                self.assertIsNone(second["token_usage"])
                self.assertNotIn("prompt", second)
                self.assertNotEqual(first.get("error"), second.get("error"))

    def test_mock_artifacts_are_never_labeled_as_model_runs(self):
        from astra.episode_recorder import EpisodeRecorder
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EpisodeRecorder(temp_dir, "stack_cups", 0,
                                       record_video=False, policy_type="mock")
            step_dir = recorder.steps_dir / "step_000"
            step_dir.mkdir()
            recorder._save_policy_artifacts(step_dir, None)
            metadata = json.loads((step_dir / "policy_metadata.json").read_text())
            self.assertEqual(metadata["policy_type"], "mock")
            self.assertIsNone(metadata["requested_model"])
            self.assertFalse((step_dir / "codex_action.json").exists())
            recorder.close()


class FakeStepTests(unittest.TestCase):
    @staticmethod
    def observation():
        images = {name: np.full((8, 8, 3), i, dtype=np.uint8)
                  for i, name in enumerate(("front", "left_shoulder",
                                            "right_shoulder", "wrist"))}
        return SimpleNamespace(
            gripper_pose=np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=float),
            gripper_open=True,
            **{name + "_rgb": image for name, image in images.items()},
        )

    def test_policy_observation_is_whitelisted_and_images_are_copied(self):
        from astra.observation_adapter import AstraObservationAdapter
        raw = self.observation()
        raw.object_pose = [9, 9, 9]
        raw.task_state = {"secret": True}
        adapted = AstraObservationAdapter().adapt(raw, "stack the cups")
        self.assertEqual(set(adapted.images), {
            "front", "left_shoulder", "right_shoulder", "wrist",
        })
        self.assertFalse(hasattr(adapted, "object_pose"))
        self.assertFalse(hasattr(adapted, "task_state"))
        self.assertFalse(hasattr(adapted, "raw_observation"))
        for name in adapted.images:
            self.assertIsNot(adapted.images[name], getattr(raw, name + "_rgb"))

    def test_recording_does_not_mutate_policy_rgb_arrays(self):
        from astra.episode_recorder import EpisodeRecorder
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EpisodeRecorder(temp_dir, "stack_cups", 0,
                                       record_video=False, policy_type="mock")
            raw = self.observation()
            original = {name: getattr(raw, name + "_rgb").copy()
                        for name in ("front", "left_shoulder", "right_shoulder", "wrist")}
            recorder.begin_step(0, raw)
            for name, image in original.items():
                np.testing.assert_array_equal(getattr(raw, name + "_rgb"), image)
            recorder.close()

    class Scene:
        def __init__(self, dt=0.1):
            self.steps = 0
            self.pyrep = SimpleNamespace(get_simulation_timestep=lambda: dt)
            tip = SimpleNamespace(get_position=lambda: [0, 0, 0])
            self.robot = SimpleNamespace(arm=SimpleNamespace(
                get_tip=lambda: tip,
            ))

        def step(self):
            self.steps += 1

    class CameraRig:
        cameras = {}

        def capture_views(self):
            return {name: np.zeros((4, 4, 3), dtype=np.uint8)
                    for name in ("front", "left_shoulder", "right_shoulder", "wrist")}

        def views_from_observation(self, obs):
            return self.capture_views()

    def test_three_actions_each_sample_motion_and_restore_scene_step(self):
        from astra.episode_recorder import EpisodeRecorder
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EpisodeRecorder(temp_dir, "stack_cups", 0,
                                       record_video=True, view_size=4, fps=5)
            recorder._camera_rig = self.CameraRig()
            recorder._render = lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8)
            motion_frames_per_action = []
            scene = self.Scene()
            original_step = scene.step
            raw = self.observation()
            for action_index in range(3):
                action = AstraAction([0, 0, 0], [0, 0, 0, 1], 1)
                recorder.begin_step(action_index, raw, action, [0, 0, 0, 0, 0, 0, 1, 1, 0])
                frame_count_before_execution = len(recorder._video_frame_paths)
                with recorder.capture_during_execution(scene):
                    scene.step()
                    scene.step()
                self.assertEqual(scene.step, original_step)
                motion_frames_per_action.append(
                    len(recorder._video_frame_paths) - frame_count_before_execution
                )
                recorder.finish_step({"reward": 0, "success": False}, raw, scene)
            self.assertEqual(motion_frames_per_action, [1, 1, 1])
            self.assertEqual(recorder.simulation_step_count, 6)
            self.assertAlmostEqual(recorder.simulation_time_seconds, 0.6)
            self.assertEqual(scene.steps, 6)
            recorder.close()

    def test_capture_error_restores_step_and_never_advances_simulator(self):
        from astra.episode_recorder import EpisodeRecorder

        class BrokenRig(self.CameraRig):
            def capture_views(self):
                raise RuntimeError("fake recording failure")

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EpisodeRecorder(temp_dir, "stack_cups", 0,
                                       record_video=True, view_size=4, fps=5)
            recorder._camera_rig = BrokenRig()
            scene = self.Scene()
            original_step = scene.step
            raw = self.observation()
            recorder.begin_step(0, raw, AstraAction([0, 0, 0], [0, 0, 0, 1], 1),
                                [0, 0, 0, 0, 0, 0, 1, 1, 0])
            with recorder.capture_during_execution(scene):
                scene.step()
            self.assertEqual(scene.step, original_step)
            self.assertEqual(scene.steps, 1)
            self.assertEqual(recorder.simulation_step_count, 1)
            self.assertEqual(recorder.recording_status, "failed")
            recorder.finish_step({"success": False}, raw, scene)
            recorder.close()

    def test_original_scene_step_exception_still_restores_monkeypatch(self):
        from astra.episode_recorder import EpisodeRecorder
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EpisodeRecorder(temp_dir, "stack_cups", 0,
                                       record_video=False, fps=5)
            scene = self.Scene()
            raw = self.observation()
            recorder.begin_step(0, raw)

            def failing_step():
                raise RuntimeError("sim step failed")

            scene.step = failing_step
            original_failing_step = scene.step
            with self.assertRaisesRegex(RuntimeError, "sim step failed"):
                with recorder.capture_during_execution(scene):
                    scene.step()
            self.assertEqual(scene.step, original_failing_step)
            self.assertEqual(recorder.simulation_step_count, 0)
            recorder.close()

    def test_video_encoding_failure_leaves_saved_success_summary(self):
        from astra.episode_recorder import EpisodeRecorder

        class Writer:
            def isOpened(self):
                return False

            def release(self):
                pass

        cv2 = types.SimpleNamespace(
            VideoWriter=lambda *args: Writer(),
            VideoWriter_fourcc=lambda *args: 0,
            IMREAD_COLOR=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EpisodeRecorder(temp_dir, "stack_cups", 0,
                                       record_video=True, policy_type="codex_cli")
            recorder._camera_rig = self.CameraRig()
            summary = {"success": True, "reward": 100, "termination_reason": "task_success"}
            recorder.write_summary(summary)
            with mock.patch.dict(sys.modules, {"cv2": cv2}):
                self.assertIsNone(recorder.finalize_video(summary))
            saved = json.loads(recorder.summary_path.read_text(encoding="utf-8"))
            self.assertTrue(saved["success"])
            self.assertEqual(saved["recording_status"], "failed")
            self.assertIn("writer", saved["recording_error"])
            recorder.close()

    def test_recording_off_and_on_count_the_same_simulator_steps(self):
        from astra.episode_recorder import EpisodeRecorder
        counts = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for record_video in (False, True):
                recorder = EpisodeRecorder(temp_dir, f"task_{record_video}", 0,
                                           record_video=record_video, view_size=4, fps=5)
                if record_video:
                    recorder._camera_rig = self.CameraRig()
                    recorder._render = lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8)
                scene = self.Scene()
                raw = self.observation()
                for index in range(3):
                    action = AstraAction([0, 0, 0], [0, 0, 0, 1], 1)
                    recorder.begin_step(index, raw, action,
                                        [0, 0, 0, 0, 0, 0, 1, 1, 0])
                    with recorder.capture_during_execution(scene):
                        scene.step()
                        scene.step()
                    recorder.finish_step({"success": False}, raw, scene)
                counts.append((scene.steps, recorder.simulation_step_count,
                               recorder.simulation_time_seconds))
                recorder.close()
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[0][:2], (6, 6))

    def test_astra_raw_observation_cache_updates_only_on_successful_step(self):
        base_module = types.ModuleType("utils.custom_rlbench_env")

        class BaseEnv:
            def __init__(self, *args, **kwargs):
                pass

            def extract_obs(self, obs, **kwargs):
                return {"step": getattr(obs, "step_id", None)}

            def reset(self):
                self.extract_obs(SimpleNamespace(step_id="reset"))

            def reset_to_demo(self, index, variation_number=-1):
                self.extract_obs(SimpleNamespace(step_id=index))

            def step(self, result):
                if result == "fail":
                    raise RuntimeError("planner failed")
                self.extract_obs(SimpleNamespace(step_id=result))
                return result

        base_module.CustomMultiTaskRLBenchEnv2 = BaseEnv
        utils_module = types.ModuleType("utils")
        utils_module.__path__ = []
        with mock.patch.dict(sys.modules, {
            "utils": utils_module,
            "utils.custom_rlbench_env": base_module,
        }):
            sys.modules.pop("astra.env", None)
            module = importlib.import_module("astra.env")
            env = module.AstraRLBenchEnv()
            env.reset_to_demo(4)
            self.assertEqual(env.last_raw_observation.step_id, 4)
            with self.assertRaisesRegex(RuntimeError, "planner failed"):
                env.step("fail")
            self.assertIsNone(env.last_raw_observation)
            env.step(5)
            self.assertEqual(env.last_raw_observation.step_id, 5)
        sys.modules.pop("astra.env", None)


class LoggingContractTests(unittest.TestCase):
    def test_infrastructure_failure_and_policy_failure_have_distinct_classes(self):
        import eval_astra
        self.assertEqual(eval_astra._classify_error(ValueError("bad pose"), "action_validation")[0],
                         "policy_failure")
        self.assertEqual(eval_astra._classify_error(RuntimeError("dataset missing"), "reset")[0],
                         "infrastructure_error")
        self.assertEqual(eval_astra._classify_error(RuntimeError("timeout"), "policy")[1],
                         "RuntimeError")


if __name__ == "__main__":
    unittest.main()
