"""Pure/fake based contract tests for the Astra evaluator."""

import importlib
import contextlib
import io
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

import eval_astra

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
from astra.output_paths import resolve_output_layout
from astra.schemas import AstraAction, AstraObservation
from astra.sim_paths import configure_project_paths, resolve_sim_roots


class ExperimentProtocolTests(unittest.TestCase):
    @staticmethod
    def result_row(task, repeat_id=1, episode_id=0, success=False,
                   evaluable=True, error_class=None, core=True):
        return {
            "task": task, "repeat_id": repeat_id, "episode_id": episode_id,
            "attempted": True, "success": success, "evaluable": evaluable,
            "error_class": error_class, "core_records_complete": core,
        }

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
              "episode_id": 0, "attempted": True, "success": True,
              "evaluable": True, "core_records_complete": True}],
            {"meat_off_grill": 1}, [1], ["meat_off_grill"],
        )
        self.assertFalse(result["main_suite_complete"])
        self.assertIsNone(result["main_suite_repeat_mean_0_1"])
        self.assertEqual(result["sanity_check_results"][0]["task_group"], "sanity_check")

    def test_repeat_aggregation_keeps_task_and_repeat_labels(self):
        episodes = [
            self.result_row(task, repeat, success=repeat == 1)
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

    def test_success_denominator_uses_only_valid_evaluable_units(self):
        task = EVAL_TASKS[0]
        rows = [
            self.result_row(task, episode_id=4, success=True),
            self.result_row(task, episode_id=7, success=False),
            self.result_row(task, episode_id=8, success=False,
                            evaluable=False, error_class="infrastructure_error"),
        ]
        result = aggregate_results(
            rows, {task: 3}, [1], [task], expected_episode_ids={task: [4, 7, 8]}
        )
        summary = result["task_results"][0]
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["evaluable_episodes"], 2)
        self.assertEqual(summary["success_rate_0_1"], 0.5)
        self.assertLessEqual(summary["success_rate_0_1"], 1.0)
        self.assertEqual(summary["completion_status"], "incomplete")

    def test_zero_evaluable_units_have_null_rate(self):
        task = EVAL_TASKS[0]
        row = self.result_row(task, success=False, evaluable=False,
                              error_class="unknown", core=False)
        result = aggregate_results([row], {task: 1}, [1], [task])
        summary = result["task_results"][0]
        self.assertEqual(summary["evaluable_episodes"], 0)
        self.assertIsNone(summary["success_rate_0_1"])
        self.assertEqual(summary["unknown_errors"], 1)

    def test_success_evaluable_conflict_is_excluded_and_reported(self):
        task = EVAL_TASKS[0]
        row = self.result_row(task, success=True, evaluable=False, core=False)
        row["observed_task_success"] = True
        summary = aggregate_results([row], {task: 1}, [1], [task])["task_results"][0]
        self.assertEqual(summary["successes"], 0)
        self.assertEqual(summary["evaluable_episodes"], 0)
        self.assertEqual(summary["data_conflicts"], 1)
        self.assertIsNone(summary["success_rate_0_1"])

    def test_duplicate_and_missing_units_make_repeat_incomplete(self):
        task = EVAL_TASKS[0]
        row = self.result_row(task, episode_id=2, success=True)
        rows = [row, dict(row)]
        result = aggregate_results(
            rows, {task: 2}, [1], [task], expected_episode_ids={task: [2, 3]}
        )
        summary = result["task_results"][0]
        self.assertEqual(summary["duplicate_experiment_units"], 2)
        self.assertEqual(summary["missing_episode_ids"], [3])
        self.assertEqual(summary["successes"], 0)
        self.assertEqual(summary["completion_status"], "incomplete")

    def test_unexpected_units_and_partial_repeat_mean_are_explicit(self):
        rows = [
            self.result_row(task, repeat, success=(repeat == 1))
            for repeat in (1, 2) for task in EVAL_TASKS
        ]
        rows.remove(next(row for row in rows if row["task"] == EVAL_TASKS[0]
                        and row["repeat_id"] == 2))
        result = aggregate_results(rows, {task: 1 for task in EVAL_TASKS},
                                   [1, 2], list(EVAL_TASKS))
        self.assertFalse(result["main_suite_complete"])
        self.assertIsNone(result["main_suite_repeat_mean_0_1"])
        self.assertEqual(result["main_suite_partial_repeat_mean_0_1"], 1.0)
        self.assertEqual(result["main_suite_partial_repeat_ids"], [1])

    def test_sanity_check_never_enters_main_macro(self):
        result = aggregate_results(
            [self.result_row("meat_off_grill", success=True)],
            {"meat_off_grill": 1}, [1], ["meat_off_grill"],
        )
        self.assertFalse(result["main_suite_complete"])
        self.assertEqual(len(result["sanity_check_results"]), 1)
        self.assertEqual(result["main_suite_macro_by_repeat"], [])


class RunEvalIntegrationTests(unittest.TestCase):
    @staticmethod
    def package(modules, name):
        value = types.ModuleType(name)
        value.__path__ = []
        modules[name] = value
        return value

    @staticmethod
    def fake_observation():
        images = {
            name + "_rgb": np.zeros((8, 8, 3), dtype=np.uint8)
            for name in ("front", "left_shoulder", "right_shoulder", "wrist")
        }
        return SimpleNamespace(
            gripper_pose=np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=float),
            gripper_open=True,
            **images,
        )

    def test_run_eval_fake_multitask_multirepeat_flow_matches_manifest(self):
        from astra.errors import InvalidPolicyOutput, SimulatorInfrastructureError

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "fake_tasks"
            task_dir.mkdir()
            for name in ("place_cups", "stack_blocks", "meat_off_grill"):
                (task_dir / f"{name}.py").write_text("# fake task\n", encoding="utf-8")
            data_dir = root / "data"
            data_dir.mkdir()
            output_root = root / "outputs with spaces"
            execution_order = []
            policy_calls = []

            modules = {}
            pyrep = types.ModuleType("pyrep")
            pyrep.__file__ = "/fake/simulator/pyrep/__init__.py"
            modules["pyrep"] = pyrep
            rlbench = self.package(modules, "rlbench")
            rlbench.__file__ = "/fake/simulator/rlbench/__init__.py"
            action_modes = self.package(modules, "rlbench.action_modes")
            gripper_modes = types.ModuleType(
                "rlbench.action_modes.gripper_action_modes"
            )
            gripper_modes.Discrete = type("Discrete", (), {})
            modules["rlbench.action_modes.gripper_action_modes"] = gripper_modes
            action_modes.gripper_action_modes = gripper_modes
            rlbench.action_modes = action_modes
            backend = self.package(modules, "rlbench.backend")
            task_module = types.ModuleType("rlbench.backend.task")
            task_module.TASKS_PATH = str(task_dir)
            modules["rlbench.backend.task"] = task_module
            backend.task = task_module
            backend_utils = types.ModuleType("rlbench.backend.utils")

            def task_file_to_task_class(name):
                return type(f"Fake_{name}", (), {"task_name": name})

            backend_utils.task_file_to_task_class = task_file_to_task_class
            modules["rlbench.backend.utils"] = backend_utils
            backend.utils = backend_utils
            rlbench.backend = backend

            class FakeScene:
                _workspace_minx = _workspace_miny = _workspace_minz = -1.0
                _workspace_maxx = _workspace_maxy = _workspace_maxz = 1.0

                def __init__(self):
                    self.step_count = 0
                    self.pyrep = SimpleNamespace(
                        get_simulation_timestep=lambda: 0.05
                    )

                def step(self):
                    self.step_count += 1

            class FakeEnv:
                def __init__(self, task_classes, swap_task_every, **kwargs):
                    self.task_classes = task_classes
                    self.swap_task_every = swap_task_every
                    self._episode_length = kwargs["episode_length"]
                    self._last_raw_observation = None
                    self._last_exception = None
                    self._lang_goal = ""
                    self._task = None
                    self.scene = None

                @property
                def last_raw_observation(self):
                    return self._last_raw_observation

                def launch(self):
                    return None

                def reset_to_demo(self, episode_id):
                    index = len(execution_order)
                    task_index = index // self.swap_task_every
                    task_name = self.task_classes[task_index].task_name
                    execution_order.append((task_name, int(episode_id)))
                    self._lang_goal = task_name
                    self.scene = FakeScene()
                    task = type(
                        "FakeTaskState", (),
                        {"get_name": lambda _self, value=task_name: value},
                    )()
                    self._task = SimpleNamespace(_task=task, _scene=self.scene)
                    self._last_raw_observation = RunEvalIntegrationTests.fake_observation()
                    return self._last_raw_observation

                def step(self, act_result):
                    self._last_raw_observation = None
                    task_name = self._task._task.get_name()
                    if task_name == "meat_off_grill":
                        raise SimulatorInfrastructureError(
                            "fake simulator transport failure"
                        )
                    self._task._scene.step()
                    self._last_raw_observation = RunEvalIntegrationTests.fake_observation()
                    reward = 100.0 if task_name == "place_cups" else 0.0
                    return SimpleNamespace(reward=reward, terminal=reward > 99.0)

                def shutdown(self):
                    return None

            env_module = types.ModuleType("astra.env")
            env_module.AstraRLBenchEnv = FakeEnv
            modules["astra.env"] = env_module

            utils = self.package(modules, "utils")
            peract_utils = types.ModuleType("utils.peract_utils_rlbench")
            peract_utils.CAMERAS = [
                "front", "left_shoulder", "right_shoulder", "wrist"
            ]
            peract_utils.DATA_FOLDER = str(data_dir)
            peract_utils.IMAGE_SIZE = 8
            modules["utils.peract_utils_rlbench"] = peract_utils
            utils.peract_utils_rlbench = peract_utils

            class FakeActionMode:
                def __init__(self, *args):
                    self.args = args

            planning = types.ModuleType("utils.rlbench_planning")
            planning.EndEffectorPoseViaPlanning2 = FakeActionMode
            planning.MoveArmThenGripper2 = FakeActionMode
            modules["utils.rlbench_planning"] = planning
            utils.rlbench_planning = planning

            bridgevla = self.package(modules, "bridgevla")
            bridgevla_libs = self.package(modules, "bridgevla.libs")
            bridgevla_peract = self.package(modules, "bridgevla.libs.peract")
            bridgevla_helpers = self.package(
                modules, "bridgevla.libs.peract.helpers"
            )
            obs_utils = types.ModuleType("bridgevla.libs.peract.helpers.utils")
            obs_utils.create_obs_config = lambda *args, **kwargs: object()
            modules["bridgevla.libs.peract.helpers.utils"] = obs_utils
            bridgevla_helpers.utils = obs_utils
            bridgevla_peract.helpers = bridgevla_helpers
            bridgevla_libs.peract = bridgevla_peract
            bridgevla.libs = bridgevla_libs

            yarr = self.package(modules, "yarr")
            yarr_agents = self.package(modules, "yarr.agents")
            agent = types.ModuleType("yarr.agents.agent")

            class ActResult:
                def __init__(self, action):
                    self.action = action

            agent.ActResult = ActResult
            modules["yarr.agents.agent"] = agent
            yarr_agents.agent = agent
            yarr.agents = yarr_agents

            mock_policy_module = types.ModuleType("astra.mock_policy")

            class FakePolicy:
                last_metadata = None

                def reset(self, instruction):
                    self.instruction = instruction

                def set_evaluation_context(self, **context):
                    self.context = context

                def act(self, observation):
                    policy_calls.append((
                        self.context["task"], self.context["repeat_id"],
                        self.context["episode_id"], self.context.get("step_id", 0),
                    ))
                    if (self.context["task"] == "stack_blocks"
                            and self.context["repeat_id"] == 1):
                        raise InvalidPolicyOutput(
                            "fake invalid model action", error_code="fake_invalid_action"
                        )
                    return AstraAction(
                        list(observation.eef_pose[:3]),
                        list(observation.eef_pose[3:7]), 1,
                    )

            mock_policy_module.MockPolicy = FakePolicy
            mock_policy_module.ManualPolicy = FakePolicy
            modules["astra.mock_policy"] = mock_policy_module

            with mock.patch.dict(sys.modules, modules), \
                 mock.patch.object(eval_astra, "_add_project_paths", return_value={
                     "rlbench_sim_stack": None, "pyrep_sim_stack": None,
                     "coppeliasim_root": None, "coppeliasim_root_exists": False,
                 }), \
                 mock.patch.object(eval_astra, "_git_value", return_value=""), \
                 contextlib.redirect_stdout(io.StringIO()):
                args = eval_astra._build_parser().parse_args([
                    "--tasks", "place_cups", "stack_blocks", "meat_off_grill",
                    "--policy", "mock", "--run-mode", "debug",
                    "--budget-protocol", "uniform25", "--eval-episodes", "1",
                    "--repeats", "2", "--max-waypoints", "1",
                    "--collision-mode", "fixed0", "--no-record-video",
                    "--output-root", str(output_root),
                ])
                results = eval_astra.run_eval(args)

            expected_prefix = [
                ("place_cups", 0), ("place_cups", 0),
                ("stack_blocks", 0), ("stack_blocks", 0),
                ("meat_off_grill", 0),
            ]
            self.assertEqual(execution_order, expected_prefix, repr(results))
            self.assertEqual(len(policy_calls), 5)
            self.assertEqual(len(results), 5)
            run_dir = next(output_root.glob("astra_*"))
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            summary = json.loads((run_dir / "run_summary.json").read_text())
            self.assertFalse(manifest["output"]["artifact_layout"]["standard_layout"])
            self.assertEqual(
                manifest["output"]["artifact_layout"]["external_overrides"],
                ["output_root"],
            )
            self.assertEqual(
                Path(manifest["output"]["policy_work_directory"]),
                run_dir / "policy_work",
            )
            actual_units = [
                (row["task"], row["repeat_id"], row["episode_id"])
                for row in results
            ]
            planned_units = [
                (row["task"], row["repeat_id"], row["episode_id"])
                for row in manifest["protocol"]["planned_episode_units"]
            ]
            manifest_units = [
                (row["task"], row["repeat_id"], row["episode_id"])
                for row in manifest["episode_results"]
            ]
            summary_units = [
                (row["task"], row["repeat_id"], row["episode_id"])
                for row in summary["episode_results"]
            ]
            self.assertEqual(planned_units[:len(actual_units)], actual_units)
            self.assertEqual(manifest_units, actual_units)
            self.assertEqual(summary_units, actual_units)
            self.assertEqual(results[2]["error_class"], "policy_failure")
            self.assertEqual(results[2]["environment_action_attempt_count"], 0)
            self.assertEqual(results[4]["error_class"], "infrastructure_error")
            self.assertFalse(results[4]["evaluable"])
            self.assertIsNone(results[4]["success"])
            self.assertEqual(results[4]["environment_action_attempt_count"], 1)
            self.assertEqual(summary["successes"], 2)
            self.assertEqual(summary["policy_failures"], 2)
            self.assertEqual(summary["infrastructure_errors"], 1)
            self.assertEqual(summary["completion_status"], "incomplete")


class OutputLayoutTests(unittest.TestCase):
    def test_default_paths_are_repo_rooted_across_working_directories(self):
        repo = Path("/repo/example").resolve()
        expected = repo / "outputs"
        paths = []
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            previous = Path.cwd()
            try:
                for cwd in (one, two):
                    os.chdir(cwd)
                    layout = resolve_output_layout(repo, "eval-1")
                    paths.append(layout.output_root)
                    self.assertEqual(layout.output_root, expected)
                    self.assertEqual(layout.run_dir, expected / "astra_eval-1")
                    self.assertEqual(layout.policy_work_dir,
                                     expected / "astra_eval-1" / "policy_work")
                    self.assertEqual(layout.run_log_path,
                                     expected / "astra_eval-1" / "run.jsonl")
                    self.assertTrue(layout.standard)
            finally:
                os.chdir(previous)
        self.assertEqual(paths, [expected, expected])

    def test_explicit_external_paths_are_detectable(self):
        layout = resolve_output_layout(
            "/repo", "eval-2", codex_work_root="/tmp/work", log_file="/tmp/run.jsonl"
        )
        self.assertEqual(set(layout.external_overrides), {"codex_work_root", "log_file"})
        self.assertFalse(layout.standard)

    def test_output_root_override_is_marked_non_unified(self):
        layout = resolve_output_layout(
            "/repo", "eval-3", output_root="/tmp/astra outputs"
        )
        self.assertEqual(layout.external_overrides, ("output_root",))
        self.assertFalse(layout.standard)


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

        captured_commands = []

        def fake_run(command, **kwargs):
            if command[-1] == "--version":
                return __import__("subprocess").CompletedProcess(
                    command, 0, "codex-cli fake-1\n", ""
                )
            captured_commands.append((list(command), kwargs["cwd"]))
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
            work_root = Path(temp_dir) / "policy work 空间"
            with mock.patch("astra.codex_policy.shutil.which", return_value="/fake/codex"), \
                 mock.patch("astra.codex_policy.subprocess.run", side_effect=fake_run):
                policy = CodexAstraPolicy(work_root=work_root)
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
                self.assertTrue(Path(first["accepted_action_path"]).is_file())
                self.assertFalse(Path(first["raw_model_output_path"]).exists())
                command, cwd = captured_commands[0]
                image_args = [command[index + 1] for index, value in enumerate(command[:-1])
                              if value == "-i"]
                expected_images = [str(work_root / policy._episode_dir.name / "step_000"
                                        / f"{camera}.png")
                                   for camera in policy.IMAGE_FIELDS]
                self.assertEqual(image_args, expected_images)
                self.assertEqual(command[-1], first["prompt"])
                self.assertNotEqual(Path(cwd).parent, work_root)
                self.assertFalse(Path(cwd).exists())
                self.assertEqual(command.count("-i"), 4)
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

    def test_rejected_action_and_stderr_are_bounded_redacted_diagnostics(self):
        from astra.codex_policy import CodexAstraPolicy
        from astra.errors import InvalidPolicyOutput

        credentials = (
            "FAKE_API_SECRET_123456", "FAKE_BEARER_SECRET_987654",
            "FAKE_EVENT_TOKEN_ABCDEF",
        )

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
                "api_key": credentials[0],
            }), encoding="utf-8")
            event_text = json.dumps({
                "type": "turn.completed",
                "item": {"type": "agent_message", "text": f"token={credentials[2]}"},
            }) + "\n"
            return __import__("subprocess").CompletedProcess(
                command, 0, event_text,
                f"Unknown CLI option --bad-option Authorization: Bearer {credentials[1]}",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            work_root = Path(temp_dir) / "policy work"
            policy = None
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch("astra.codex_policy.shutil.which", return_value="/fake/codex"), \
                 mock.patch("astra.codex_policy.subprocess.run", side_effect=fake_run), \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                policy = CodexAstraPolicy(work_root=work_root)
                policy.reset("test task")
                policy.set_evaluation_context("eval-2", 1, "test_task", 0)
                images = {name: np.zeros((4, 4, 3), dtype=np.uint8)
                          for name in policy.IMAGE_FIELDS}
                observation = AstraObservation("test task", images,
                                               [0, 0, 0, 0, 0, 0, 1], True)
                with self.assertRaises(InvalidPolicyOutput):
                    policy.act(observation)
            metadata = policy.last_metadata
            self.assertEqual(metadata["error_code"], "invalid_action_fields")
            self.assertEqual(metadata["step_id"], 0)
            self.assertEqual(metadata["cli_invocation_count"], 1)
            self.assertFalse(Path(metadata["accepted_action_path"]).exists())
            rejected = json.loads(Path(metadata["rejected_output_path"]).read_text())
            self.assertEqual(rejected["status"], "rejected")
            self.assertEqual(rejected["action_fields"]["position"], [0.1, 0.2, 0.3])
            self.assertEqual(rejected["unexpected_field_count"], 1)
            stderr_path = Path(metadata["event_jsonl_path"]).with_name("codex_stderr.log")
            stderr_text = stderr_path.read_text(encoding="utf-8")
            self.assertIn("Unknown CLI option --bad-option", stderr_text)
            persisted = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in work_root.rglob("*") if path.is_file()
            )
            for credential in credentials:
                self.assertNotIn(credential, persisted)
                self.assertNotIn(credential, stdout.getvalue())
                self.assertNotIn(credential, stderr.getvalue())

    def test_tool_event_is_rejected_before_action_is_accepted(self):
        from astra.codex_policy import CodexAstraPolicy
        from astra.errors import PolicyToolViolation

        def fake_run(command, **kwargs):
            if command[-1] == "--version":
                return __import__("subprocess").CompletedProcess(command, 0, "v1", "")
            output_path = command[command.index("--output-last-message") + 1]
            Path(output_path).write_text(json.dumps({
                "position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "gripper": 1,
            }), encoding="utf-8")
            events = json.dumps({"type": "command_execution", "command": "ignored"}) + "\n"
            return __import__("subprocess").CompletedProcess(command, 0, events, "")

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch("astra.codex_policy.shutil.which", return_value="/fake/codex"), \
             mock.patch("astra.codex_policy.subprocess.run", side_effect=fake_run):
            policy = CodexAstraPolicy(work_root=Path(temp_dir) / "policy")
            policy.reset("test")
            images = {name: np.zeros((4, 4, 3), dtype=np.uint8)
                      for name in policy.IMAGE_FIELDS}
            obs = AstraObservation("test", images, [0, 0, 0, 0, 0, 0, 1], True)
            with self.assertRaises(PolicyToolViolation):
                policy.act(obs)
            self.assertFalse(Path(policy.last_metadata["accepted_action_path"]).exists())
            evidence = json.loads(Path(policy.last_metadata["rejected_output_path"]).read_text())
            self.assertEqual(evidence["reason_code"], "tool_event_detected")

    def test_cli_service_error_keeps_redacted_exit_diagnostics(self):
        from astra.codex_policy import CodexAstraPolicy
        from astra.errors import ModelServiceError

        secret = "FAKE_SERVICE_SECRET_012345"

        def fake_run(command, **kwargs):
            if command[-1] == "--version":
                return __import__("subprocess").CompletedProcess(
                    command, 0, "codex-cli fake-1\n", ""
                )
            return __import__("subprocess").CompletedProcess(
                command, 2, '{"type":"turn.failed"}\n',
                f"Unknown CLI option --bad-option api_key={secret}",
            )

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch("astra.codex_policy.shutil.which", return_value="/fake/codex"), \
             mock.patch("astra.codex_policy.subprocess.run", side_effect=fake_run):
            work_root = Path(temp_dir) / "policy"
            policy = CodexAstraPolicy(work_root=work_root)
            policy.reset("test task")
            policy.set_evaluation_context("eval-service", 1, "stack_cups", 0)
            images = {name: np.zeros((4, 4, 3), dtype=np.uint8)
                      for name in policy.IMAGE_FIELDS}
            observation = AstraObservation(
                "test task", images, [0, 0, 0, 0, 0, 0, 1], True
            )
            with self.assertRaises(ModelServiceError) as raised:
                policy.act(observation)
            self.assertEqual(raised.exception.error_code, "codex_cli_nonzero_exit")
            self.assertEqual(
                eval_astra._classify_error(raised.exception, "policy")[0],
                "infrastructure_error",
            )
            metadata = policy.last_metadata
            self.assertEqual(metadata["codex_return_code"], 2)
            self.assertEqual(metadata["cli_invocation_count"], 1)
            self.assertIn("Unknown CLI option --bad-option", metadata["stderr_summary"])
            self.assertNotIn(secret, metadata["stderr_summary"])
            rejected = json.loads(Path(metadata["rejected_output_path"]).read_text())
            self.assertEqual(rejected["status"], "no_output_file")
            persisted = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in work_root.rglob("*") if path.is_file()
            )
            self.assertNotIn(secret, persisted)

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
                raise RuntimeError("fake recording failure api_key=FAKE_CAPTURE_SECRET")

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
            execution = json.loads(
                (recorder.steps_dir / "step_000" / "execution.json").read_text()
            )
            self.assertNotIn("FAKE_CAPTURE_SECRET", json.dumps(execution))
            self.assertNotIn("FAKE_CAPTURE_SECRET", recorder.recording_error)
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

    def test_core_execution_and_episode_summary_write_failures_raise(self):
        import astra.episode_recorder as episode_recorder
        from astra.errors import CoreArtifactWriteError
        from astra.episode_recorder import EpisodeRecorder

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EpisodeRecorder(temp_dir, "stack_cups", 0,
                                       record_video=False)
            recorder.begin_step(0, self.observation())
            original_write_json = episode_recorder._write_json

            def fail_execution(path, value):
                if Path(path).name == "execution.json":
                    raise CoreArtifactWriteError("synthetic core write failure")
                return original_write_json(path, value)

            with mock.patch.object(episode_recorder, "_write_json", side_effect=fail_execution):
                with self.assertRaises(CoreArtifactWriteError):
                    recorder.finish_step({"success": False})
            self.assertIsNone(recorder.recording_error)
            recorder.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EpisodeRecorder(temp_dir, "stack_cups", 0,
                                       record_video=False)

            def fail_summary(path, value):
                if Path(path) == recorder.summary_path:
                    raise CoreArtifactWriteError("synthetic summary write failure")
                return original_write_json(path, value)

            with mock.patch.object(episode_recorder, "_write_json", side_effect=fail_summary):
                with self.assertRaises(CoreArtifactWriteError):
                    recorder.write_summary({"success": True, "core_records_complete": True})
            self.assertIsNone(recorder.recording_error)
            recorder.close()

    def test_episode_jsonl_failure_is_redundant_not_a_recording_error(self):
        from astra.episode_recorder import EpisodeRecorder

        class FailedLog:
            closed = False

            def write(self, _text):
                raise OSError("synthetic log write failure")

            def flush(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EpisodeRecorder(temp_dir, "stack_cups", 0,
                                       record_video=False)
            raw = self.observation()
            recorder.begin_step(0, raw)
            real_log = recorder._log_file
            recorder._log_file = FailedLog()
            recorder.finish_step({"success": False}, raw)
            recorder.write_summary({
                "success": False, "evaluable": True,
                "core_records_complete": True,
            })
            recorder._log_file = real_log
            saved = json.loads(recorder.summary_path.read_text(encoding="utf-8"))
            self.assertTrue((recorder.steps_dir / "step_000" / "execution.json").is_file())
            self.assertIsNone(recorder.recording_error)
            self.assertTrue(any(
                item["path"] == "episode_log.jsonl"
                for item in saved["redundant_record_errors"]
            ))
            recorder.close()

    def test_run_meta_failure_is_redundant_when_episode_summary_persists(self):
        import astra.episode_recorder as episode_recorder
        from astra.episode_recorder import EpisodeRecorder

        original_write_json = episode_recorder._write_json

        def fail_run_meta(path, value):
            if Path(path).name == "run_meta.json":
                raise OSError("synthetic redundant metadata write failure")
            return original_write_json(path, value)

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.object(episode_recorder, "_write_json", side_effect=fail_run_meta):
            recorder = EpisodeRecorder(temp_dir, "stack_cups", 0,
                                       record_video=False)
            recorder.begin_step(0, self.observation())
            recorder.finish_step({"success": False})
            recorder.write_summary({
                "success": False, "evaluable": True,
                "core_records_complete": True,
            })
            saved = json.loads(recorder.summary_path.read_text(encoding="utf-8"))
            self.assertIsNone(recorder.recording_error)
            self.assertTrue(any(
                item["path"] == "run_meta.json"
                for item in saved["redundant_record_errors"]
            ))
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
                         "unknown")
        from astra.errors import InferenceDeadlineExceeded, InvalidPolicyOutput, ModelServiceError
        self.assertEqual(eval_astra._classify_error(
            InvalidPolicyOutput("invalid", error_code="non_finite_position"), "policy"),
            ("policy_failure", "non_finite_position"))
        self.assertEqual(eval_astra._classify_error(
            InvalidPolicyOutput("invalid", error_code="non_finite_quaternion"), "policy"),
            ("policy_failure", "non_finite_quaternion"))
        self.assertEqual(eval_astra._classify_error(
            InferenceDeadlineExceeded("deadline"), "policy"),
            ("policy_failure", "inference_deadline_exceeded"))
        self.assertEqual(eval_astra._classify_error(
            ModelServiceError("unsupported model"), "policy")[0], "infrastructure_error")

    def test_safe_diagnostics_redact_bound_and_fail_closed(self):
        from astra.errors import sanitize_diagnostic
        text = "api_key=FAKE_API_SECRET_123 Authorization: Bearer FAKE_BEARER_456 sk-proj-FAKEKEY123456789"
        result = sanitize_diagnostic(text)
        self.assertNotIn("FAKE_API_SECRET_123", result["text"])
        self.assertNotIn("FAKE_BEARER_456", result["text"])
        self.assertNotIn("sk-proj-FAKEKEY123456789", result["text"])
        self.assertIn("Authorization", result["text"])
        long = sanitize_diagnostic("x" * 5000)
        self.assertTrue(long["truncated"])
        self.assertEqual(len(long["text"]), 4096)
        with mock.patch("astra.errors._SECRET_ASSIGNMENT_PATTERNS", (object(),)):
            failed = sanitize_diagnostic("secret=DO_NOT_WRITE_ABC")
        self.assertTrue(failed["redaction_failed"])
        self.assertNotIn("DO_NOT_WRITE_ABC", failed["text"])


if __name__ == "__main__":
    unittest.main()
