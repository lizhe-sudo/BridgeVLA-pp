"""Contract tests for Astra run naming, finalization, and path relocation."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_DIR))

from astra.output_naming import (
    build_friendly_run_name,
    build_model_slug,
    build_run_result_slug,
    choose_collision_safe_run_path,
    finalize_persisted_run,
    finalize_run_directory,
    find_structured_path_references,
    resolve_run_directory,
    rewrite_structured_artifact_paths,
)


class FriendlyNameTests(unittest.TestCase):
    def test_known_model_names_and_requested_single_episode_examples(self):
        self.assertEqual(build_model_slug("gpt-6-luna"), "luna")
        self.assertEqual(build_model_slug("GPT-6-ASTRA"), "astra")
        self.assertEqual(
            build_friendly_run_name("open_drawer", "gpt-6-luna", "max", "failed"),
            "open_drawer_luna_max_failed",
        )
        self.assertEqual(
            build_friendly_run_name("open_drawer", "gpt-6-astra", "high", "success"),
            "open_drawer_astra_high_success",
        )
        self.assertEqual(
            build_friendly_run_name("place_cups", "gpt-6-astra", "high", "failed"),
            "place_cups_astra_high_failed",
        )

    def test_unknown_model_task_and_reasoning_are_sanitized(self):
        self.assertEqual(build_model_slug("Vendor/Model v2"), "vendor_model_v2")
        name = build_friendly_run_name(
            "task/../../danger", "Vendor/Model v2", "HIGH effort!", "failed"
        )
        self.assertEqual(name, "task_danger_vendor_model_v2_high_effort_failed")
        self.assertNotIn("/", name)
        self.assertNotIn("..", name)
        self.assertFalse(name.endswith(" "))
        self.assertTrue(name)
        self.assertEqual(
            build_friendly_run_name("", "", "", "not-a-result"),
            "unknown_task_unknown_model_unknown_reasoning_incomplete",
        )

    def test_success_failed_incomplete_and_batch_aggregation(self):
        self.assertEqual(build_run_result_slug(
            [{"evaluable": True, "success": True}], "complete", 1
        ), "success")
        self.assertEqual(build_run_result_slug(
            [{"evaluable": True, "success": False}], "complete", 1
        ), "failed")
        self.assertEqual(build_run_result_slug(
            [{"evaluable": False, "success": None}], "incomplete", 1,
            infrastructure_errors=1,
        ), "incomplete")
        mixed_episode_rows = [
            {"evaluable": True, "success": True},
            {"evaluable": True, "success": False},
        ]
        result = build_run_result_slug(mixed_episode_rows, "complete", 2)
        self.assertEqual(result, "mixed")
        self.assertEqual(
            build_friendly_run_name("open_drawer", "gpt-6-astra", "high", result),
            "open_drawer_astra_high_mixed",
        )
        self.assertEqual(
            build_friendly_run_name(
                ["open_drawer", "place_cups"], "gpt-6-astra", "high", result
            ),
            "multi_task_astra_high_mixed",
        )
        self.assertEqual(build_run_result_slug(
            mixed_episode_rows, "complete", 3
        ), "incomplete")

    def test_collision_uses_evaluation_id_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = root / "place_cups_astra_high_failed"
            original.mkdir()
            marker = original / "keep.txt"
            marker.write_text("existing run")
            candidate = choose_collision_safe_run_path(
                root, "place_cups_astra_high_failed", "20260928T012345_123456Z_abcd1234"
            )
            self.assertEqual(
                candidate.name,
                "place_cups_astra_high_failed__20260928T012345_123456Z_abcd1234",
            )
            self.assertEqual(marker.read_text(), "existing run")
            candidate.mkdir()
            with self.assertRaises(FileExistsError):
                choose_collision_safe_run_path(
                    root, "place_cups_astra_high_failed",
                    "20260928T012345_123456Z_abcd1234",
                )


class RunFinalizationTests(unittest.TestCase):
    def make_run(self, output_root, evaluation_id, success, status="complete",
                 extra_rows=None, artifact_index=False):
        old_root = output_root / "astra_{}".format(evaluation_id)
        episode_dir = old_root / "episodes" / "episode0"
        episode_dir.mkdir(parents=True)
        episode_summary = {
            "evaluation_id": evaluation_id,
            "task": "place_cups",
            "success": success,
            "reward": 100.0 if success else 0.0,
            "output_directory": str(episode_dir),
            "instruction": "place cups",
        }
        (episode_dir / "episode_summary.json").write_text(
            json.dumps(episode_summary, indent=2) + "\n"
        )
        manifest = {
            "evaluation_id": evaluation_id,
            "requested_model": "gpt-6-astra",
            "reasoning_effort": "high",
            "output": {
                "run_directory": str(old_root),
                "artifact_layout": {"run_directory": str(old_root)},
            },
            "episode_results": [{"output_directory": str(episode_dir)}],
        }
        summary = {
            "evaluation_id": evaluation_id,
            "completion_status": status,
            "planned_episodes": 1 if extra_rows is None else len(extra_rows),
            "attempted_episodes": 1 if extra_rows is None else len(extra_rows),
            "infrastructure_errors": 0 if status == "complete" else 1,
            "unknown_errors": 0,
            "episode_results": extra_rows or [{
                "task": "place_cups", "evaluable": True, "success": success,
                "output_directory": str(episode_dir),
            }],
        }
        (old_root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        (old_root / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        (old_root / "prompt.txt").write_text(
            "Keep this prompt text unchanged; it mentions {} in prose.\n".format(old_root)
        )
        video = episode_dir / "video.mp4"
        video.write_bytes(b"fake-video-binary\x00\x01")
        if artifact_index:
            files = []
            for path in (old_root / "run_manifest.json", old_root / "run_summary.json",
                         episode_dir / "episode_summary.json", video):
                files.append({
                    "path": path.relative_to(old_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
            (old_root / "artifact_index.json").write_text(json.dumps({
                "evaluation_id": evaluation_id,
                "file_count": len(files),
                "files": files,
            }, indent=2) + "\n")
        return old_root, episode_dir, video

    def test_complete_success_and_failed_runs_finalize_and_preserve_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "outputs"
            root.mkdir()
            for evaluation_id, success, result in (
                    ("eval-success", True, "success"),
                    ("eval-failed", False, "failed")):
                old_root, old_episode, video = self.make_run(
                    root, evaluation_id, success, artifact_index=True
                )
                video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
                final_root = finalize_run_directory(
                    old_root, root, evaluation_id, "place_cups", "gpt-6-astra",
                    "high", result,
                )
                self.assertFalse(old_root.exists())
                self.assertTrue(final_root.is_dir())
                self.assertEqual(final_root.name, "place_cups_astra_high_" + result)
                self.assertEqual(
                    hashlib.sha256((final_root / old_episode.relative_to(old_root) /
                                    "video.mp4").read_bytes()).hexdigest(),
                    video_hash,
                )
                manifest = json.loads((final_root / "run_manifest.json").read_text())
                summary = json.loads((final_root / "run_summary.json").read_text())
                episode = json.loads((final_root / "episodes/episode0/episode_summary.json").read_text())
                self.assertEqual(manifest["evaluation_id"], evaluation_id)
                self.assertEqual(summary["evaluation_id"], evaluation_id)
                self.assertEqual(manifest["friendly_run_name"], final_root.name)
                self.assertEqual(manifest["run_result"], result)
                self.assertEqual(manifest["final_run_directory"], str(final_root))
                self.assertEqual(manifest["output"]["run_directory"], str(final_root))
                self.assertEqual(summary["final_run_directory"], str(final_root))
                self.assertEqual(episode["success"], success)
                self.assertEqual(episode["reward"], 100.0 if success else 0.0)
                self.assertEqual((final_root / "prompt.txt").read_text().count(str(old_root)), 1)
                self.assertEqual(find_structured_path_references(final_root, [old_root]), [])
                self.assertEqual(resolve_run_directory(root, evaluation_id), final_root)

                index = json.loads((final_root / "artifact_index.json").read_text())
                self.assertEqual(index["file_count"], 4)
                for entry in index["files"]:
                    path = final_root / entry["path"]
                    self.assertEqual(entry["size_bytes"], path.stat().st_size)
                    self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_collision_suffix_and_incomplete_run_finalization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "outputs"
            root.mkdir()
            existing = root / "open_drawer_astra_high_incomplete"
            existing.mkdir()
            (existing / "sentinel").write_text("leave intact")
            old_root, _, _ = self.make_run(root, "eval-incomplete", False, status="incomplete")
            final_root = finalize_run_directory(
                old_root, root, "eval-incomplete", "open_drawer", "gpt-6-astra",
                "high", "incomplete",
            )
            self.assertEqual(
                final_root.name,
                "open_drawer_astra_high_incomplete__eval-incomplete",
            )
            self.assertEqual((existing / "sentinel").read_text(), "leave intact")
            self.assertTrue(final_root.is_dir())

    def test_evaluation_id_is_resolvable_after_friendly_rename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "outputs"
            root.mkdir()
            old_root, _, _ = self.make_run(root, "eval-resolve", True)
            final_root = finalize_run_directory(
                old_root, root, "eval-resolve", "open_drawer", "gpt-6-astra",
                "high", "success",
            )
            self.assertEqual(resolve_run_directory(root, "eval-resolve"), final_root)

    def test_post_exit_supervisor_writes_are_finalized_using_evaluation_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "outputs"
            root.mkdir()
            old_root, _, _ = self.make_run(root, "eval-supervised", True)
            (old_root / "supervision.jsonl").write_text(json.dumps({
                "event": "supervisor_exit",
                "evaluation_id": "eval-supervised",
                "run_dir": str(old_root),
            }) + "\n")
            (old_root / "supervisor_stdout.log").write_text(
                "historical output path: {}\n".format(old_root)
            )

            final_root = finalize_persisted_run(root, "eval-supervised")
            self.assertEqual(final_root.name, "place_cups_astra_high_success")
            event = json.loads((final_root / "supervision.jsonl").read_text())
            self.assertEqual(event["run_dir"], str(final_root))
            self.assertEqual(resolve_run_directory(root, "eval-supervised"), final_root)
            self.assertIn(str(old_root), (final_root / "supervisor_stdout.log").read_text())

    def test_cross_run_path_rewrite_is_exact_and_refreshes_indexed_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current = root / "current_run"
            current.mkdir()
            old_a = root / "astra_old_a"
            new_a = root / "open_drawer_astra_high_success"
            old_b = root / "astra_old_b"
            new_b = root / "place_cups_luna_max_failed"
            comparison = current / "action_comparison.json"
            comparison.write_text(json.dumps({
                "sources": {
                    "run_summary": str(old_a / "run_summary.json"),
                    "episode_log": str(old_b / "episodes/episode0/episode_log.jsonl"),
                },
                "review_note": "historical prose mentions {}".format(old_a),
            }, indent=2) + "\n")
            (current / "supervision.jsonl").write_text(json.dumps({
                "run_dir": str(old_b),
                "last_stdout_line": "path mentioned in terminal: {}".format(old_b),
            }) + "\n")
            index = {
                "file_count": 1,
                "files": [{
                    "path": "action_comparison.json",
                    "size_bytes": comparison.stat().st_size,
                    "sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
                }],
            }
            (current / "artifact_index.json").write_text(json.dumps(index, indent=2) + "\n")

            changed_files = rewrite_structured_artifact_paths(current, {
                old_a: new_a,
                old_b: new_b,
            })
            self.assertEqual(changed_files, 2)
            updated = json.loads(comparison.read_text())
            self.assertEqual(updated["sources"]["run_summary"], str(new_a / "run_summary.json"))
            self.assertEqual(
                updated["sources"]["episode_log"],
                str(new_b / "episodes/episode0/episode_log.jsonl"),
            )
            self.assertIn(str(old_a), updated["review_note"])
            supervision = json.loads((current / "supervision.jsonl").read_text())
            self.assertEqual(supervision["run_dir"], str(new_b))
            self.assertIn(str(old_b), supervision["last_stdout_line"])
            indexed = json.loads((current / "artifact_index.json").read_text())
            self.assertEqual(indexed["files"][0]["size_bytes"], comparison.stat().st_size)
            self.assertEqual(
                indexed["files"][0]["sha256"],
                hashlib.sha256(comparison.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
