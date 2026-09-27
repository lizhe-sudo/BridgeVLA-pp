"""Stable, collision-safe naming and relocation for Astra run artifacts."""

from __future__ import annotations

import ctypes
import datetime as _datetime
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path


_KNOWN_MODEL_SLUGS = {
    "gpt-6-luna": "luna",
    "gpt-6-astra": "astra",
}
_VALID_RESULTS = {"success", "failed", "incomplete", "mixed"}
def _slug(value, fallback="unknown"):
    value = "" if value is None else str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or fallback


def build_model_slug(model):
    """Return a predictable short model name, with explicit known aliases."""
    normalized = "" if model is None else str(model).strip().lower()
    return _KNOWN_MODEL_SLUGS.get(normalized, _slug(normalized, "unknown_model"))


def build_run_result_slug(episode_results, completion_status,
                          planned_episodes=None, infrastructure_errors=0,
                          unknown_errors=0):
    """Summarize a run without treating infrastructure errors as task failure."""
    if (completion_status != "complete" or infrastructure_errors or
            unknown_errors):
        return "incomplete"

    rows = list(episode_results or ())
    if not rows or (planned_episodes is not None and len(rows) != planned_episodes):
        return "incomplete"

    evaluable = []
    for row in rows:
        if row.get("evaluable") is not True or not isinstance(row.get("success"), bool):
            return "incomplete"
        evaluable.append(row["success"])

    if all(evaluable):
        return "success"
    if not any(evaluable):
        return "failed"
    return "mixed"


def build_friendly_run_name(task, model, reasoning, result):
    """Build a single-task or batch run name from already-resolved run facts."""
    tasks = [task] if isinstance(task, str) else list(task or ())
    task_slugs = sorted({_slug(item, "unknown_task") for item in tasks})
    task_slug = task_slugs[0] if len(task_slugs) == 1 else "multi_task"
    result_slug = _slug(result, "incomplete")
    if result_slug not in _VALID_RESULTS:
        result_slug = "incomplete"
    return "{}_{}_{}_{}".format(
        task_slug,
        build_model_slug(model),
        _slug(reasoning, "unknown_reasoning"),
        result_slug,
    )


def choose_collision_safe_run_path(output_root, friendly_name, evaluation_id):
    """Choose the friendly path or its evaluation-id-qualified variant."""
    output_root = Path(output_root).expanduser().absolute()
    friendly_name = str(friendly_name)
    evaluation_id = str(evaluation_id)
    if (not friendly_name or friendly_name in (".", "..") or
            Path(friendly_name).name != friendly_name or "/" in friendly_name or
            "\\" in friendly_name):
        raise ValueError("friendly run name must be a safe single path component")
    if (not evaluation_id or evaluation_id in (".", "..") or
            Path(evaluation_id).name != evaluation_id or "/" in evaluation_id or
            "\\" in evaluation_id):
        raise ValueError("evaluation_id must be a safe single path component")

    candidate = output_root / friendly_name
    if not os.path.lexists(str(candidate)):
        return candidate
    candidate = output_root / "{}__{}".format(friendly_name, evaluation_id)
    if os.path.lexists(str(candidate)):
        raise FileExistsError(
            "both friendly and evaluation-id-qualified run paths exist: {}".format(
                candidate
            )
        )
    return candidate


def resolve_run_directory(output_root, evaluation_id):
    """Resolve a run's current directory from its stable evaluation identity."""
    output_root = Path(output_root).expanduser().absolute()
    matches = []
    if not output_root.is_dir():
        raise FileNotFoundError("Astra output root does not exist: {}".format(output_root))
    for child in output_root.iterdir():
        manifest_path = child / "run_manifest.json"
        if not child.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("evaluation_id") == evaluation_id:
            matches.append(child)
    if len(matches) != 1:
        if not matches:
            raise FileNotFoundError(
                "no run directory found for evaluation_id {}".format(evaluation_id)
            )
        raise RuntimeError(
            "multiple run directories found for evaluation_id {}: {}".format(
                evaluation_id, matches
            )
        )
    return matches[0]


def _replace_path_value(value, path_mapping):
    if isinstance(value, str):
        for old_root, new_root in path_mapping:
            if value == old_root:
                return new_root, True
            if value.startswith(old_root + os.sep):
                return new_root + value[len(old_root):], True
        return value, False
    if isinstance(value, list):
        changed = False
        output = []
        for item in value:
            new_item, item_changed = _replace_path_value(item, path_mapping)
            output.append(new_item)
            changed = changed or item_changed
        return output, changed
    if isinstance(value, dict):
        changed = False
        output = {}
        for key, item in value.items():
            new_item, item_changed = _replace_path_value(item, path_mapping)
            output[key] = new_item
            changed = changed or item_changed
        return output, changed
    return value, False


def _serialize_json_like(original_text, value, jsonl=False):
    newline = "\r\n" if "\r\n" in original_text else "\n"
    had_trailing_newline = original_text.endswith(("\n", "\r"))
    if jsonl:
        separators = (",", ": ") if '": ' in original_text else (",", ":")
        rendered = json.dumps(value, ensure_ascii=False, separators=separators)
    else:
        indent_match = re.search(r"\n([ \t]+)\S", original_text)
        if indent_match:
            indent = len(indent_match.group(1).replace("\t", "    "))
            rendered = json.dumps(value, ensure_ascii=False, indent=indent)
        else:
            separators = (",", ": ") if '": ' in original_text else (",", ":")
            rendered = json.dumps(value, ensure_ascii=False, separators=separators)
    if had_trailing_newline:
        rendered += newline
    return rendered


def _plan_structured_path_updates(root, path_mapping, friendly_name=None,
                                  final_run_directory=None, run_result=None):
    """Parse JSON and JSONL structurally; replace only whole path values."""
    updates = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".json", ".jsonl"):
            continue
        # This audit field intentionally preserves the historical source path.
        if path.name == "run_relocation.json":
            continue
        original = path.read_bytes()
        text = original.decode("utf-8")
        is_jsonl = path.suffix.lower() == ".jsonl"
        if is_jsonl:
            output_lines = []
            changed = False
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    output_lines.append(line)
                    continue
                try:
                    item = json.loads(line)
                except ValueError as exc:
                    raise ValueError("invalid JSONL in {}:{}".format(path, line_number)) from exc
                item, item_changed = _replace_path_value(item, path_mapping)
                if item_changed:
                    line = _serialize_json_like(line, item, jsonl=True)
                    changed = True
                output_lines.append(line)
            rendered = "\n".join(output_lines)
            if text.endswith(("\n", "\r")):
                rendered += "\r\n" if text.endswith("\r\n") else "\n"
            if changed:
                updates[path.relative_to(root)] = (original, rendered.encode("utf-8"))
            continue

        try:
            item = json.loads(text)
        except ValueError as exc:
            raise ValueError("invalid JSON artifact: {}".format(path)) from exc
        item, changed = _replace_path_value(item, path_mapping)
        if path.name in ("run_manifest.json", "run_summary.json"):
            if not isinstance(item, dict):
                raise ValueError("{} must contain a JSON object".format(path))
            if friendly_name is not None:
                item["friendly_run_name"] = friendly_name
            if run_result is not None:
                item["run_result"] = run_result
            if final_run_directory is not None:
                item["final_run_directory"] = str(final_run_directory)
                item["output_directory"] = str(final_run_directory)
            if path.name == "run_manifest.json" and final_run_directory is not None:
                output = item.setdefault("output", {})
                if isinstance(output, dict):
                    output["friendly_run_name"] = friendly_name
                    if run_result is not None:
                        output["run_result"] = run_result
                    output["final_run_directory"] = str(final_run_directory)
                    output["run_directory"] = str(final_run_directory)
                    layout = output.get("artifact_layout")
                    if isinstance(layout, dict):
                        layout["run_directory"] = str(final_run_directory)
            changed = True
        if changed:
            rendered = _serialize_json_like(text, item).encode("utf-8")
            updates[path.relative_to(root)] = (original, rendered)
    return updates


def _atomic_write_bytes(path, data, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary_name, mode)
        else:
            try:
                os.chmod(temporary_name, stat.S_IMODE(path.stat().st_mode))
            except FileNotFoundError:
                pass
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _rename_noreplace(source, destination):
    """Rename atomically without collisions, including on filesystems lacking renameat2."""
    source = Path(source)
    destination = Path(destination)
    lock_identity = hashlib.sha256(
        os.fsencode(str(destination.parent.resolve()))
    ).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / (
        "astra-run-directory-rename-{}.lock".format(lock_identity)
    )
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if os.path.lexists(str(destination)):
            raise FileExistsError(str(destination))
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                  ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100, os.fsencode(str(source)), -100,
                os.fsencode(str(destination)), 1
            )
            if result == 0:
                return
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(str(destination))
            unsupported = {
                errno.EINVAL, errno.ENOSYS,
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                getattr(errno, "ENOTSUP", errno.EINVAL),
            }
            if error_number not in unsupported:
                raise OSError(
                    error_number, os.strerror(error_number),
                    str(source), str(destination)
                )
        # Some network filesystems reject RENAME_NOREPLACE. Every Astra
        # finalizer serializes through this output-root lock and rechecks the
        # target immediately before the same-filesystem atomic rename.
        if os.path.lexists(str(destination)):
            raise FileExistsError(str(destination))
        os.rename(str(source), str(destination))


def _refresh_artifact_index(root, changed_paths=(), added_paths=()):
    index_path = Path(root) / "artifact_index.json"
    if not index_path.is_file():
        return False
    original = index_path.read_bytes()
    index = json.loads(original.decode("utf-8"))
    collection_key = "files" if isinstance(index.get("files"), list) else "artifacts"
    entries = index.get(collection_key)
    if not isinstance(entries, list):
        raise ValueError("unrecognized artifact_index.json structure")
    changed_set = {str(Path(path).as_posix()) for path in changed_paths}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        relative = Path(entry["path"])
        if relative.is_absolute() or relative.as_posix() not in changed_set:
            continue
        artifact = Path(root) / relative
        if not artifact.is_file():
            raise FileNotFoundError("artifact index path is missing: {}".format(artifact))
        entry["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if "size_bytes" in entry:
            entry["size_bytes"] = artifact.stat().st_size
        elif "bytes" in entry:
            entry["bytes"] = artifact.stat().st_size
    existing = {str(entry.get("path")) for entry in entries if isinstance(entry, dict)}
    for relative_raw in added_paths:
        relative = Path(relative_raw)
        key = relative.as_posix()
        artifact = Path(root) / relative
        if key in existing or not artifact.is_file():
            continue
        size_key = "size_bytes" if collection_key == "files" else "bytes"
        entries.append({
            "path": key,
            size_key: artifact.stat().st_size,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        })
    if "file_count" in index:
        index["file_count"] = len(entries)
    rendered = _serialize_json_like(original.decode("utf-8"), index).encode("utf-8")
    _atomic_write_bytes(index_path, rendered)
    return True


def rewrite_structured_artifact_paths(run_dir, path_mapping):
    """Repair exact run-root prefixes in structured files after cross-run moves.

    ``path_mapping`` maps historical absolute run roots to current absolute
    roots. Text prompts, natural-language fields, and plain-text logs are not
    scanned or rewritten.
    """
    root = Path(run_dir).expanduser().absolute()
    normalized = []
    for old_root, new_root in dict(path_mapping).items():
        normalized.append((
            str(Path(old_root).expanduser().absolute()),
            str(Path(new_root).expanduser().absolute()),
        ))
    normalized.sort(key=lambda item: len(item[0]), reverse=True)
    updates = _plan_structured_path_updates(root, normalized)
    if not updates:
        return 0

    written = []
    index_path = root / "artifact_index.json"
    index_original = index_path.read_bytes() if index_path.is_file() else None
    try:
        _apply_updates(root, updates, written)
        if index_original is not None:
            written.append((Path("artifact_index.json"), index_original))
        changed_paths = set(updates)
        changed_paths.discard(Path("artifact_index.json"))
        _refresh_artifact_index(root, changed_paths)
    except Exception:
        _restore_updates(root, written)
        raise
    return len(updates)


def _apply_updates(root, updates, written):
    for relative, (original, updated) in updates.items():
        path = Path(root) / relative
        _atomic_write_bytes(path, updated)
        written.append((relative, original))


def _restore_updates(root, written):
    for relative, original in reversed(written):
        _atomic_write_bytes(Path(root) / relative, original)


def relocate_run_directory(run_dir, new_run_dir, evaluation_id,
                           friendly_run_name=None, run_result=None,
                           relocation_audit=None):
    """Atomically move a run and repair structured paths without touching media."""
    source = Path(run_dir).expanduser().absolute()
    destination = Path(new_run_dir).expanduser().absolute()
    if not source.is_dir():
        raise FileNotFoundError("run directory does not exist: {}".format(source))
    if os.path.lexists(str(destination)):
        raise FileExistsError("refusing to replace run directory: {}".format(destination))
    if relocation_audit is not None and (source / "run_relocation.json").exists():
        raise FileExistsError(
            "refusing to replace existing relocation audit: {}".format(
                source / "run_relocation.json"
            )
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise OSError("run directory rename must stay on the same filesystem")

    path_mapping = [(str(source), str(destination))]
    updates = _plan_structured_path_updates(
        source,
        path_mapping,
        friendly_name=friendly_run_name,
        final_run_directory=destination,
        run_result=run_result,
    )
    moved = False
    written = []
    audit_path = destination / "run_relocation.json"
    audit_created = False
    try:
        _rename_noreplace(source, destination)
        moved = True
        _apply_updates(destination, updates, written)
        added_paths = []
        if relocation_audit is not None:
            audit = dict(relocation_audit)
            audit.setdefault("artifact_type", "run_relocation_audit")
            audit.setdefault("evaluation_id", evaluation_id)
            audit.setdefault("old_run_directory", str(source))
            audit.setdefault("new_run_directory", str(destination))
            audit.setdefault("relocated_at", _datetime.datetime.now(
                _datetime.timezone.utc
            ).isoformat())
            audit.setdefault("content_semantics_changed", False)
            audit.setdefault("binary_artifacts_modified", False)
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            audit_created = True
            added_paths.append(audit_path.relative_to(destination))

        changed_paths = set(updates)
        changed_paths.discard(Path("artifact_index.json"))
        index_path = destination / "artifact_index.json"
        index_original = index_path.read_bytes() if index_path.is_file() else None
        if index_original is not None:
            written.append((Path("artifact_index.json"), index_original))
        _refresh_artifact_index(destination, changed_paths, added_paths)

        stale = find_structured_path_references(destination, [str(source)])
        # The relocation record deliberately preserves the historical source path.
        stale = [item for item in stale if item[0] != "run_relocation.json"]
        if stale:
            raise RuntimeError("stale structured artifact paths remain: {}".format(stale[:5]))
    except Exception:
        try:
            if audit_created:
                audit_path.unlink()
            _restore_updates(destination, written)
            if moved and not os.path.lexists(str(source)):
                _rename_noreplace(destination, source)
        except Exception as rollback_error:
            raise RuntimeError(
                "run relocation failed and rollback was incomplete; inspect {} and {}: {}".format(
                    source, destination, rollback_error
                )
            )
        raise

    return destination


def finalize_run_directory(run_dir, output_root, evaluation_id,
                           task, model, reasoning, result):
    """Give a complete/incomplete run its final friendly name."""
    friendly_name = build_friendly_run_name(task, model, reasoning, result)
    for attempt in range(2):
        destination = choose_collision_safe_run_path(
            output_root, friendly_name, evaluation_id
        )
        try:
            return relocate_run_directory(
                run_dir,
                destination,
                evaluation_id=evaluation_id,
                friendly_run_name=friendly_name,
                run_result=result,
            )
        except FileExistsError:
            if attempt:
                raise
    raise AssertionError("unreachable collision retry")


def finalize_persisted_run(output_root, evaluation_id):
    """Finalize a persisted run, for evaluators and post-exit supervisors."""
    run_dir = resolve_run_directory(output_root, evaluation_id)
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "run_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "run needs both run_manifest.json and run_summary.json before finalization"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation_id") != evaluation_id or summary.get("evaluation_id") != evaluation_id:
        raise ValueError("manifest and summary evaluation_id do not match")
    status = summary.get("completion_status")
    if status not in ("complete", "incomplete"):
        raise ValueError("run summary does not contain a reliable final status")
    protocol = manifest.get("protocol") or {}
    tasks = protocol.get("selected_tasks") or sorted({
        row.get("task") for row in summary.get("episode_results", ())
        if isinstance(row, dict) and row.get("task")
    })
    result = build_run_result_slug(
        summary.get("episode_results"),
        status,
        planned_episodes=summary.get("planned_episodes"),
        infrastructure_errors=summary.get("infrastructure_errors", 0),
        unknown_errors=summary.get("unknown_errors", 0),
    )
    model = manifest.get("resolved_model") or manifest.get("requested_model")
    reasoning = manifest.get("reasoning_effort")
    friendly_name = build_friendly_run_name(tasks, model, reasoning, result)
    if (manifest.get("friendly_run_name") == friendly_name and
            manifest.get("run_result") == result and
            summary.get("friendly_run_name") == friendly_name and
            summary.get("run_result") == result and
            manifest.get("final_run_directory") == str(run_dir) and
            summary.get("final_run_directory") == str(run_dir)):
        return run_dir
    return finalize_run_directory(
        run_dir,
        output_root,
        evaluation_id,
        tasks,
        model,
        reasoning,
        result,
    )


def find_structured_path_references(run_dir, old_roots):
    """List JSON/JSONL scalar path values still rooted at any old run path."""
    root = Path(run_dir)
    old_roots = tuple(str(Path(value).expanduser().absolute()) for value in old_roots)
    found = []

    def walk(value, location, file_name):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, location + (str(key),), file_name)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, location + (str(index),), file_name)
        elif isinstance(value, str):
            for old_root in old_roots:
                if value == old_root or value.startswith(old_root + os.sep):
                    found.append((file_name, "/".join(location), value))
                    break

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".json", ".jsonl"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                walk(json.loads(text), (), str(path.relative_to(root)))
            else:
                for line_number, line in enumerate(text.splitlines(), 1):
                    if line.strip():
                        walk(json.loads(line), (str(line_number),),
                             str(path.relative_to(root)))
        except (OSError, ValueError, UnicodeError):
            continue
    return found
