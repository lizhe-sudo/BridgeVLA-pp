"""Create and close one real Codex App Server thread without starting a turn.

Run as ``python -m astra.app_server_preflight --output <new-json-path>`` from
finetune/RLBench. This uses CodexAppServerSession's production initialize and
thread/start payload builders and never invokes turn/start.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .codex_session import CodexAppServerSession
from .errors import ModelServiceError, safe_exception_record


PREFLIGHT_INSTRUCTIONS = (
    "This is an App Server session-creation preflight only. Do not use tools. "
    "No turn or control request will be sent."
)


def _codex_version(executable):
    result = subprocess.run(
        [executable, "--version"], capture_output=True, text=True,
        timeout=5, check=False,
    )
    version = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"codex-cli [^\r\n]{1,100}", version):
        raise ModelServiceError(
            "Codex CLI version could not be verified",
            error_code="codex_version_unverified",
        )
    return version


def _write_result(path, result):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def run_preflight(output_path, model="gpt-6-luna", reasoning_effort="max",
                  timeout=180, executable=None):
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(str(output_path))
    started_at = datetime.now(timezone.utc)
    result = {
        "preflight_type": "codex_app_server_thread_create_only",
        "started_at_utc": started_at.isoformat(),
        "requested_model": model,
        "reasoning_effort": reasoning_effort,
        "inference_or_model_request_performed": False,
        "turn_start_called": False,
        "status": "failed",
    }
    session = None
    work_dir = None
    error = None
    process_pid = None
    process_group_id = None
    try:
        executable = executable or shutil.which("codex")
        if not executable:
            raise ModelServiceError(
                "Codex CLI executable was not found",
                error_code="codex_executable_missing",
            )
        executable = str(Path(executable).resolve())
        result["codex_executable"] = executable
        result["codex_cli_version"] = _codex_version(executable)

        control_root = Path(tempfile.gettempdir()) / "astra_codex_control"
        control_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="preflight_", dir=str(control_root)))
        result["isolated_working_directory"] = str(work_dir)
        session = CodexAppServerSession(
            executable=executable,
            cwd=work_dir,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
        identity = session.create_thread(PREFLIGHT_INSTRUCTIONS)
        actual_identity_matches = (
            identity.get("thread_id") == session.thread_id
            and identity.get("session_id") == session.session_id
            and isinstance(session.thread_id, str) and bool(session.thread_id.strip())
            and isinstance(session.session_id, str) and bool(session.session_id.strip())
        )
        if not actual_identity_matches:
            raise ModelServiceError(
                "preflight could not verify the actual thread/session response identities",
                error_code="preflight_thread_identity_mismatch",
            )
        result.update({
            "app_server_info": session.app_server_info,
            "app_server_capabilities": dict(session.app_server_capabilities),
            "thread_start_request": session.thread_start_request_metadata,
            "thread_start_response": session.thread_start_response_metadata,
            "thread_id_present_and_verified": True,
            "session_id_present_and_verified": True,
            "thread_id_sha256": hashlib.sha256(
                session.thread_id.encode("utf-8")
            ).hexdigest(),
            "session_id_sha256": hashlib.sha256(
                session.session_id.encode("utf-8")
            ).hexdigest(),
            "session_creation_count": 1,
            "app_server_rpc_request_count": session.rpc_request_count,
            "control_turn_count": session.turn_count,
            "last_protocol_error": session.last_protocol_error,
        })
        result["status"] = "thread_created_no_turn"
    except Exception as exc:
        error = exc
        result["error"] = safe_exception_record(exc)
    finally:
        if session is not None:
            if session.process is not None:
                process_pid = session.process.pid
                try:
                    process_group_id = os.getpgid(process_pid)
                except OSError:
                    process_group_id = None
            session.close(force=error is not None)
            result.update({
                "app_server_capabilities": dict(session.app_server_capabilities),
                "thread_start_request": session.thread_start_request_metadata,
                "thread_start_response": session.thread_start_response_metadata,
                "last_protocol_error": session.last_protocol_error,
                "session_creation_count": int(session.thread_id is not None),
                "app_server_rpc_request_count": session.rpc_request_count,
                "control_turn_count": session.turn_count,
                "process_pid": process_pid,
                "process_group_id": process_group_id,
                "process_group_exit_confirmed": session.process_group_exit_confirmed,
            })
            if session.turn_count != 0:
                result["status"] = "failed"
                result["turn_start_called"] = True
                result["error"] = {
                    "error_type": "PreflightInvariantError",
                    "error_code": "preflight_unexpected_turn",
                    "error_summary": "preflight must not start a control turn",
                }
            if session.process_group_exit_confirmed is not True:
                result["status"] = "failed"
                result["error"] = {
                    "error_type": "ProcessCleanupError",
                    "error_code": "preflight_process_exit_unconfirmed",
                    "error_summary": "preflight App Server process group exit was not confirmed",
                }
        if work_dir is not None:
            try:
                work_dir.rmdir()
                result["isolated_working_directory_removed"] = True
            except OSError:
                result["isolated_working_directory_removed"] = False

    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["model_inference_verified"] = False
    result["result_path"] = str(output_path)
    _write_result(output_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
                        help="new path for the sanitized preflight JSON (never overwritten)")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--reasoning", default="max")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--codex", default=None,
                        help="Codex executable path; defaults to PATH lookup")
    args = parser.parse_args(argv)
    try:
        result = run_preflight(
            args.output, args.model, args.reasoning, args.timeout, args.codex
        )
    except Exception as exc:
        print(json.dumps({
            "status": "failed_to_save_preflight_result",
            "error": safe_exception_record(exc),
        }, ensure_ascii=False))
        return 2
    print(json.dumps({
        "status": result["status"],
        "result_path": result["result_path"],
        "codex_cli_version": result.get("codex_cli_version"),
        "session_creation_count": result.get("session_creation_count", 0),
        "control_turn_count": result.get("control_turn_count", 0),
        "process_group_exit_confirmed": result.get(
            "process_group_exit_confirmed"
        ),
        "error": result.get("error"),
    }, ensure_ascii=False))
    return 0 if result["status"] == "thread_created_no_turn" else 1


if __name__ == "__main__":
    raise SystemExit(main())
