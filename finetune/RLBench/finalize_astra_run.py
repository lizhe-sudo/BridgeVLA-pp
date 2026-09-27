#!/usr/bin/env python3
"""Finalize a persisted Astra run after an external supervisor has stopped writing."""

import argparse
import json
from pathlib import Path

from astra.output_naming import finalize_persisted_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[2] / "outputs"),
    )
    args = parser.parse_args(argv)
    final_directory = finalize_persisted_run(args.output_root, args.evaluation_id)
    manifest = json.loads(
        (final_directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (final_directory / "run_summary.json").read_text(encoding="utf-8")
    )
    print(json.dumps({
        "kind": "run_directory_finalized",
        "evaluation_id": args.evaluation_id,
        "friendly_run_name": manifest.get("friendly_run_name"),
        "final_run_directory": str(final_directory),
        "completion_status": summary.get("completion_status"),
        "result": manifest.get("run_result"),
    }, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
