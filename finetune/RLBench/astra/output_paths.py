"""Resolve Astra's persistent artifact layout independently of process cwd."""

from dataclasses import dataclass
from pathlib import Path


def _absolute(value):
    return Path(value).expanduser().absolute()


def _is_inside(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class OutputLayout:
    output_root: Path
    run_dir: Path
    policy_work_dir: Path
    run_log_path: Path
    external_overrides: tuple

    @property
    def standard(self):
        return not self.external_overrides

    def as_dict(self):
        return {
            "standard_layout": self.standard,
            "output_root": str(self.output_root),
            "run_directory": str(self.run_dir),
            "policy_work_directory": str(self.policy_work_dir),
            "run_log": str(self.run_log_path),
            "external_overrides": list(self.external_overrides),
        }


def resolve_output_layout(repository_root, evaluation_id, output_root=None,
                          codex_work_root=None, log_file=None):
    """Build `<root>/astra_<id>` and keep default artifacts inside that run."""
    repository_root = Path(repository_root).resolve()
    default_root = repository_root / "outputs"
    root = _absolute(output_root) if output_root else default_root
    run_dir = root / f"astra_{evaluation_id}"
    policy_work = (
        _absolute(codex_work_root)
        if codex_work_root else run_dir / "policy_work"
    )
    run_log = _absolute(log_file) if log_file else run_dir / "run.jsonl"
    external = []
    if root.resolve() != default_root.resolve():
        external.append("output_root")
    if not _is_inside(policy_work, run_dir):
        external.append("codex_work_root")
    if not _is_inside(run_log, run_dir):
        external.append("log_file")
    return OutputLayout(
        output_root=root,
        run_dir=run_dir,
        policy_work_dir=policy_work,
        run_log_path=run_log,
        external_overrides=tuple(external),
    )
