"""Strict RLBench/PyRep path selection used by the Astra entry point."""

import glob
import os
import sys
from pathlib import Path


def resolve_sim_roots(finetune_dir, environ=None, allow_shared=False):
    environ = os.environ if environ is None else environ
    finetune_dir = Path(finetune_dir).resolve()
    defaults = {
        "rlbench": finetune_dir / "bridgevla" / "libs" / "RLBench_peract587",
        "pyrep": finetune_dir / "bridgevla" / "libs" / "PyRep_stepjam231",
    }
    result = {}
    for key, env_name in (("rlbench", "RLBENCH_SIM_STACK"),
                          ("pyrep", "PYREP_SIM_STACK")):
        raw = environ.get(env_name)
        if raw == "":
            if not allow_shared:
                raise RuntimeError(
                    f"{env_name} is empty; shared simulator dependencies require "
                    "the explicit --allow-shared-sim-stack option"
                )
            result[key] = None
            continue
        root = Path(raw).expanduser().resolve() if raw else defaults[key]
        if key == "rlbench":
            valid = (root / "rlbench").is_dir()
            requirement = "a directory containing rlbench/"
        else:
            valid = bool(glob.glob(str(root / "pyrep" / "backend" / "_sim_cffi*.so")))
            requirement = "a compiled pyrep/backend/_sim_cffi*.so"
        if not valid:
            if allow_shared and raw is None:
                result[key] = None
                continue
            raise RuntimeError(
                f"{env_name} dependency is missing or incomplete at {root}; "
                f"expected {requirement}. Build the dedicated stack or explicitly "
                "use --allow-shared-sim-stack for a nonstandard run."
            )
        result[key] = root
    return result


def _module_path(module):
    value = getattr(module, "__file__", None)
    return Path(value).resolve() if value else None


def _under(path, root):
    if path is None or root is None:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def configure_project_paths(script_dir, finetune_dir, environ=None,
                            allow_shared=False, loaded_modules=None):
    """Insert paths in eval.sh precedence order and reject preloaded conflicts."""
    environ = os.environ if environ is None else environ
    roots = resolve_sim_roots(finetune_dir, environ, allow_shared=allow_shared)
    loaded_modules = sys.modules if loaded_modules is None else loaded_modules
    for name, key in (("rlbench", "rlbench"), ("pyrep", "pyrep")):
        module = loaded_modules.get(name)
        if module is not None and roots[key] is not None:
            if not _under(_module_path(module), roots[key]):
                raise RuntimeError(
                    f"{name} was already imported from {_module_path(module)}; "
                    f"the requested dedicated stack is {roots[key]}. Start a fresh process."
                )

    finetune_dir = Path(finetune_dir).resolve()
    script_dir = Path(script_dir).resolve()
    coppeliasim_root = Path(
        environ.get("COPPELIASIM_ROOT")
        or (finetune_dir / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04")
    ).expanduser().resolve()
    environ["COPPELIASIM_ROOT"] = str(coppeliasim_root)
    ld_paths = [part for part in environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    if str(coppeliasim_root) not in ld_paths:
        environ["LD_LIBRARY_PATH"] = os.pathsep.join([str(coppeliasim_root), *ld_paths])
    environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(coppeliasim_root)
    environ["QT_PLUGIN_PATH"] = (
        str(coppeliasim_root) + os.pathsep + "/usr/lib/x86_64-linux-gnu/qt5/plugins"
    )
    local_paths = [
        script_dir,
        finetune_dir,
        finetune_dir / "bridgevla" / "libs" / "point-renderer",
        finetune_dir / "bridgevla" / "libs" / "peract_colab",
        finetune_dir / "bridgevla" / "libs" / "YARR",
        finetune_dir / "bridgevla" / "libs" / "peract",
        finetune_dir / "GemBench",
        finetune_dir / "bridgevla" / "libs" / "PyRep",
    ]
    preferred = [roots["pyrep"], roots["rlbench"]] + local_paths
    normalized = []
    for candidate in preferred:
        if candidate is None or not candidate.exists():
            continue
        candidate = str(candidate.resolve())
        if candidate not in normalized:
            normalized.append(candidate)
    # Keep all unrelated entries after project-controlled roots, while avoiding
    # duplicates that could put the shared PyRep copy ahead of the dedicated one.
    remainder = [entry for entry in sys.path if entry and os.path.realpath(entry) not in normalized]
    sys.path[:] = normalized + remainder
    return {
        "rlbench_sim_stack": str(roots["rlbench"]) if roots["rlbench"] else None,
        "pyrep_sim_stack": str(roots["pyrep"]) if roots["pyrep"] else None,
        "shared_sim_stack_allowed": bool(allow_shared),
        "coppeliasim_root": str(coppeliasim_root),
        "coppeliasim_root_exists": coppeliasim_root.is_dir(),
        "sys_path_prefix": normalized,
    }


def module_identity(module):
    path = _module_path(module)
    version = getattr(module, "__version__", None)
    source_root = path.parent if path and path.name == "__init__.py" else (path.parent if path else None)
    git_sha = None
    if source_root is not None:
        current = source_root
        for _ in range(8):
            if (current / ".git").exists():
                import subprocess
                try:
                    git_sha = subprocess.run(
                        ["git", "-C", str(current), "rev-parse", "HEAD"],
                        check=True, capture_output=True, text=True, timeout=2,
                    ).stdout.strip()
                except (OSError, subprocess.SubprocessError):
                    git_sha = None
                break
            if current.parent == current:
                break
            current = current.parent
    return {"file": str(path) if path else None, "version": version,
            "git_sha": git_sha}
