"""Runtime fingerprint for the interface facts used by interface_grounded_v1."""

import hashlib
import inspect
import subprocess
from pathlib import Path


ROBOT_INTERFACE_NOTES_VERSION = "panda_rlbench_interface_v1"

VERIFIED_DEPENDENCIES = {
    "rlbench": {
        "version": "1.2.0",
        "git_sha": "587a6a0e6dc8cd36612a208724eb275fe8cb4470",
    },
    "pyrep": {
        "version": "4.1.0.3",
        "git_sha": "231a1ac6b0a179cff53c1d403d379260b9f05f2f",
    },
}

VERIFIED_SOURCE_SHA256 = {
    "rlbench/action_modes/action_mode.py":
        "a3b9bf7fff8aa5f709012e2a1326d541cacfd20d4967d16358df3fdb4809569c",
    "rlbench/action_modes/arm_action_modes.py":
        "85d720485e84e8414d00c9b51c026b6248019db5f30319e7f0d8cba4945faa9b",
    "rlbench/action_modes/gripper_action_modes.py":
        "b5c2a95e7835dcc67f8736f35c812dcd2c0dbfed8111a8a1dbc15c4de81a7c52",
    "rlbench/backend/observation.py":
        "272a8ae5c2e54e2abbe8319e62c62c5a869eacb7e8b16cd0661f21c7443980b1",
    "rlbench/backend/scene.py":
        "581fe9d45c34f8484d8305e11de98380d872788c1ecbfbe132ce74d0cb6445a1",
    "rlbench/environment.py":
        "9ae042ec8de9bd601a82758e5b9d8e81b409526d42ea6d09aae263a0ab0238a6",
    "pyrep/objects/object.py":
        "7944481c73e7c8fe19fc043a0486bae500275eeed3b48db069b8da2900f3a37c",
    "pyrep/robots/arms/arm.py":
        "1010b5d446421bb2640894fd7a1a40795dc557d71c90fe394c247a1a1e863fd2",
    "pyrep/robots/arms/panda.py":
        "d7bef525367296f512e5aa63e1dd203935805480fc8b05bc638758396102f3d0",
    "pyrep/robots/end_effectors/panda_gripper.py":
        "802e6cdf1b77dd97a7c508fbe69c059f9e0ad05d04514f5abd918f89f5108563",
}

VERIFIED_SCENE_SHA256 = (
    "66e1cfa0a6ee5a5e635917d23ce1b5f8ba7159ee1a5326588d798030b972306a"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True,
            check=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_runtime_robot_interface_configuration(
        rlbench_module, pyrep_module, rlbench_environment_module,
        action_mode):
    """Capture loaded package, control-mode, source, and scene identities."""
    rlbench_package = Path(rlbench_module.__file__).resolve().parent
    pyrep_package = Path(pyrep_module.__file__).resolve().parent
    rlbench_root = rlbench_package.parent
    pyrep_root = pyrep_package.parent
    source_paths = {
        path: rlbench_root / path
        for path in VERIFIED_SOURCE_SHA256
        if path.startswith("rlbench/")
    }
    source_paths.update({
        path: pyrep_root / path
        for path in VERIFIED_SOURCE_SHA256
        if path.startswith("pyrep/")
    })
    source_hashes = {
        name: _sha256(path) if path.is_file() else None
        for name, path in source_paths.items()
    }
    scene_path = Path(rlbench_environment_module.DIR_PATH) / \
        rlbench_environment_module.TTT_FILE
    try:
        scene_hash = _sha256(scene_path)
    except OSError:
        scene_hash = None
    robot_setup = inspect.signature(
        rlbench_environment_module.Environment.__init__
    ).parameters["robot_setup"].default
    return {
        "robot_setup": robot_setup,
        "rlbench": {
            "file": str(Path(rlbench_module.__file__).resolve()),
            "version": getattr(rlbench_module, "__version__", None),
            "git_sha": _git_output(rlbench_root, "rev-parse", "HEAD"),
            "git_status_short": _git_output(rlbench_root, "status", "--short"),
        },
        "pyrep": {
            "file": str(Path(pyrep_module.__file__).resolve()),
            "version": getattr(pyrep_module, "__version__", None),
            "git_sha": _git_output(pyrep_root, "rev-parse", "HEAD"),
            "git_status_short": _git_output(pyrep_root, "status", "--short"),
        },
        "action_mode": {
            "combined_class": type(action_mode).__name__,
            "arm_class": type(action_mode.arm_action_mode).__name__,
            "absolute_mode": bool(action_mode.arm_action_mode._absolute_mode),
            "frame": action_mode.arm_action_mode._frame,
            "gripper_class": type(action_mode.gripper_action_mode).__name__,
        },
        "source_file_sha256": source_hashes,
        "scene_asset": {
            "path": str(scene_path.resolve()),
            "sha256": scene_hash,
        },
    }


def verify_runtime_robot_interface_configuration(configuration):
    """Refuse to send version-specific notes to an unverified simulator stack."""
    errors = []
    if configuration.get("robot_setup") != "panda":
        errors.append("robot_setup")
    if configuration.get("action_mode") != {
            "combined_class": "MoveArmThenGripper2",
            "arm_class": "EndEffectorPoseViaPlanning2",
            "absolute_mode": True,
            "frame": "world",
            "gripper_class": "Discrete",
    }:
        errors.append("action_mode")
    for dependency, expected in VERIFIED_DEPENDENCIES.items():
        actual = configuration.get(dependency) or {}
        for field, expected_value in expected.items():
            if actual.get(field) != expected_value:
                errors.append(f"{dependency}.{field}")
    actual_hashes = configuration.get("source_file_sha256") or {}
    for path, expected_hash in VERIFIED_SOURCE_SHA256.items():
        if actual_hashes.get(path) != expected_hash:
            errors.append(f"source:{path}")
    scene = configuration.get("scene_asset") or {}
    if scene.get("sha256") != VERIFIED_SCENE_SHA256:
        errors.append("scene_asset.sha256")
    if errors:
        raise RuntimeError(
            "interface_grounded_v1 requires the audited Panda/RLBench stack; "
            "runtime verification failed for: " + ", ".join(errors)
        )
    configuration["robot_interface_notes_version"] = ROBOT_INTERFACE_NOTES_VERSION
    configuration["robot_interface_verification"] = "verified"
    return configuration
