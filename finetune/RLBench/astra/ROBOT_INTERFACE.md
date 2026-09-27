# Astra robot interface audit

This document records the evidence behind `interface_grounded_v1`. A
`confirmed` fact is limited to the loaded RLBench/PyRep/scene fingerprint below;
it is not a claim about another robot, scene asset, or simulator version.
Unresolved facts are kept out of the model-facing notes. Runtime object handles,
task-object state, and diagnostic files are never sent to the policy.

Paths written as `astra/...`, `utils/...`, and `eval_astra.py` below are relative
to `finetune/RLBench/`. RLBench and PyRep paths are relative to the runtime
dependency roots stated in the software table.

## Audit context and actual execution chain

At audit start the checkout was `/remote_userdata/lizhe/VLA/BridgeVLA/BridgeVLA-Astra`,
branch `astra-rlbench-eval`, HEAD
`d62c3af3afc488412a89d43337c74365675abca9`, clean, with `origin` set to
`https://github.com/lizhe-sudo/BridgeVLA-pp.git`. The active shell was Conda
base Python 3.12.9; the simulator process used
`/home/lizhe/miniconda3/envs/bridgevla_plus_rlbench/bin/python` 3.9.23. Codex
CLI reported `codex-cli 0.157.1`.

The actual control chain is:

`eval_astra.sh` → `eval_astra.py` → `AstraRLBenchEnv` and
`AstraObservationAdapter` → `AstraActionAdapter` →
`MoveArmThenGripper2(EndEffectorPoseViaPlanning2, Discrete)` → the selected
RLBench/PyRep modules below. The observation adapter copies raw
`Observation.gripper_pose`, four RGB images, and `gripper_open`; it does not
read task-object poses (`astra/observation_adapter.py:12-53`). The evaluator
instantiates this custom action mode at `eval_astra.py:646-657` and passes its
raw observation through `AstraRLBenchEnv` (`astra/env.py:6-40`).

The policy’s model action has position, quaternion, and gripper fields (plus
`ignore_collisions` only in `predict` mode). The Astra adapter packs the internal
9-value vector `[XYZ, XYZW, gripper, ignore_collisions]`
(`astra/action_adapter.py:36-98`). The custom combined action mode consumes the
trailing collision value; its inherited reported action shape counts the 7 arm
values plus 1 gripper value (`utils/rlbench_planning.py:29-68`; RLBench
`action_modes/action_mode.py:27-40`). This is existing interface behavior; this
change does not alter it.

### Loaded software and scene

| Component | Runtime identity | Checkout state and evidence |
| --- | --- | --- |
| RLBench | `rlbench.__file__ = /remote_userdata/lizhe/VLA/BridgeVLA/BridgeVLA-pp/finetune/bridgevla/libs/RLBench_peract587/rlbench/__init__.py`; version `1.2.0`; HEAD `587a6a0e6dc8cd36612a208724eb275fe8cb4470` | Worktree had pre-existing edits in `rlbench/environment.py`, `rlbench/utils.py`, and `setup.py`. `environment.py` adds optional `load_images`; the other changes add the matching demo-loading path and package entry. The robot-interface source `backend/scene.py` was clean. |
| PyRep | `pyrep.__file__ = /remote_userdata/lizhe/VLA/BridgeVLA/BridgeVLA-pp/finetune/bridgevla/libs/PyRep_stepjam231/pyrep/__init__.py`; version `4.1.0.3`; HEAD `231a1ac6b0a179cff53c1d403d379260b9f05f2f` | Worktree clean. |
| Scene asset | `.../RLBench_peract587/rlbench/task_design.ttt` | SHA-256 `66e1cfa0a6ee5a5e635917d23ce1b5f8ba7159ee1a5326588d798030b972306a`. This is the scene actually loaded during the audit. |
| CoppeliaSim | Root `.../CoppeliaSim_Edu_V4_1_0_Ubuntu20_04` | Simulator launch completed. The root name identifies the configured package; its executable’s version string was not separately queried. |
| Robot/action mode | `Panda` / `PandaGripper`; `MoveArmThenGripper2`, `EndEffectorPoseViaPlanning2`, `Discrete`; `absolute_mode=True`, `frame='world'` | Read from the loaded classes and action-mode instance. RLBench `Environment` defaults to `robot_setup='panda'` (`environment.py:30-42, 97-124`). |

The actual diagnostic was `outputs/interface_audit_20260927T044000Z/`.
It launched the dedicated simulator, started the scene, and read the Panda arm
base, `Panda_tip`, gripper root, joints, and robot-subtree geometry. It did not
load an RLBench task, request task observations, query task objects, call
`eval_env.step`, send a target action, call a demo API, or call a model. The
diagnostic reports `actions_sent=0`, `demo_calls=0`, `model_turns=0`, and
`simulator_shutdown_confirmed=true`; the external supervisor confirmed process
group `3884083` exited with status 0 and no owned processes remaining. The
captured simulation timestep was `0.05000000074505806` seconds.

An earlier isolated diagnostic attempt hit a recorder `TypeError` while
serializing a scalar action-shape value. Its failed record remains at
`outputs/interface_audit_20260927T043458Z/runtime_diagnostics.json`; that
process also shut down cleanly. The retry above used a new directory and did
not replace the failed record.

### Runtime source fingerprints

These hashes were read from the actual imported source files. The profile
validator checks these files, dependency versions/commits, action-mode values,
Panda setup, and scene asset before allowing `interface_grounded_v1` to run.
It fails closed on a mismatch. RLBench `environment.py` is intentionally
fingerprinted at its observed local-modified content; its checkout HEAD alone
does not establish its working-tree contents.

| Loaded file | SHA-256 |
| --- | --- |
| RLBench `action_modes/action_mode.py` | `a3b9bf7fff8aa5f709012e2a1326d541cacfd20d4967d16358df3fdb4809569c` |
| RLBench `action_modes/arm_action_modes.py` | `85d720485e84e8414d00c9b51c026b6248019db5f30319e7f0d8cba4945faa9b` |
| RLBench `action_modes/gripper_action_modes.py` | `b5c2a95e7835dcc67f8736f35c812dcd2c0dbfed8111a8a1dbc15c4de81a7c52` |
| RLBench `backend/observation.py` | `272a8ae5c2e54e2abbe8319e62c62c5a869eacb7e8b16cd0661f21c7443980b1` |
| RLBench `backend/scene.py` | `581fe9d45c34f8484d8305e11de98380d872788c1ecbfbe132ce74d0cb6445a1` |
| RLBench `environment.py` (locally modified) | `9ae042ec8de9bd601a82758e5b9d8e81b409526d42ea6d09aae263a0ab0238a6` |
| PyRep `objects/object.py` | `7944481c73e7c8fe19fc043a0486bae500275eeed3b48db069b8da2900f3a37c` |
| PyRep `robots/arms/arm.py` | `1010b5d446421bb2640894fd7a1a40795dc557d71c90fe394c247a1a1e863fd2` |
| PyRep `robots/arms/panda.py` | `d7bef525367296f512e5aa63e1dd203935805480fc8b05bc638758396102f3d0` |
| PyRep `robots/end_effectors/panda_gripper.py` | `802e6cdf1b77dd97a7c508fbe69c059f9e0ad05d04514f5abd918f89f5108563` |

## Facts and evidence

“Model: yes” means the concise statement is included in the fixed notes for
this verified profile. Numeric startup measurements and implementation details
stay in this developer record and run artifacts.

| Question | Confirmed conclusion | Evidence and exact reference | Scope | Model | Status |
| --- | --- | --- | --- | --- | --- |
| What does `Observation.gripper_pose` describe? | `Scene.get_observation()` reads `self.robot.arm.get_tip().get_pose()`. For this loaded Panda that object is `Panda_tip`, the IK tip dummy. PyRep `get_pose(None)` is absolute/world and returns XYZ + XYZW. | RLBench `backend/scene.py:170-172, 292-297`; PyRep `objects/object.py:225-236`; `robots/arms/arm.py:32-36, 450-457`; runtime snapshot. | Loaded Panda scene. | Yes: common measured/control reference. | confirmed |
| What coordinate frame and mode does the action use? | `EndEffectorPoseViaPlanning2` inherits `absolute_mode=True`, `frame='world'`; for world frame the planner passes `relative_to=None`. Its target is the same arm tip used as IK tip. | RLBench `action_modes/arm_action_modes.py:147-166, 190-196`; PyRep `robots/arms/arm.py:34-36, 450-457`; Astra `utils/rlbench_planning.py:71-86`; runtime action-mode instance. | Current custom mode and verified dependency fingerprint. | Yes: absolute world XYZ in meters and XYZW orientation. | confirmed |
| Do measured pose and target share a point and transform convention? | Yes: measured pose reads `Panda_tip`; path planning/IK uses `arm.get_tip()` / `_ik_tip`, the same dummy. The observation is copied in order XYZ + XYZW; action adapter sends XYZW to the world-frame pose mode. | RLBench `backend/scene.py:170-172, 292-297`; RLBench arm mode `:190-196`; PyRep `arm.py:32-36, 450-457`; Astra `observation_adapter.py:19-29`, `action_adapter.py:43-52`. | Audited Panda mode. | Yes. | confirmed |
| Are Panda-base axes identical to world axes? | No in the audited `task_design.ttt` startup. The measured base rotation matrix was `[[0,-0.156188,0.987727],[0,-0.987727,-0.156188],[1,0,0]]`; the loaded scene defines base placement. The base transform across other scene assets/configurations was not measured. | Runtime diagnostic JSON; RLBench `environment.py:91-124`; the scene asset SHA above. | This specific loaded scene only. | No exact base transform; notes say the scene sets it and axes need not align. | needs_runtime_verification across other scenes |
| Can a camera name identify a world direction? | No such relationship is established. The camera names are sensor identifiers; the runtime audit did not infer world axes from camera views. | Astra `observation_adapter.py:8-10, 31-42`; prompt template. | All camera-name interpretation. | Yes: do not infer world directions from names. | confirmed |
| What orientation does XYZW represent? | PyRep returns object quaternion relative to the requested frame; `None` means world. For this observation, the quaternion describes `Panda_tip` orientation relative to world. | PyRep `objects/object.py:187-198, 225-236`; RLBench `backend/scene.py:292-297`. | Audited pose chain. | Yes. | confirmed |
| How are tool axes encoded? | `get_matrix(None)` is the world transform. Its rotation columns are local +X, +Y, +Z expressed in world. A nontrivial runtime tip quaternion produced a rotation matrix within `3.03e-8` maximum absolute error of the PyRep matrix. | PyRep `objects/object.py:290-302`; runtime snapshot and numerical check; fixture `tests/fixtures/panda_tip_runtime_rotation.json`. | Verified loaded Panda, startup pose. | Yes: axes are explained without assigning world directions. | confirmed |
| How do those axes relate to the gripper? | At startup with normalized opening amount `[1,1]`, the measured vector from left to right finger-respondable centers in `Panda_tip` coordinates was `[-0.000092, 0.101051, -0.000063]` m. The two fingers separate along local Y. Robot-subtree transforms and local bounds show local Z runs from palm toward finger ends and local X is across finger thickness. | Runtime `panda_robot_subtree_objects` and finger-pair measurement; `PandaGripper` uses joints `Panda_gripper_joint1/2` (`pyrep/robots/end_effectors/panda_gripper.py:4-8`); test fixture. | This Panda geometry and audited open startup state. | Yes: qualitative axes only. | confirmed |
| Is there a fixed TCP-to-fingertip or pinch-point offset? | No constant fingertip/pinch-point offset was established. Finger spacing changes with gripper opening. The exact inner jaw clearance, contact point, and maximum usable opening were not independently measured by moving the gripper. | Runtime finger-pair/shape snapshot; PyRep `robots/end_effectors/gripper.py:74-100`; no actuation was permitted. | Do not extrapolate the open-state center spacing to closed or contact geometry. | No offset or maximum-width number. | unavailable |
| Are quaternion order and normalization consistent? | Observation is XYZ + XYZW; action schema/adapter uses XYZW; action adapter rejects non-finite and near-zero quaternions and normalizes other non-unit values before submit. | PyRep `objects/object.py:187-198, 225-236`; Astra `observation_adapter.py:19-29`, `action_adapter.py:43-52`; RLBench `arm_action_modes.py:190-192`. | Current action path. | Yes: XYZW, unit orientation on submit. | confirmed |
| Are `q` and `-q` treated as the same orientation? | Yes. Orientation error uses the absolute normalized quaternion dot product. The audit used a nontrivial quaternion; its matrix matched the runtime transform, and the numerical `q`/`-q` shortest-angle check returned 0 degrees. | Astra `visualization.py:24-37`; runtime diagnostic; `tests/test_astra_interface_grounded.py`. | Diagnostic comparison. | Not repeated as a special action target. | confirmed |
| What do gripper commands mean and when are they applied? | Astra accepts only `0` (close) or `1` (open). `MoveArmThenGripper2` calls arm planning, waits for low joint speed or the step cap, applies the discrete gripper command, then waits again. | Astra `action_adapter.py:53-60`; `utils/rlbench_planning.py:42-68`; RLBench `action_modes/gripper_action_modes.py:27-77`; PyRep `robots/end_effectors/gripper.py:74-100`. | This discrete Panda action mode. | Yes. | confirmed |
| Does settling mean arrival? | No. The wrapper checks max absolute joint velocity `<0.01 rad/s` for at most 100 scene steps; the call can stop at the cap. It does not compare `Panda_tip` with the requested target. | Astra `utils/rlbench_planning.py:9-17, 42-68`; runtime timestep `0.05 s`; evaluator reach calculation below. | Existing execution only. | Yes: settling is not arrival. | confirmed |
| What does observed `gripper_open` mean? | RLBench reports `1.0` only when the first normalized gripper opening amount is `>0.95`, otherwise `0.0`; Astra converts this to `bool`. This loses partial aperture and is not a generic object-held/contact sensor. | RLBench `backend/scene.py:289-297`; Astra `observation_adapter.py:44-53`; PyRep `robots/end_effectors/gripper.py:74-100`. | Audited RLBench source and Astra adapter. | Yes: boolean open state does not prove an object is held. | confirmed |
| What does action validation change? | The adapter validates finite XYZ/XYZW numbers, shape, and gripper; it normalizes a nonzero quaternion. It does not clip XYZ or split/retry an action. | Astra `action_adapter.py:20-98`. | Policy-to-environment adapter. | No extra numerical rule is needed in the notes. | confirmed |
| Where is XYZ clipped and what does it mean? | `EndEffectorPoseViaPlanning2.action()` clips XYZ to the scene workspace bounds inset by `1e-7`; orientation is not clipped. This is a position envelope, not proof that a pose is reachable or safe. `effective_planner_target` is Astra’s passive calculation matching that clip, not a planner-returned measurement. | Astra `utils/rlbench_planning.py:71-86`; `action_adapter.py:100-119`; evaluator records effective target/source at `eval_astra.py:985-987, 1032-1037`. | Current action mode/workspace. | Yes: scene bounds constrain position only. | confirmed |
| What path search actually occurs? | PyRep first tries a linear path, then a nonlinear path. With `ignore_collisions=False`, RLBench handles an already-colliding arm by disabling collision participation on selected non-robot, non-grasped colliding shapes. The action-mode function does not restore those flags before returning. If the path search raises `ConfigurationPathError`, RLBench retries `get_path()` once with `ignore_collisions=True`. | PyRep `robots/arms/arm.py:396-448`; RLBench `action_modes/arm_action_modes.py:198-253`. | Exact hashed RLBench/PyRep modules; whether a later reset changes those scene flags was not measured. | Yes, summarized briefly. | confirmed |
| What does `fixed0` guarantee? | Only that Astra submits `ignore_collisions=0`. It does **not** guarantee collision checks throughout the action because of the existing colliding-shape handling and unchecked fallback above. This behavior is dependency code, not an Astra-added retry. | Astra `action_adapter.py:62-98`; RLBench `arm_action_modes.py:198-249`; dynamic notes renderer. | The audited RLBench hash. | Yes, accurately qualified. | confirmed |
| What is `effective_planner_target`? | It is a passive Astra-side XYZ clip calculation based on current workspace bounds. It is distinct from raw requested XYZ, the normalized submitted action, and the measured after-pose. | Astra `action_adapter.py:77-119`; evaluator `eval_astra.py:985-987, 1032-1037`. | Current record implementation. | Yes: execution diagnostics distinguish target types. | confirmed |
| What does `environment_step_returned` mean? | A non-null evaluator transition was returned. RLBench’s wrapper can return a terminal transition after catching a planner/IK action exception, so this alone does not mean planning succeeded. | Evaluator `eval_astra.py:1012-1017, 1068-1076`; shared wrapper is read-only evidence. | Existing evaluator semantics. | No field expansion. | confirmed |
| What does `planner_returned` mean? | The evaluator observed no planner error and has a fresh `raw_after` observation. It is not a target-arrival or task-success claim. | Evaluator `eval_astra.py:977-984, 1012`; Astra `env.py:37-40`. | Existing evaluator semantics. | Yes, through diagnostics explanation. | confirmed |
| What do reach/error fields mean? | Position error to requested target is measured after-pose XYZ minus raw requested XYZ; error to effective target uses clipped XYZ. `position_reached` is error to effective target `< position_tolerance_m`; `pose_reached` additionally requires shortest quaternion orientation error `< orientation_tolerance_deg`. `target_reached` aliases `position_reached`. | Evaluator `eval_astra.py:985-1011, 1060-1067`; `visualization.py:24-37`. | Values come from parsed evaluator arguments. | Yes, with those actual values rendered per run. | confirmed |
| What do displacement fields mean? | Requested displacement is requested target XYZ minus before-pose XYZ; actual displacement is measured after-pose XYZ minus before-pose XYZ. Small movements should be judged using both numbers and images, not the tolerance flag alone. | `astra/execution_feedback.py:122-130`; evaluator action record. | Allowlisted feedback. | Yes. | confirmed |
| What is unavailable when there is no fresh after-observation? | `actual_eef_pose`, measured displacement/errors, and reached flags remain null/unavailable; no request target is substituted for an actual pose. | `astra/env.py:37-40`; evaluator `eval_astra.py:977-1012`; `execution_feedback.py:56-57, 114-132`. | Existing failure path. | Yes. | confirmed |
| Does model feedback include task success/reward/object truth? | No. The feedback builder allowlists measured robot state and execution diagnostics; it excludes reward, task-success internals, object truth, and raw environment data. | `astra/execution_feedback.py:43-55, 92-137`. | Existing feedback contract. | No added fields. | confirmed |

## Confirmed values, limits, and model selection

The runtime snapshot measured the `Panda` base pose as
`[-0.308951, 0, 0.820018, 0.704934, -0.055391, 0.704934, 0.055391]` in the
loaded scene, not at the world origin with identity orientation. This single
measurement does not establish an invariant base transform across another
scene, robot setup, or randomization configuration. No fixed world-axis
direction is assigned to the words front/left/right.

The open-startup finger-centroid separation is about `0.101051 m` along local
Y; it is not the inner jaw gap or a task grasp point. Joint positions were
`[0.04, 0.04]` m in intervals `[0, 0.04]`, with normalized opening `[1,1]`.
No gripper action was executed during this audit, so there is no closed-state
measurement or independent maximum usable opening measurement. The notes
describe axis directions and state semantics but provide no constant
tip-to-fingertip offset or aperture number.

The versioned model file is `robot_interface_notes_v1.txt`. It includes only
the same-world `Panda_tip` pose contract, XYZW/world orientation meaning,
qualitative Panda jaw axes, open-state limitation, arm-then-gripper ordering,
the current collision-mode behavior, scene-workspace clipping limitation, and
run-specific arrival thresholds. It excludes source paths/hashes, measured
startup coordinates, task/object facts, task strategies, extra sensors, and
diagnostic output paths. A dependency or scene fingerprint mismatch stops the
new profile before the first model turn; the note is not silently reused.

## Prompt profile and artifact records

`--motion-prompt-profile interface_grounded_v1` is additive; the old
`adaptive_small_steps` and `general_closed_loop_v1` template constants are
retained. The new initial template contains three parts: a task-neutral
controller preamble, the rendered robot-interface notes, and the requested
general control principles followed by current task/state and four current
images. The first-step approximately 0.01 m preference remains prompt-only.
The follow-up template contains the current instruction/state, the previous
allowlisted measured feedback, and the four newest images; native conversation
thread handling remains one thread per episode. No action limiting, splitting,
programmatic retries, task skills, policy image changes, or recording changes
were introduced.

The notes renderer replaces collision mode and both arrival thresholds from
the active evaluator arguments. It rejects missing/duplicate placeholders and
unrecognized leftovers. The run manifest, session manifest, per-turn metadata,
and prompt records carry:

- `motion_prompt_profile`, both template versions and hashes;
- notes version, source-template SHA-256, rendered-notes SHA-256, and full
  rendered notes;
- actual application-provided developer instructions and their hash;
- loaded robot/action/dependency configuration and configuration hash;
- parsed position/orientation tolerances and each exact prompt/hash.

`general_closed_loop_v1` remains an old traceable template and includes
drawer-specific wording; it is not the recommended cross-task profile. Its
template text is preserved. The dynamic collision line for existing profiles
was corrected so it no longer says that zero guarantees collision checking.

## Verification performed

The read-only runtime audit and unit tests verify source/config identity, the
nontrivial runtime quaternion-to-matrix conversion, local-Y finger separation,
`q`/`-q` error handling, task-independent prompt invariance, collision and
threshold rendering, fail-closed missing/mismatched inputs, one thread over
three feedback turns, one policy target per fake environment step, the existing
action-adapter behavior, four-image names/resolution, and recording dimensions.
The read-only diagnostic is interface evidence only; it is not a policy control
trial or a task-success test.
