# Astra RLBench evaluator

`eval_astra.py` runs a closed loop absolute-pose policy through the existing
BridgeVLA++ RLBench action mode. The experiment is an **Astra-Direct-RGB**
policy: the model sees a language instruction, front/left-shoulder/
right-shoulder/wrist RGB, the current `Observation.gripper_pose`, and the
gripper-open flag. It does not see depth, point clouds, object poses, masks,
task internals, success-detector state, handles, or demonstration actions.
`reset_to_demo` still reads the selected demonstration to initialize an
episode. The model receives no demonstration actions or future trajectory.

This is a four-RGB-view plus robot-state direct-pose experiment. It does not
establish that Astra and point-cloud BridgeVLA receive identical information,
and its results must not be described as an input-matched comparison. Public
paper scores are separate from local results.

## Fixed experiment definition

The main task set is fixed and ordered:

1. `place_cups`
2. `place_shape_in_shape_sorter`
3. `put_groceries_in_cupboard`
4. `stack_blocks`
5. `stack_cups`

`meat_off_grill` is a separate sanity-check task and is never included in the
five-task macro average. A debug subset remains labeled `debug` and cannot
produce a full-suite macro average.

The policy emits one world-frame target per decision: XYZ in meters, absolute
unit quaternion XYZW, and gripper command (`0` close, `1` open). The reference
point is the RLBench arm tip used by the existing pose action mode. The action
mode moves the arm before applying the gripper command. Planner substeps do not
count as policy targets. The evaluator adds no action chunks, retries,
recovery actions, high-level skills, or cross-step memory. Simulation does not
advance while the policy is running.

The collision flag is `ignore_collisions`:

| Mode | Policy output | Applied flag | Comparison boundary |
| --- | --- | --- | --- |
| `fixed0` | 8 values | always `0` | Default; collision checking stays enabled. This is not the baseline's predictive collision action space. |
| `fixed1` | 8 values | always `1` | Planner is asked to ignore collisions. This is not the predictive action space. |
| `predict` | 9 values, including the flag | policy value `0` or `1` | Predictive flag is part of the policy action space. |

No mode changes automatically based on task, planner result, or model output.
Formal runs must state `--collision-mode` explicitly.

## Budgets, run modes, episodes, and repeats

Two explicit budget protocols are available:

- `uniform25`: 25 policy targets per selected task.
- `repo_step_limits`: reads `configs/eval_step_limit.yml` and saves the parsed
  table and final task budgets in `run_manifest.json`. At this revision,
  `place_cups` and `stack_blocks` resolve to 35; the other fixed main tasks
  resolve to 25.

`--max-waypoints` (also accepted as the old `--episode-length` spelling) is a
debug-only cap. It can lower a protocol budget but cannot raise it. There is no
implicit CLI-over-table precedence. Formal mode requires the entire ordered
five-task set, 25 episodes per task, 5 repeats, and no debug cap. Formal runs
are never started automatically.

Episode IDs can be selected with `--start-episode`, `--episode-ids`, or a JSON
file passed to `--episode-list` (`[0, 1, ...]` or `{"episode_ids": [...]}`).
The list length must match `--eval-episodes`. Repeat ID, demonstration episode
ID, and optional random seed are distinct fields. Repeats of the same episode
ID are repeated evaluations of the same demonstration scene, not new
independent scenes. Python and NumPy seeds are set when `--seed` is provided;
Astra does not set PyRep/CoppeliaSim's internal random seed and does not claim
full simulator or planner determinism.

## Model and policy identity

The real Codex policy defaults to `gpt-6-luna` with reasoning effort `max`;
both remain configurable. `requested_model`, response-reported
`resolved_model`, Codex CLI version, App Server `serverInfo`, and reasoning
effort are recorded separately. If the response does not identify a resolved
model, that field stays null. Mock and manual policies have no model identity.

### Episode-scoped native control session

The backend is Codex App Server, verified locally with Codex CLI `0.157.1`.
The implementation performs the JSON-RPC `initialize` handshake, starts one
non-ephemeral native thread with `thread/start`, and requires both the returned
`thread.id` and `thread.sessionId`. It writes these IDs to the episode session
manifest before returning the first action. Every policy decision sends a new
`turn/start` to that same thread, with the current text, four ordered
`localImage` inputs, model, reasoning effort, and a fresh action schema. It
accepts only a completed `agentMessage` with `phase: final_answer` after a
matching `turn/completed` event. Commentary and streaming deltas are never
action candidates. A repeated item ID is deduplicated across item completion
and the completed-turn snapshot; conflicting content or phase, or distinct
conflicting final messages, fails the turn. The installed App Server schema
allows a null/omitted phase for provider compatibility. Astra accepts that
legacy form only when the completed turn has exactly one unique phase-unknown
agent message and no explicit commentary, and records that fallback in the
turn metadata; ambiguous phase-less output is rejected.

One absolute monotonic deadline covers each control request from `turn/start`
write through its start response and matching completion event. Thread/App
Server initialization time is recorded separately. If startup times out before
a turn ID is confirmed, Astra does not interrupt a previous turn ID; it closes
the episode-owned App Server process group and records the end state as
unknown. After a confirmed turn timeout, an interrupt acknowledgement alone
does not count as cancellation: Astra waits for the matching `turn/completed`
event. A timed-out or otherwise unconfirmed turn cannot yield a late action.
The CLI resume path was not selected: local `codex exec resume --help` did not
provide the verified JSON, image, and structured-output flow required here.

Each episode gets a stable temporary control working directory outside the
repository and a new App Server process/thread; reset clears the application
binding and pending feedback. The app never uses `--last`, guesses a session,
resumes the development thread, or creates a replacement thread after a
control-turn error. It checks the returned instruction sources are empty and
sets read-only sandboxing, no approvals, and no dynamic tools. A detected tool
item or unknown item type fails the episode closed. Protocol-defined
`contextCompaction` items are recorded by thread, turn, item, and lifecycle
location and do not yield actions; the locally generated schema also exposes
the deprecated `thread/compacted` notification, which is recorded when seen.
Tool items remain rejected even in a turn that also compacts context. This is a
narrow verified boundary, not a claim of complete isolation from all user-level
or service-added context. The user's global Codex configuration is not changed.

The first prompt contains the control rules, task instruction, measured pose,
gripper state, step/budget, and the initial four images. Later turns add only
the latest four images and measured feedback for the immediately previous
action. Feedback is bound by `step_id`, `action_id`, and observation identity;
each turn record also has its own application control-request ID, distinct from
thread, turn, and action IDs. A failed new `turn/start` retains a null turn ID
and cannot rewrite a prior turn record. The next request is not sent until the
prior environment call and feedback record finish. Old images
and prompts are not replayed by the application. Native conversation history
is separate from `control_messages.jsonl`, which records the exact app-added
text/image references, final structured action, and allowlisted measured
feedback. Feedback preserves the model-requested quaternion from adapter
diagnostics, the normalized quaternion submitted to RLBench, and the measured
before/after orientations as separate values; a missing requested field is a
diagnostic error. It does not copy hidden reasoning or the native rollout. App Server
history may be compacted; compression events are recorded when observed, and
otherwise the manifest leaves that information unknown. Native history does
not prove that every earlier image remains available to the model.

Movement-size guidance is prompt-only. The action interface still accepts one
absolute target per decision and retains the existing numeric, quaternion,
gripper, collision, and workspace validation. No new motion clipping,
splitting, retry, or forced gripper rule was added. Codex may use tools despite
the prompt; the app rejects a reported tool event, but post-hoc event checking
is not a hardened security boundary.

## Failure classes, denominators, and timing

Each episode records planned/attempted/evaluable status, successes, failure
class, termination reason, action budget, counters, timings, and recording
status. Policy/task failures include invalid policy actions, detected tool
use, planning/IK failures, environment terminal failure, and budget exhaustion.
Infrastructure errors include initialization/data errors, missing fresh
observations, and model service/process failures. Recording errors are tracked
separately and do not invalidate a saved task result. The implementation
distinguishes the configured Codex inference timeout from service/process
errors; ambiguous errors remain unknown rather than being attributed to robot
capability.

Success uses the existing sparse RLBench convention (`reward > 99`, verified
against `eval.py`). Success rates are fractions in `[0, 1]`. Task summaries
show planned, attempted, evaluable, and successful counts. Infrastructure
failures make a run or task incomplete; any temporary rate retains its
explicit evaluable denominator. A main-suite macro average is emitted only
when all five tasks are completely covered for that repeat. Repeat mean and
sample standard deviation are null when not computable. Sanity-task results
are reported separately.

Counters distinguish policy decisions, App Server thread creation, turns and
requests, CLI version probes, environment target-action attempts, simulation
steps/time, policy time, turn time, action execution time, and episode
wall-clock time. The underlying model request count and unavailable token
fields remain null. Simulation time counts
successful `scene.step()` calls during action execution; video playback time
and video duration are separate. Episode wall-clock ends before video
encoding. Position and pose arrival checks are diagnostics only and never
replace RLBench task success.

## Dependencies and output layout

The standard entry point requires the same dedicated stacks selected by
`eval.sh`:

- `RLBENCH_SIM_STACK`, default
  `finetune/bridgevla/libs/RLBench_peract587`, containing `rlbench/`.
- `PYREP_SIM_STACK`, default
  `finetune/bridgevla/libs/PyRep_stepjam231`, containing a compiled
  `pyrep/backend/_sim_cffi*.so`.

Missing or uncompiled dedicated stacks fail explicitly. Shared dependencies
are allowed only with `--allow-shared-sim-stack`, which marks the run
nonstandard; formal mode disallows that option. If `rlbench` or `pyrep` has
already been imported from the wrong location, the evaluator asks for a fresh
process instead of changing paths and continuing.

Each run defaults to `<repository-root>/outputs/astra_<evaluation-id>`,
regardless of the launch directory. `--output-root` explicitly changes the
root; `--codex-work-root` and `--log-file` explicitly override their paths.
Any override that moves artifacts outside the default unified layout prints a
startup warning and is listed in the manifest. A stable, empty control working
directory outside the repository is used for the full episode and removed
when the episode closes, if empty. Codex's native conversation store remains
managed by Codex under the current user's configured storage; Astra records a
safe location and IDs but does not copy native session files into `outputs`.
Application evidence and episode artifacts remain under the current run.

Each run creates:

```text
outputs/astra_<evaluation-id>/
  run_manifest.json       # resolved protocol, code/dependency/model identity
  run_summary.json        # episode/task/repeat aggregates
  run.jsonl               # step and summary events
  policy_work/             # episode session manifest, inputs and bounded evidence
  episodes/<episode-run>/
    episode_summary.json
    episode_log.jsonl
    run_meta.json
    steps/step_NNN/        # policy RGB, accepted/rejected action, execution record
    video/                 # optional composite video and intermediate frames
```

The manifest's planned unit order matches evaluator execution order (task,
repeat, episode). Aggregate successes and denominators both use the same
evaluable episode rows. Infrastructure, unknown, missing, duplicate, or
conflicting rows make coverage incomplete; a success with `evaluable=false`
is excluded and reported as a data conflict. Sanity checks never enter the
five-task macro average, and incomplete repeats are not presented as complete
repeat means.

Policy output validation, tool-use rejection, inference deadlines, model or
CLI service errors, simulator errors, core artifact write errors, and unknown
errors use typed categories and error codes. Rejected model output is stored
separately from accepted actions; only allowlisted action fields are retained
when safe. Event and stderr evidence is bounded and redacted, with truncation
and redaction status recorded. Raw model output and hidden reasoning are not
stored. If safe redaction fails, the original diagnostic text is omitted.

Core records include policy inputs and metadata, the per-step canonical
`execution.json`, `episode_summary.json`, `run_manifest.json`, and
`run_summary.json`. A failed core write raises an evaluation error and leaves
the run incomplete where the remaining storage permits. `execution.json` and
`episode_summary.json` are the per-step and per-episode authoritative records;
`episode_log.jsonl` and `run_meta.json` are redundant convenience records.
Failures in `episode_log.jsonl` and `run_meta.json` are listed in the
authoritative episode summary. `run.jsonl` is also a convenience event stream;
write failures are retained in the run summary. Camera captures, annotated
frames, and MP4 encoding are optional visualization; their failures set
`recording_error` without changing a saved task result. Episode results are
committed before video encoding.

There is no environment execution watchdog in this revision. Planner/IK and
simulator calls run synchronously, so `--codex-timeout` covers an App Server
turn only and does not cover `eval_env.step()`. A native simulator call can
still block indefinitely; use external process supervision when running
unreviewed real episodes.

## Commands

Run `eval_astra.sh` to activate `bridgevla_plus_rlbench` when needed and set the
same CoppeliaSim/Qt/Xvfb paths used by `eval.sh`. Leave the default dedicated
simulator paths in place for standard runs; the Python entry point verifies
them before importing RLBench or PyRep.

The current single-task debug target is `open_drawer`. The drawer level, handle,
approach, and pull direction come from the task language and current images;
they are not hard-coded. The five-task formal set above remains unchanged.

Run the current open-drawer configuration (this is a real-model command and is
not run by the automated fake tests):

```bash
cd finetune/RLBench
bash eval_astra.sh \
  --policy codex \
  --run-mode debug \
  --tasks open_drawer \
  --eval-episodes 1 \
  --repeats 1 \
  --start-episode 0 \
  --budget-protocol uniform25 \
  --max-waypoints 25 \
  --collision-mode fixed0 \
  --codex-model gpt-6-luna \
  --codex-reasoning max \
  --codex-timeout 180 \
  --session-mode episode \
  --motion-prompt-profile adaptive_small_steps \
  --recording-width 1280 \
  --recording-height 720 \
  --recording-fps 20 \
  --record-video
```

`--max-waypoints 25` is a debug cap on this task's 25-target `uniform25`
budget. The output path is `outputs/astra_<evaluation-id>/` under the
repository. A run contains app-level policy evidence in `policy_work/`,
per-step execution records under `episodes/<episode-run>/steps/`, and the
four-view MP4 and frame timeline under that episode's `video/` directory.

Policy input resolution remains the existing square `IMAGE_SIZE` (recorded in
the manifest); it is independent of the video sensors. Video uses four new
recording-only PyRep vision sensors configured for native 1280×720 capture and
a 2560×1440 grid at 20 fps. For 16:9 capture, the horizontal field of view is
expanded to preserve the source camera's vertical field of view. This keeps the
views geometrically consistent without enlarging policy images. Before and
after frames are captured at the actual environment states; sampled motion
frames do not add simulation steps. Video time, simulation time, and episode
wall time are recorded separately. Camera/encoding failure is reported and
does not overwrite an already saved episode result.

The recording-only sensors do not enter policy messages, collision geometry,
task objects, or success checks. Their resolution and projection are recorded
in `run_meta.json`; the `video/frame_timeline.jsonl` file maps display frames
to steps, phases, and available simulation time.

Debug mock run (one task, one demo, two targets):

```bash
cd finetune/RLBench
bash eval_astra.sh \
  --policy mock --run-mode debug --tasks stack_cups \
  --budget-protocol uniform25 --eval-episodes 1 --repeats 1 \
  --start-episode 0 --max-waypoints 2 --collision-mode fixed0 \
  --no-record-video
```

Formal five-task command (documented only; do not use it for a smoke test):

```bash
cd finetune/RLBench
bash eval_astra.sh \
  --policy codex --run-mode formal \
  --tasks place_cups place_shape_in_shape_sorter put_groceries_in_cupboard stack_blocks stack_cups \
  --budget-protocol repo_step_limits \
  --eval-episodes 25 --repeats 5 --start-episode 0 \
  --collision-mode fixed0 --codex-model gpt-6-luna --codex-reasoning max
```

The real `open_drawer` command has not been run in this code/fake-verification
round. Fake App Server protocol tests and the fake evaluator validate request
identity, schema/image delivery, measured-feedback ordering, and new-thread
behavior; they do not prove that a real model remembers earlier turns or that
real 720p sensors render correctly in CoppeliaSim. The evaluator does not
automatically retry episodes or launch a formal batch.
