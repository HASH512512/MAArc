# OpenCode Handoff Context

This file compresses the current MAArc conversation state so the work can be continued in another OpenCode TUI client.

## Project

- Workspace project: `D:\Code\MAArc`
- Original baseline project kept intact: `D:\Code\arcaea-auto-play`
- GitHub remote: `https://github.com/HASH512512/MAArc`
- Current branch: `main`
- Important: the worktree may contain local MaaSupport/config/sample-file changes that should not be committed unless explicitly requested.

## Goal

Migrate the Arcaea auto-play project into a MaaFramework project while preserving the original scrcpy touch backend. The Maa pipeline should load an AFF chart, visually detect song start/loading transition, then dispatch the complete generated touch sequence with strict timing traces.

## Design Constraints

- Do not rewrite the original `D:\Code\arcaea-auto-play` project.
- First implementation may use hardcoded local parameters.
- Preserve scrcpy touch path as the primary backend; Maa native touch remains optional.
- Maa native touch API uses:
  - `post_touch_down(x, y, contact=contact, pressure=1)`
  - `post_touch_move(x, y, contact=contact, pressure=1)`
  - `post_touch_up(contact=contact)`
- `assets/interface.json` uses `display_raw: true` to keep raw coordinates consistent.
- Pipeline order: `SinglePlay -> LoadChart -> FindStartOrRetry -> ExecuteTouch -> Finished`.
- Do not reuse old broken `autoplay/vision/detector.py`; the Maa path uses new detectors in `agent/loading_detector.py`.

## Important Files

- `agent/play_song.py`: Maa custom actions, schedule computation, loading detection loop integration, touch dispatch, timing/state tracing.
- `agent/loading_detector.py`: `LoadingEndDetector` and `FreezeChangeDetector`.
- `agent/touch_backends.py`: `ScrcpyTouchBackend`, `MaaTouchBackend`, `PointerMapper`.
- `agent/main.py`: Maa Agent entrypoint.
- `agent/smoke_load_chart.py`: offline AFF parse/solver smoke test.
- `control.py`: original scrcpy server/control socket implementation and video decoder.
- `assets/resource/pipeline/auto_play.json`: local pipeline parameters; may contain local absolute paths and should not be blindly committed.
- `assets/config/maa_pi_config.json`: local MaaSupport config; do not commit unless explicitly requested.
- `debug/timing/*_execute_touch.jsonl`: timing traces.
- `debug/state_trace/<run_id>/*_timeline.jsonl`: state transition traces and screenshots.

## Implemented So Far

- Created MAArc MaaFramework project structure.
- Copied core solver/config/control code from original auto-play project.
- Added Maa custom actions:
  - `PlaySong.LoadChart`
  - `PlaySong.ExecuteTouch`
- Added dual touch backends:
  - `input_backend: "scrcpy"`
  - `input_backend: "maa"`
- Added `FreezeChangeDetector` start detection mode.
- Added timing traces with relative milliseconds from `t_action_enter`.
- Added per-run debug dirs under `debug/frames/<run_id>` and `debug/state_trace/<run_id>`.
- Added key state snapshots:
  - `still_waiting_frame_detected`
  - `still_waiting_frame_change_detected`
  - loading screen state variants.
- Removed fixed 10 ms detection loop sleep; use `detection_poll_sleep_ms`, recommended `0`.
- Recommended low-latency settings: `trace_state_snapshots: true`, `trace_frames: false`.
- Added scrcpy touch backend prewarm in `agent/play_song.py` so `create_touch_backend()` happens before schedule sleep, not inside dispatch after sleeping.

## Latest Code Change

The most recent code change moved touch backend creation before schedule sleep.

Before:

```text
compute target_start
sleep until target_start
create ScrcpyTouchBackend / DeviceController
dispatch first touch late by several seconds
```

After:

```text
compute target_start
create/prewarm touch backend
log backend init timing
sleep remaining time
dispatch events using prewarmed backend
```

New trace records:

- `touch_backend_init_begin`
- `touch_backend_ready`
- `init_duration_ms`
- `remaining_ms`
- `first_touch_remaining_ms`

Verification run:

```powershell
python -m py_compile agent\play_song.py agent\touch_backends.py
```

passed with no output.

## Test Findings

### Test `20260509_193929_execute_touch`

Problem: first touch was very late.

Important trace values:

- `estimated_change_timestamp`: `6335.16ms`
- `fixed_delay_ms`: `3600`
- `first_tick_ms`: `1500`
- `target_start`: `9935.16ms`
- `first_touch_target`: `11435.16ms`
- `delay_sleep_end`: `9940.41ms`
- first touch `t_call_begin`: `14811.31ms`
- first touch `lateness_before_call_ms`: `3376.14ms`

Root cause: scrcpy backend was initialized after schedule sleep; `DeviceController()` startup consumed several seconds before first touch.

### Test `20260509_202847_execute_touch`

After prewarm change, first touch was fixed.

Important trace values:

- `touch_backend_ready.init_duration_ms`: `773.41ms`
- `first_touch_remaining_ms`: `4257.74ms`
- first touch `lateness_before_call_ms`: `0.07ms`

New issue: dispatch later stalled around tick `9000`.

Important trace values:

- tick `8500` late: `0.31ms`
- tick `9000` late: `6403.63ms`
- timing file ended without normal completed events.

Observed console:

```text
[server] ERROR: Capture/encoding error: java.io.IOException: android.system.ErrnoException: write failed: ECONNRESET (Connection reset by peer)
```

Current interpretation:

- The first-note multi-second delay is fixed.
- A separate mid-run stall/connection issue remains.
- `ECONNRESET` comes from scrcpy server video stream writing to a closed/reset socket. It may be a shutdown side effect, but with the incomplete timing trace it likely relates to the mid-run interruption.

## Timing/Latency Notes

- `screencap_wait_ms`: time waiting for Maa `post_screencap().wait()`.
- `cached_image_ms`: time reading Maa cached image into Python/OpenCV usable form.
- `total_ms`: total detector iteration time.
- `period_ms` / `frame_interval_ms`: actual visual polling interval between frames.
- Offset can compensate stable average delay, but cannot eliminate sampling uncertainty, screenshot timestamp ambiguity, USB/ADB jitter, Python scheduling jitter, or Android input injection jitter.
- With Maa screenshot polling, strict first-note guarantee inside Arcaea PM +/-25 ms is not guaranteed; average can be tuned, but jitter remains.
- Scrcpy video stream based detection is a likely future improvement for lower latency and better sampling.

## Recommended Runtime Parameters

For full real-device scrcpy touch test, use local pipeline params like:

```json
{
  "chart_path": "D:/Code/MAArc/tests/samples/steganography/3.aff",
  "input_backend": "scrcpy",
  "maa_wait_mode": "none",
  "start_detection_mode": "freeze_change",
  "start_detection_delay_ms": 0,
  "detection_poll_sleep_ms": 0,
  "loading_timeout_ms": 10000,
  "debug_log": true,
  "trace_state_snapshots": true,
  "state_trace_scale": 0.5,
  "state_trace_jpeg_quality": 80,
  "trace_frames": false,
  "dry_run": false,
  "skip_loading_detection": false
}
```

Do not assume these local params should be committed; they include local paths and test-specific flags.

## Current Git Hygiene Notes

Known local/unwanted worktree changes at the time of this handoff:

- `assets/config/maa_pi_config.json`: local MaaSupport config; do not commit.
- `assets/resource/pipeline/auto_play.json`: contains local testing params and absolute path; do not commit unless cleaned.
- `tests/samples/steganography_cut - 副本/...`: many deletions from local sample cleanup; do not commit unless explicitly requested.
- `tests/samples/steganography/`: untracked local sample chart; verify before committing.

Expected commit-worthy files from the latest work:

- `agent/play_song.py`
- `OPENCODE_CONTEXT.md`

## Next Work Steps

1. Add dispatch-loop diagnostics to locate the new mid-run stall:
   - `dispatch_loop_gap`
   - `tick_wait_begin`
   - `tick_wait_end`
   - `tick_dispatch_begin`
   - `tick_dispatch_end`
   - `backend_dispatch_error`
   - normal `touch_dispatch_finished` in timing log.
2. Detect long gaps inside `_dispatch_events()` with `time.perf_counter()` and log any gap > 100 ms.
3. Investigate whether scrcpy video decoding thread is disturbing touch timing or causing connection reset.
4. Consider a scrcpy control-only or no-decode touch backend, because current `DeviceController()` starts a video decoder even when only touch is needed.
5. Re-run full real-device task and inspect first touch plus mid-run gaps.
6. Only after dispatch is stable, tune `fixed_delay_ms` / `user_offset_ms` for first-note alignment.

## Commands Useful To Continue

```powershell
cd D:\Code\MAArc
python -m py_compile agent\play_song.py agent\touch_backends.py
python agent\smoke_load_chart.py
git status --short --branch
git diff -- agent/play_song.py OPENCODE_CONTEXT.md
```
