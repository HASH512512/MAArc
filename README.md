# MAArc

MaaFramework-based wrapper for `arcaea-auto-play`.

This project keeps the original parser/analyzer/solver code and adds MaaFramework project files plus a Python Agent adapter.

## Layout

- `assets/interface.json`: Maa ProjectInterfaceV2 entry.
- `assets/resource/pipeline/auto_play.json`: coarse Maa task flow.
- `agent/`: Python Agent custom actions and input backends.
- `autoplay/`, `algo/`, `control.py`, `easing.py`: copied core logic from `arcaea-auto-play`.
- `config/`: copied runtime configuration.

## First Task Flow

```text
SinglePlay -> LoadChart -> FindStartOrRetry -> ExecuteTouch -> Finished
```

`LoadChart` solves the touch sequence before START/retry is clicked. `ExecuteTouch` waits for the loading screen to disappear, then dispatches the cached event sequence.

## Input Backends

- `scrcpy`: uses the original `DeviceController.touch` path.
- `maa`: uses MaaFramework `post_touch_down/move/up` with pointer-to-contact mapping.

The first version defaults to `scrcpy` for maximum compatibility.

## Step 2: Agent Startup Test

Install dependencies first:

```powershell
python -m pip install -r requirements.txt
```

Then run the agent entry directly only to verify imports:

```powershell
python agent\main.py 0
```

Expected behavior without a Maa client: the process may block, fail to connect to socket `0`, or print Maa connection errors depending on MaaFw internals. That is acceptable for this direct invocation. The important failure to fix first is:

```text
ModuleNotFoundError: No module named 'maa'
```

That error means MaaFramework Python binding is not installed in the Python environment used to launch the agent. It happens before any custom action registration, so it does not mean registration succeeded.

For a real registration test, load `assets/interface.json` with a MaaFramework client/debugger. Successful registration is indicated by the client being able to start the agent and execute `PlaySong.LoadChart` / `PlaySong.ExecuteTouch` custom actions.

## Step 3: LoadChart Smoke Test

This test does not need Maa UI, ADB, OCR, screenshots, or a device. It verifies that the copied parser/solver/config chain works inside the Maa project:

```powershell
python agent\smoke_load_chart.py
```

Optional custom chart:

```powershell
python agent\smoke_load_chart.py "tests\samples\test_steganography.aff"
```

Expected output is a JSON object with non-zero `tick_count` and `event_count`.

## Step 4: Maa Task Dry Run

Use Maa UI/debugger to load `assets/interface.json`, then temporarily set the `ExecuteTouch` custom params in `assets/resource/pipeline/auto_play.json`:

```json
"dry_run": true,
"skip_loading_detection": true
```

This verifies Maa Pipeline ordering and custom action calls without waiting for loading detection and without sending touch input. It should still run `LoadChart`, OCR click, and `ExecuteTouch` dry-run logging.

If you do not want OCR to click during this test, temporarily replace `FindStartOrRetry` with a `DirectHit` + `DoNothing` node or run only the custom action from a Maa debugger if supported.

## Step 5: Input Backend Timing Test

After dry run succeeds, test real input with a short chart.

For scrcpy compatibility mode:

```json
"input_backend": "scrcpy",
"dry_run": false,
"skip_loading_detection": false
```

For Maa native input:

```json
"input_backend": "maa",
"maa_wait_mode": "wait_each",
"dry_run": false,
"skip_loading_detection": false
```

Timing logs are written to:

```text
debug/timing/*_execute_touch.jsonl
```

All `ExecuteTouch` trace timestamps in this file are relative milliseconds from `t_action_enter`, rounded to two decimal places. The raw origin is kept as `origin_perf_counter` in the first `action_enter` record.

Strict state-transition screenshots are written to:

```text
debug/state_trace/<run_id>/*_timeline.jsonl
debug/state_trace/<run_id>/*.jpg
```

Enable them with:

```json
"trace_state_snapshots": true,
"state_trace_scale": 1.0,
"state_trace_jpeg_quality": 92
```

For `start_detection_mode: "zigzag"`, the state trace includes screenshots for:

- `loading_screen_detected`
- `loading_screen_change_detected`

For `start_detection_mode: "freeze_change"`, the state trace includes screenshots for:

- `still_waiting_frame_detected`
- `still_waiting_frame_change_detected`

Each timeline record includes the screenshot path, screenshot timestamp, detector timestamp, screencap wait timing, detector timing, phase, and detector metrics. This is the strict trace used to verify whether the expected state transition matched the actual device video state.

Timing records also include `recognition_timing`:

- `frame_interval_ms`: interval from the previous sampled frame to the current sampled frame.
- `avg_frame_interval_ms`: average adjacent-frame interval during this detector loop.
- `avg_screencap_wait_ms`: average `post_screencap().wait()` cost.
- `avg_cached_image_ms`: average `cached_image` read cost.
- `avg_detector_ms`: average detector compute cost.
- `max_recognition_induced_delay_ms`: conservative estimate of recognition-introduced delay, calculated as average frame interval plus average screenshot/cached-image/detector costs.

Trigger records additionally include `trigger_latency`:

- `trigger_max_polling_delay_ms`: `trigger_frame_timestamp - trigger_previous_frame_timestamp`, the maximum polling delay for the exact trigger pair.
- `trigger_previous_frame_timestamp`: timestamp of the frame immediately before the trigger frame.
- `trigger_frame_timestamp`: timestamp of the frame that triggered the transition.

The visual polling loop is synchronous: each iteration waits for `post_screencap().wait()`, reads `cached_image`, then runs the detector. Therefore the practical polling rate is dominated by screenshot wait time. If screenshot wait averages about 20 ms, the theoretical ceiling is about 50 Hz before `cached_image`, OpenCV, logging, and optional frame tracing costs.

For low-latency timing-sensitive runs, prefer:

```json
"detection_poll_sleep_ms": 0,
"trace_state_snapshots": true,
"trace_frames": false
```

`trace_state_snapshots` only saves exact transition frames and is normally acceptable. `trace_frames` saves sampled frames for approximate replay and can add disk/encoding pressure; keep it off unless you need a fuller visual trace.

Continuous sampled frames are still optional and separate:

```json
"trace_frames": true,
"trace_frame_scale": 0.5,
"trace_jpeg_quality": 80
```

These frames are useful as an approximate video trace, but exact transition analysis should use `debug/state_trace`.

Continuous frames are grouped per run:

```text
debug/frames/<run_id>/*.jpg
```

Frame and state screenshot filenames include the relative millisecond timestamp, for example:

```text
20260508_120000_execute_touch_000001234.56ms_still_waiting_frame_detected.jpg
```

Compare these fields between backends:

- `loading_sample.t_screencap_wait_end - t_screencap_post_begin`
- `touch_dispatch.lateness_before_call_ms`
- `touch_dispatch.call_duration_ms`
- first touch dispatch timestamp versus scheduled due time

If Maa native input has large `call_duration_ms` or accumulating lateness, try `maa_wait_mode: "batch_tick"` and then `"none"` for comparison.

## Start Detection Modes

`ExecuteTouch.custom_action_param.start_detection_mode` supports two modes:

```json
"start_detection_mode": "zigzag"
```

Detects the Arcaea loading screen by its right-side bright/dark zigzag boundary. It triggers only after the loading boundary disappears for consecutive frames.

```json
"start_detection_mode": "freeze_change"
```

Does not detect semantic image features. After OCR clicks START/retry, it waits `start_detection_delay_ms`, confirms that the screen is visually still for several frames, then triggers at the first detected frame change.

Suggested freeze-change test parameters:

```json
{
  "start_detection_mode": "freeze_change",
  "start_detection_delay_ms": 100,
  "input_backend": "scrcpy",
  "debug_log": true
}
```

Compare `freeze_change_sample` and `loading_sample` records in `debug/timing/*.jsonl` to evaluate screenshot polling delay and detector processing cost.

## Offline Loading Detector Test

Put real loading screenshots under `tests/`, then run:

```powershell
python agent\smoke_loading_detector.py tests
```

Use one-frame confirmation for quick static screenshot checks:

```powershell
python agent\smoke_loading_detector.py tests --confirm-frames 1
```

Write JSONL metrics for later comparison:

```powershell
python agent\smoke_loading_detector.py tests --jsonl debug\loading_detector.jsonl
```

Important output fields:

- `phase`: detector state after this screenshot.
- `loading_seen`: this screenshot confirmed loading screen detection.
- `triggered`: this screenshot detected loading end.
- `metrics.brightness_ratio`: left/right brightness separation in the loading ROI.
- `metrics.largest_area_ratio`: largest OTSU contour area in the loading ROI.
- `metrics.changed_ratio`: frame difference ratio after entering monitoring.

If no screenshot reaches `loading_seen: true`, lower `--min-contour-area-ratio`, lower `--min-turns`, or adjust `--roi`. If `loading_seen` works but `triggered` never appears, lower `--frame-diff-ratio-threshold` or provide a post-loading screenshot after loading screenshots.
