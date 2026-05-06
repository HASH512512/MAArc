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

Compare these fields between backends:

- `loading_sample.t_screencap_wait_end - t_screencap_post_begin`
- `touch_dispatch.lateness_before_call_ms`
- `touch_dispatch.call_duration_ms`
- first touch dispatch timestamp versus scheduled due time

If Maa native input has large `call_duration_ms` or accumulating lateness, try `maa_wait_mode: "batch_tick"` and then `"none"` for comparison.
