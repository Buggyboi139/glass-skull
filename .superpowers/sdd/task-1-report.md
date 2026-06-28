# Task 1 Report: Normalized Activation-Map Payload

## Scope

- `glass_skull/activation_map.py`
- `smoke_check.py` Task 1 smoke additions only

## TDD Evidence

### RED

1. Added the required import and payload assertions from the task brief to `smoke_check.py`.
2. Ran `python smoke_check.py` from the repo root.
   - Result: `/bin/bash: line 1: python: command not found`
3. Switched to the repo interpreter and reran with `.venv/bin/python smoke_check.py`.
   - Result:

```text
Traceback (most recent call last):
  File "/home/dsmason321/repos/glass-skull/smoke_check.py", line 11, in <module>
    from glass_skull.activation_map import build_activation_map_payload, build_model_meta
ModuleNotFoundError: No module named 'glass_skull.activation_map'
```

This confirmed the new smoke coverage was failing for the missing Task 1 module before implementation.

### GREEN

1. Created `glass_skull/activation_map.py`.
2. Implemented:
   - `build_model_meta(summary, local_model_context, backend)`
   - `build_activation_map_payload(artifact, summary, local_model_context=None, selected_layer=None, selected_group=None, selected_batch=None)`
   - supporting helpers for group sizing, index chunking, batch rows, trace filtering, diagnostics, and heatmap stats
3. Ran `.venv/bin/python smoke_check.py`.
   - Result:

```text
Glass Skull smoke check passed.
```

## Changed Files

- `glass_skull/activation_map.py`
- `smoke_check.py`

## Notes

- The environment does not expose a `python` binary, so verification used the repo venv interpreter directly.
- `smoke_check.py` already had unrelated user changes in the worktree. I left them intact and staged only the Task 1 hunks for commit.

## Commit

- `c454c3d` - `Add activation map payload model`

## Review Fix Pass

### Findings Addressed

1. Activation-map payloads with only unavailable or otherwise unusable trace rows now report `visualizationMode: "unavailable"` at the top level and in diagnostics, and they preserve trace-row `unavailable_reason` when present.
2. `heatmapStats["groups"]` now aggregates grouped rows instead of collapsing each group to a single active/inactive sample, so repeated rows in the same group contribute to count, max, and mean.
3. `smoke_check.py` now covers both grouped heatmap aggregation and unavailable-mode payload behavior.

### Verification

#### RED

Command:

```text
.venv/bin/python smoke_check.py
```

Output:

```text
Traceback (most recent call last):
  File "/home/dsmason321/repos/glass-skull/smoke_check.py", line 695, in <module>
    main()
  File "/home/dsmason321/repos/glass-skull/smoke_check.py", line 549, in main
    assert layer_zero_group["activationCount"] == 3
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

#### GREEN

Command:

```text
.venv/bin/python smoke_check.py
```

Output:

```text
Glass Skull smoke check passed.
```

## Final Re-review Fix

### Finding Addressed

- Batch-level `traceAvailable` now matches activation-path usability instead of raw `trace_available` flags alone. A batch is marked available only when at least one trace row is not explicitly unavailable, has a concrete layer, and has a numeric `activation_norm`.

### Verification

#### RED

Command:

```text
.venv/bin/python smoke_check.py
```

Output:

```text
Traceback (most recent call last):
  File "/home/dsmason321/repos/glass-skull/smoke_check.py", line 696, in <module>
    main()
  File "/home/dsmason321/repos/glass-skull/smoke_check.py", line 589, in main
    assert unavailable_payload["batches"][0]["traceAvailable"] is False
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

#### GREEN

Command:

```text
.venv/bin/python smoke_check.py
```

Output:

```text
Glass Skull smoke check passed.
```
