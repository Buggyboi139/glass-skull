# Task 2 Report

## Scope

- Implemented `glass_skull/activation_map_view.py`
- Added Task 2 smoke assertions in `smoke_check.py`

## TDD Evidence

### RED

1. Added the required import and smoke assertions from the Task 2 brief:
   - `from glass_skull.activation_map_view import activation_map_html`
   - HTML assertions for canvas, draw functions, interaction hooks, payload content, and `visualizationMode`
2. Ran:

```bash
.venv/bin/python smoke_check.py
```

3. Observed expected failure:

```text
Traceback (most recent call last):
  File "/home/dsmason321/repos/glass-skull/smoke_check.py", line 12, in <module>
    from glass_skull.activation_map_view import activation_map_html
ModuleNotFoundError: No module named 'glass_skull.activation_map_view'
```

### GREEN

1. Implemented `glass_skull/activation_map_view.py` with the required public API:
   - `activation_map_html(payload: dict, height: int = 960) -> str`
   - `render_activation_map(payload: dict, key: str = "activation_map_canvas", height: int = 960) -> None`
2. Added the required canvas script structure and helpers:
   - `drawOverview`
   - `drawLayerPane`
   - `drawDrilldownPane`
   - `drawDiagnostics`
   - `panel`
   - `drawPath`
   - `drawFret`
   - `drawGroupDot`
   - `drawSelectedTarget`
   - `drawDiagnosticText`
   - `tooltipHtml`
   - `addHitTarget`
3. Verified the full smoke check again:

```bash
.venv/bin/python smoke_check.py
```

4. Observed passing output:

```text
Glass Skull smoke check passed.
```

## Implementation Notes

- The renderer consumes `build_activation_map_payload(...)` output directly through an embedded JSON script payload.
- The canvas layout matches the brief’s required panel structure:
  - top overview panel
  - middle layer pane
  - lower drilldown pane
  - diagnostic pane
- Interaction support is included via canvas hit targets for:
  - layer selection
  - group selection
  - batch/path selection
  - hover tooltips
- The visual treatment uses a dark glassmorphism shell, dim overview node dots, and glowing path strokes.

## Changed Files

- `glass_skull/activation_map_view.py`
- `smoke_check.py`

## Commit Hygiene

- `smoke_check.py` already contained unrelated local edits in the workspace.
- Staged only the Task 2 import/assertion hunks from `smoke_check.py` for the Task 2 commit.

## Review Fixes

### Review finding 1: unavailable mode tooltip/diagnostic consistency

1. Added smoke assertions covering the unavailable renderer path:
   - `function effectiveVisualizationState` is present in the generated HTML
   - payload-wide `unavailable` handling is explicit in the renderer script
   - unavailable diagnostics include the unavailable reason row
2. Ran the smoke check after adding the failing assertions:

```bash
.venv/bin/python smoke_check.py
```

3. Observed the expected RED failure before the renderer fix:

```text
Traceback (most recent call last):
  File "/home/dsmason321/repos/glass-skull/smoke_check.py", line 719, in <module>
    main()
  File "/home/dsmason321/repos/glass-skull/smoke_check.py", line 556, in main
    assert 'function effectiveVisualizationState' in map_html
AssertionError
```

4. Fixed `glass_skull/activation_map_view.py` so payload-wide `visualizationMode == "unavailable"` overrides per-layer/per-group tooltip rows and diagnostics, and propagates `unavailableReason` when present.

### Review finding 2: visible Canvas text too verbose

1. Added smoke assertions to prevent the always-visible canvas labels from returning:
   - overview title
   - drilldown title
   - diagnostics title
   - model name header label
   - `Layer ...` header label
   - group count header text
   - selected batch text
   - destination token text
2. Removed those visible labels while preserving:
   - fret layer labels (`L0`, etc.)
   - selected layer/group identifiers
   - diagnostic field names and values

### Verification

Ran the required smoke check after the renderer fix:

```bash
.venv/bin/python smoke_check.py
```

Observed passing output:

```text
Glass Skull smoke check passed.
```
