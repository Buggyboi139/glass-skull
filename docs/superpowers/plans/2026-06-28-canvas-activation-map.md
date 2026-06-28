# Canvas Activation Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the top-level Map tab as a Canvas-based, guitar-neck activation map that remaps from the latest single-message or batch artifact.

**Architecture:** Add a pure Python activation-map payload builder and a Streamlit Canvas renderer wrapper. The Map tab becomes a thin integration layer that passes current artifact/model metadata into the builder, renders the Canvas component, and keeps existing behavior/model panels only as supporting diagnostics.

**Tech Stack:** Python 3.12, Streamlit, `st.components.v1.html`, embedded HTML/CSS/JavaScript Canvas 2D, pandas, existing `smoke_check.py` assertions.

## Global Constraints

- The graphic follows the provided reference: dark glassmorphism, vertical layer frets, dotted node groups, glowing left-to-right activation paths, stacked drilldown panes.
- Visible graph text is limited to layer labels such as `L0`, selected labels, and compact status badges; details go in tooltips or diagnostics.
- Renderer uses embedded Canvas 2D through `st.components.v1.html`; no frontend build pipeline.
- Layer count and node/group counts derive from loaded model metadata, not hardcoded architecture values.
- Visualization modes are exactly `exact`, `sampled`, `clustered`, `approximate`, `unavailable`.
- If exact activations are unavailable, payload/tooltips/diagnostics must say so instead of pretending.
- Existing dirty worktree changes are user work; do not revert them.

---

## File Structure

- Create `glass_skull/activation_map.py`
  - Normalized payload builder.
  - Pure Python, no Streamlit.
  - Public functions:
    - `build_model_meta(summary: dict, local_model_context: dict | None, backend: str) -> dict`
    - `build_activation_map_payload(artifact: dict, summary: dict, local_model_context: dict | None = None, selected_layer: int | None = None, selected_group: str | None = None, selected_batch: str | None = None) -> dict`

- Create `glass_skull/activation_map_view.py`
  - Streamlit render wrapper.
  - Public functions:
    - `activation_map_html(payload: dict, height: int = 960) -> str`
    - `render_activation_map(payload: dict, key: str = "activation_map_canvas", height: int = 960) -> None`

- Modify `smoke_check.py`
  - Add imports for the new public functions.
  - Add synthetic payload assertions before implementation.
  - Add renderer HTML assertions.

- Modify `main.py`
  - Import the new builder/renderer.
  - Replace the current top-level Map visualization body with the Canvas-first map while preserving supporting diagnostics and model metadata.

---

### Task 1: Normalized Activation-Map Payload

**Files:**
- Create: `glass_skull/activation_map.py`
- Modify: `smoke_check.py`

**Interfaces:**
- Consumes: `build_run_artifact(...)`, existing artifact dict shape, `summary` from `model_summary(...)`, optional `local_model_context(...)` dict.
- Produces: `build_model_meta(...) -> dict`, `build_activation_map_payload(...) -> dict`.

- [ ] **Step 1: Write failing smoke imports**

Add to `smoke_check.py` imports:

```python
from glass_skull.activation_map import build_activation_map_payload, build_model_meta
```

- [ ] **Step 2: Write failing payload assertions**

Add after the existing `path_rank_bar_fig(ranked_paths)` assertion:

```python
    map_summary = {
        "model_name": "tiny-trace",
        "device": "cpu",
        "layers": 3,
        "heads": 2,
        "d_model": 12,
        "d_head": 6,
        "d_mlp": 48,
        "vocab_size": 100,
        "parameters": 1234,
        "dtype": "torch.float32",
    }
    meta = build_model_meta(map_summary, None, "TransformerLens")
    assert meta["layerCount"] == 3
    assert meta["hiddenSize"] == 12
    assert meta["attentionHeads"] == 2
    assert meta["visualizationMode"] == "clustered"

    activation_payload = build_activation_map_payload(
        artifact,
        map_summary,
        local_model_context=None,
        selected_layer=1,
        selected_group="L1-G0",
        selected_batch="batch-8",
    )
    assert activation_payload["visualizationMode"] in {"sampled", "clustered", "approximate", "unavailable"}
    assert activation_payload["modelMeta"]["layerCount"] == 3
    assert [layer["name"] for layer in activation_payload["layers"]] == ["L0", "L1", "L2"]
    assert activation_payload["layers"][1]["selected"] is True
    assert activation_payload["nodeGroups"], "node groups should derive from hidden size or trace dimensions"
    assert activation_payload["activationPaths"], "available trace rows should produce activation paths"
    assert activation_payload["diagnostics"]["selectedLayer"]["name"] == "L1"
    assert activation_payload["diagnostics"]["selectedGroup"]["groupId"] == "L1-G0"
    assert activation_payload["diagnostics"]["modelMeta"]["hiddenSize"] == 12
    first_path = activation_payload["activationPaths"][0]
    assert {"batchId", "promptId", "points", "strength", "visualizationMode"}.issubset(first_path)
    assert first_path["points"], "paths should include per-layer Canvas points"
```

- [ ] **Step 3: Run smoke check to verify it fails**

Run: `python smoke_check.py`

Expected: fail with `ModuleNotFoundError: No module named 'glass_skull.activation_map'`.

- [ ] **Step 4: Implement `glass_skull/activation_map.py`**

Create the module with this structure:

```python
from __future__ import annotations

from datetime import datetime, timezone
from math import ceil
from typing import Any

import pandas as pd

from .run_artifacts import activation_path_df

VISUALIZATION_MODES = {"exact", "sampled", "clustered", "approximate", "unavailable"}


def _int_or_none(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _float_or_zero(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _preview(value: Any, limit: int = 72) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_model_meta(summary: dict, local_model_context: dict | None, backend: str) -> dict:
    local_model_context = local_model_context or {}
    is_local = bool(local_model_context)
    metadata = local_model_context.get("metadata") or {}
    layer_count = (
        _int_or_none(local_model_context.get("block_count"))
        if is_local
        else _int_or_none(summary.get("layers"))
    )
    hidden_size = (
        _int_or_none(local_model_context.get("embedding_length"))
        if is_local
        else _int_or_none(summary.get("d_model"))
    )
    heads = (
        _int_or_none(local_model_context.get("head_count"))
        if is_local
        else _int_or_none(summary.get("heads"))
    )
    context_length = (
        _int_or_none(local_model_context.get("context_length"))
        if is_local
        else _int_or_none(summary.get("context_length") or summary.get("n_ctx"))
    )
    return {
        "source": local_model_context.get("source") if is_local else "TransformerLens",
        "backend": backend,
        "modelName": local_model_context.get("display_name") if is_local else str(summary.get("model_name", "")),
        "architecture": local_model_context.get("architecture") if is_local else str(summary.get("architecture", "transformer")),
        "layerCount": max(layer_count or 0, 0),
        "hiddenSize": max(hidden_size or 0, 0),
        "attentionHeads": heads,
        "kvHeads": _int_or_none(local_model_context.get("head_count_kv")) if is_local else _int_or_none(summary.get("kv_heads")),
        "mlpIntermediateSize": _int_or_none(local_model_context.get("d_mlp")) if is_local else _int_or_none(summary.get("d_mlp")),
        "vocabSize": _int_or_none(metadata.get("tokenizer.ggml.tokens")) or _int_or_none(summary.get("vocab_size")),
        "contextLength": context_length,
        "quantization": str(metadata.get("general.file_type") or metadata.get("quantization") or ""),
        "parameterCount": _int_or_none(summary.get("parameters")),
        "tensorCount": _int_or_none(local_model_context.get("tensor_count")) if is_local else None,
        "metadataSource": "gguf" if is_local else "transformerlens",
        "metadataErrors": list(local_model_context.get("errors") or []),
        "visualizationMode": "clustered" if layer_count else "unavailable",
    }
```

Add these concrete helper functions below `build_model_meta(...)`:

```python
def _group_count(hidden_size: int, available_rows: int) -> int:
    if hidden_size <= 0:
        return 0
    target = max(50, min(150, available_rows or 96))
    return max(1, min(target, hidden_size))


def _source_indices(group_index: int, group_count: int, hidden_size: int) -> list[int]:
    if hidden_size <= 0 or group_count <= 0:
        return []
    chunk = max(1, ceil(hidden_size / group_count))
    start = group_index * chunk
    end = min(hidden_size, start + chunk)
    return list(range(start, end))


def _batch_rows(artifact: dict) -> list[dict]:
    rows = []
    for prompt in artifact.get("prompts", []):
        prompt_id = prompt.get("prompt_id", len(rows))
        rows.append({
            "batchId": f"batch-{prompt_id}",
            "promptId": prompt_id,
            "label": prompt.get("label") or "unlabeled",
            "promptPreview": _preview(prompt.get("prompt")),
            "tokenRange": "",
            "outputToken": _preview(prompt.get("output"), 32),
            "traceAvailable": any(row.get("trace_available", True) for row in prompt.get("trace_rows", [])),
        })
    return rows


def _available_path_df(artifact: dict) -> pd.DataFrame:
    df = activation_path_df(artifact)
    if df.empty:
        return df
    df = df[df["trace_available"] != False].copy()
    df["activation_norm"] = pd.to_numeric(df["activation_norm"], errors="coerce")
    return df.dropna(subset=["layer", "activation_norm"])
```

`build_activation_map_payload(...)` must then:

1. Call `build_model_meta(...)`.
2. Build `layers` as `L0` through `L{layerCount - 1}`.
3. Build groups per layer using `_group_count(...)` and `_source_indices(...)`.
4. Assign active group values from the strongest available trace rows by layer.
5. Build one activation path per prompt/batch with one point for each traced layer.
6. Build `heatmapStats` from grouped path rows.
7. Build `diagnostics` from the selected layer/group/batch, defaulting to the first available values.
8. Set top-level `visualizationMode` to `sampled` when trace rows exist, `unavailable` when no rows exist, or `clustered` when only grouped metadata exists.

- [ ] **Step 5: Run smoke check to verify Task 1 passes**

Run: `python smoke_check.py`

Expected: all assertions pass or fail only because renderer functions from Task 2 are not imported yet.

- [ ] **Step 6: Commit Task 1**

```bash
git add glass_skull/activation_map.py smoke_check.py
git commit -m "Add activation map payload model"
```

---

### Task 2: Canvas Renderer Module

**Files:**
- Create: `glass_skull/activation_map_view.py`
- Modify: `smoke_check.py`

**Interfaces:**
- Consumes: payload dict from `build_activation_map_payload(...)`.
- Produces: `activation_map_html(payload: dict, height: int = 960) -> str`, `render_activation_map(payload: dict, key: str = "activation_map_canvas", height: int = 960) -> None`.

- [ ] **Step 1: Write failing renderer imports**

Add to `smoke_check.py` imports:

```python
from glass_skull.activation_map_view import activation_map_html
```

- [ ] **Step 2: Write failing renderer assertions**

Add after the Task 1 payload assertions:

```python
    map_html = activation_map_html(activation_payload, height=720)
    assert '<canvas id="gs-activation-map"' in map_html
    assert 'function drawOverview' in map_html
    assert 'function drawLayerPane' in map_html
    assert 'function drawDrilldownPane' in map_html
    assert 'function drawDiagnostics' in map_html
    assert 'mousemove' in map_html
    assert 'click' in map_html
    assert '"layerCount": 3' in map_html
    assert 'visualizationMode' in map_html
```

- [ ] **Step 3: Run smoke check to verify it fails**

Run: `python smoke_check.py`

Expected: fail with `ModuleNotFoundError: No module named 'glass_skull.activation_map_view'`.

- [ ] **Step 4: Implement `activation_map_view.py`**

Create the module with:

```python
from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


def _json_script_payload(payload: dict[str, Any]) -> str:
    return html.escape(json.dumps(payload, ensure_ascii=False), quote=False)


def activation_map_html(payload: dict, height: int = 960) -> str:
    data = _json_script_payload(payload)
    return f"""
<div class="gs-map-shell" style="height:{int(height)}px">
  <canvas id="gs-activation-map"></canvas>
  <div id="gs-map-tooltip" class="gs-map-tooltip"></div>
  <script id="gs-map-data" type="application/json">{data}</script>
  <script>
  const payload = JSON.parse(document.getElementById('gs-map-data').textContent);
  const canvas = document.getElementById('gs-activation-map');
  const tooltip = document.getElementById('gs-map-tooltip');
  const ctx = canvas.getContext('2d');
  let hoverTarget = null;
  let selected = {{
    layerId: payload.diagnostics?.selectedLayer?.layerId || null,
    groupId: payload.diagnostics?.selectedGroup?.groupId || null,
    batchId: payload.diagnostics?.selectedBatch?.batchId || null
  }};
  let hitTargets = [];

  function resizeCanvas() {{
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawAll();
  }}

  function drawAll() {{
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    hitTargets = [];
    drawOverview(rect);
    drawLayerPane(rect);
    drawDrilldownPane(rect);
    drawDiagnostics(rect);
  }}

  function drawOverview(rect) {{
    /* Draw dark glass panel, layer frets, node dots, and glowing batch paths. */
  }}

  function drawLayerPane(rect) {{
    /* Draw expanded selected layer groups. */
  }}

  function drawDrilldownPane(rect) {{
    /* Draw selected group drilldown target and highlighted path continuation. */
  }}

  function drawDiagnostics(rect) {{
    /* Draw compact diagnostic pane; details still appear in tooltip data. */
  }}

  function findTarget(x, y) {{
    for (let i = hitTargets.length - 1; i >= 0; i--) {{
      const t = hitTargets[i];
      if (x >= t.x && x <= t.x + t.w && y >= t.y && y <= t.y + t.h) return t;
    }}
    return null;
  }}

  canvas.addEventListener('mousemove', (event) => {{
    const rect = canvas.getBoundingClientRect();
    hoverTarget = findTarget(event.clientX - rect.left, event.clientY - rect.top);
    if (hoverTarget) {{
      tooltip.style.display = 'block';
      tooltip.style.left = event.clientX - rect.left + 14 + 'px';
      tooltip.style.top = event.clientY - rect.top + 14 + 'px';
      tooltip.innerHTML = hoverTarget.tooltip;
    }} else {{
      tooltip.style.display = 'none';
    }}
    requestAnimationFrame(drawAll);
  }});

  canvas.addEventListener('click', () => {{
    if (!hoverTarget) return;
    if (hoverTarget.layerId) selected.layerId = hoverTarget.layerId;
    if (hoverTarget.groupId) selected.groupId = hoverTarget.groupId;
    if (hoverTarget.batchId) selected.batchId = hoverTarget.batchId;
    requestAnimationFrame(drawAll);
  }});

  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();
  </script>
</div>
"""


def render_activation_map(payload: dict, key: str = "activation_map_canvas", height: int = 960) -> None:
    components.html(activation_map_html(payload, height=height), height=height, scrolling=False)
```

Implement the draw functions with these exact responsibilities:

```javascript
function drawOverview(rect) {
  const x = 12, y = 10, w = rect.width - 24, h = Math.max(260, rect.height * 0.36);
  panel(x, y, w, h);
  const layers = payload.layers || [];
  const paths = payload.activationPaths || [];
  const left = x + 56, right = x + w - 56, top = y + 48, bottom = y + h - 54;
  paths.forEach((path, index) => drawPath(path, left, right, top, bottom, index));
  layers.forEach((layer, index) => drawFret(layer, index, layers.length, left, right, top, bottom));
}

function drawLayerPane(rect) {
  const x = 62, y = Math.max(300, rect.height * 0.41), w = rect.width - 124, h = 150;
  panel(x, y, w, h);
  const groups = (payload.nodeGroups || []).filter(g => g.layerId === selected.layerId).slice(0, 120);
  groups.forEach((group, index) => drawGroupDot(group, index, groups.length, x + 76, y + 36, w - 132, h - 70));
}

function drawDrilldownPane(rect) {
  const x = 62, y = Math.max(470, rect.height * 0.61), w = rect.width - 124, h = 150;
  panel(x, y, w, h);
  drawSelectedTarget(x + w / 2, y + h / 2, selected.groupId);
}

function drawDiagnostics(rect) {
  const x = 62, y = Math.max(640, rect.height * 0.80), w = rect.width - 124, h = rect.height - y - 16;
  panel(x, y, w, Math.max(110, h));
  const d = payload.diagnostics || {};
  drawDiagnosticText(x + 18, y + 28, d);
}
```

Support these helper functions in the same script: `panel`, `drawPath`, `drawFret`, `drawGroupDot`, `drawSelectedTarget`, `drawDiagnosticText`, `tooltipHtml`, and `addHitTarget`. The visible labels inside the Canvas are limited to layer names, selected group/layer identifiers, compact mode text, and diagnostic field names.

- [ ] **Step 5: Run smoke check to verify Task 2 passes**

Run: `python smoke_check.py`

Expected: all assertions pass before UI wiring.

- [ ] **Step 6: Commit Task 2**

```bash
git add glass_skull/activation_map_view.py smoke_check.py
git commit -m "Add canvas activation map renderer"
```

---

### Task 3: Top-Level Map Tab Integration

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `latest_behavior_artifact()`, `summary`, `local_model_context`, `build_activation_map_payload(...)`, `render_activation_map(...)`.
- Produces: Canvas-first Map tab UI.

- [ ] **Step 1: Add imports**

Add near other imports in `main.py`:

```python
from glass_skull.activation_map import build_activation_map_payload
from glass_skull.activation_map_view import render_activation_map
```

- [ ] **Step 2: Replace Map tab visualization body**

Inside `with tab_dash:`, after `artifact = latest_behavior_artifact()`, build and render:

```python
        activation_payload = build_activation_map_payload(
            artifact,
            summary,
            local_model_context=local_model_context,
            selected_layer=st.session_state.get("map_selected_layer"),
            selected_group=st.session_state.get("map_selected_group"),
            selected_batch=st.session_state.get("map_selected_batch"),
        )
        render_activation_map(
            activation_payload,
            key=f"activation_map_canvas_{artifact.get('run_id', 'empty')}",
            height=960,
        )
```

Keep the empty state when no artifact exists. Keep the model metadata section below the Canvas map. Move existing behavior score tables and Plotly heatmaps into an expander named `Legacy map tables` so the Canvas graphic is the main Map tab graphic.

- [ ] **Step 3: Preserve unavailable mode**

When `available_path.empty`, still render `activation_payload` if model metadata exists. The Canvas must show layers and an unavailable/approximate diagnostic pane rather than an empty fake activation path.

- [ ] **Step 4: Run compile check**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/glass-skull-pycache python -m compileall main.py glass_skull scripts smoke_check.py
```

Expected: command exits 0.

- [ ] **Step 5: Run smoke check**

Run: `python smoke_check.py`

Expected: command exits 0.

- [ ] **Step 6: Commit Task 3**

```bash
git add main.py
git commit -m "Render canvas activation map in Map tab"
```

---

### Task 4: Debugger Verification

**Files:**
- Modify only files needed to fix concrete failures discovered by verification.

**Interfaces:**
- Consumes: Tasks 1-3 implementation.
- Produces: passing compile and smoke checks plus a short failure/fix report.

- [ ] **Step 1: Run compile check**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/glass-skull-pycache python -m compileall main.py glass_skull scripts smoke_check.py
```

Expected: command exits 0.

- [ ] **Step 2: Run smoke check**

Run: `python smoke_check.py`

Expected: command exits 0.

- [ ] **Step 3: Inspect Map HTML output**

Run a small Python snippet:

```bash
python - <<'PY'
from glass_skull.activation_map import build_activation_map_payload
from glass_skull.activation_map_view import activation_map_html

artifact = {
    "run_id": "debug",
    "mode": "Batch run",
    "backend": "transformerlens",
    "model": "tiny",
    "created_at": "2026-06-28T00:00:00+00:00",
    "summary": {"prompt_count": 1, "trace_supported": True},
    "prompts": [{
        "prompt_id": 0,
        "label": "debug",
        "prompt": "hello",
        "output": "world",
        "trace_rows": [{
            "run_id": "debug",
            "prompt_id": 0,
            "label": "debug",
            "layer": 0,
            "stream": "resid_post",
            "component": "resid_post",
            "token_index": 0,
            "token": "hello",
            "activation_norm": 1.0,
            "trace_available": True,
            "trace_source": "transformerlens",
            "unavailable_reason": "",
            "top_dims": [{"dimension": 1, "activation": 0.5}],
        }],
    }],
}
summary = {"model_name": "tiny", "layers": 2, "d_model": 8, "heads": 2, "d_mlp": 32, "vocab_size": 100, "parameters": 1000}
payload = build_activation_map_payload(artifact, summary)
html = activation_map_html(payload)
assert "gs-activation-map" in html
assert "drawOverview" in html
assert "L0" in html
print("canvas html ok")
PY
```

Expected output: `canvas html ok`.

- [ ] **Step 4: Fix failures with minimal patches**

If a command fails, patch only the file implicated by the failure. Do not refactor unrelated modules. Re-run the failing command after each fix.

- [ ] **Step 5: Commit debugger fixes**

If fixes were needed:

```bash
git add glass_skull/activation_map.py glass_skull/activation_map_view.py main.py smoke_check.py
git commit -m "Fix canvas activation map verification issues"
```

If no fixes were needed, do not create an empty commit.

---

### Task 5: Designator Acceptance Review

**Files:**
- Modify only files needed to close acceptance gaps.

**Interfaces:**
- Consumes: implemented feature and design spec.
- Produces: acceptance checklist result and final verification.

- [ ] **Step 1: Compare implementation to spec**

Open:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-06-28-activation-map-canvas-design.md
```

Check each acceptance criterion against the implemented modules.

- [ ] **Step 2: Verify no hardcoded architecture values**

Run:

```bash
rg -n "93|2048|4096|151936|Qwen|Llama|n_layer|d_model" glass_skull/activation_map.py glass_skull/activation_map_view.py main.py
```

Expected: any matches are either metadata field names, existing app model labels outside the new map logic, or test fixture values in `smoke_check.py`; no new map layer/node counts are hardcoded.

- [ ] **Step 3: Verify visible labels stay minimal**

Run:

```bash
rg -n "Overview|Group of|Nodes per|activation summary|batch number" glass_skull/activation_map_view.py
```

Expected: explanatory phrases appear in tooltips/diagnostics data or comments, not as always-visible Canvas labels except compact pane/status labels.

- [ ] **Step 4: Run final verification**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/glass-skull-pycache python -m compileall main.py glass_skull scripts smoke_check.py
python smoke_check.py
```

Expected: both commands exit 0.

- [ ] **Step 5: Commit final acceptance fixes**

If fixes were needed:

```bash
git add glass_skull/activation_map.py glass_skull/activation_map_view.py main.py smoke_check.py
git commit -m "Polish canvas activation map acceptance"
```

If no fixes were needed, do not create an empty commit.
