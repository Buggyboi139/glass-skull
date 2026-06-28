# Canvas Activation Map Design

Date: 2026-06-28
Repo: glass-skull
Target surface: top-level `Map` tab

## Goal

Implement the main Map tab as an interactive LLM activation map inspired by the provided guitar-neck reference image. The view should remap after each single message or batch by consuming the latest run artifact already written by the Run flow. The UI must be word-minimal in the graph itself: visible text is limited to layer labels such as `L0`, selected layer/group labels, and compact status badges. Details belong in hover/click tooltips and the diagnostic pane.

## Visual Direction

The graphic should follow the reference image:

- Dark glassmorphism surface with a bordered, translucent instrument-panel feel.
- Top overview graph with vertical frets for model layers.
- Dots inside each fret for node groups.
- Glowing horizontal activation paths that flow left to right from input toward output.
- Multiple batch paths layered together as a heatmap: stronger or repeated activations glow brighter.
- Warm/cool color progression across layers and batches, with inactive nodes dimmed.
- Middle and lower panes connected visually to the selected layer/group, like the reference's expanded rows and drilldown target.

The renderer will use embedded Canvas 2D through `st.components.v1.html`. This avoids adding a frontend build pipeline while still enabling custom drawing, hit testing, level-of-detail, hover tooltips, and drilldown interactions.

## Architecture

Add two modules:

- `glass_skull/activation_map.py`
  - Pure Python data shaping.
  - Builds a normalized activation-map payload from the latest run artifact, TransformerLens model summary, GGUF metadata, and any available trace rows.
  - Contains no Streamlit code.

- `glass_skull/activation_map_view.py`
  - Streamlit-facing render wrapper.
  - Emits HTML/CSS/JavaScript with a Canvas renderer.
  - Handles browser-side hover, click selection, drilldown, and tooltip display.

Wire the top-level `Map` tab in `main.py` to call the builder and renderer. Keep existing behavior scoring and model metadata views only where they support the new four-pane map; avoid broad refactors.

## Normalized Data Model

The payload shape is:

- `modelMeta`
  - `source`, `backend`, `modelName`, `architecture`
  - `layerCount`, `hiddenSize`, `attentionHeads`, `kvHeads`
  - `mlpIntermediateSize`, `vocabSize`, `contextLength`
  - `quantization`, `parameterCount`, `tensorCount`
  - `metadataSource`, `metadataErrors`

- `batches`
  - `batchId`, `promptId`, `label`, `promptPreview`
  - `tokenRange`, `outputToken`, `traceAvailable`

- `layers`
  - `layerId`, `index`, `name`, `layerType`
  - `nodeCount`, `groupCount`, `activationDensity`
  - `topActiveGroups`, `visualizationMode`

- `nodeGroups`
  - `groupId`, `layerId`, `name`
  - `sourceIndices`, `groupingMethod`
  - `activationValue`, `attributionScore`
  - `batchParticipation`, `visualizationMode`

- `activationPaths`
  - `pathId`, `batchId`, `promptId`
  - `points`: layer/group coordinates used by Canvas
  - `strength`, `frequency`, `tokenRange`, `outputToken`
  - `activationSummary`, `attributionScore`, `confidence`
  - `visualizationMode`

- `heatmapStats`
  - per-layer and per-group activation counts, max/mean activation, density, and batch participation

- `diagnostics`
  - selected batch, layer, node/group, activation value, attribution score, confidence/mode, source token, destination token, top contributing heads/features, metadata snapshot, capture timestamp

- `visualizationMode`
  - one of `exact`, `sampled`, `clustered`, `approximate`, `unavailable`

## Data Semantics

Layer count and node/group counts must derive from the loaded model, never from hardcoded architecture assumptions.

For TransformerLens:

- Use `model_summary(model)` for metadata.
- Use run artifact trace rows for layer/stream activation norms.
- Use cached `top_dims` where available for node/group drilldown.
- If raw per-dimension activation is not available in the artifact, mark overview groups as `clustered` or `sampled` rather than `exact`.

For local GGUF / llama.cpp:

- Use `local_gguf_context(...)` for metadata.
- Use `/glass-skull/info` and `/glass-skull/trace` data when available.
- If llama.cpp does not expose activation summaries, build metadata-only layers and mark activation paths as `unavailable`.
- If only token-level or layer-level summaries are available, mark paths as `approximate` or `sampled` according to the source fields.

Grouping target:

- Default to roughly 50 to 150 visible groups per layer.
- Adapt group count to viewport, layer hidden size, and available telemetry.
- Prefer meaningful grouping in this order: SAE feature grouping if present, activation similarity from `top_dims`, attention head grouping, MLP channel grouping, residual stream contribution, then index chunking as fallback.
- Fallback index chunking must be marked as `approximate` or `clustered`.

## Interaction Model

Canvas hit testing will support:

- Hover line: tooltip shows batch number, prompt/message id, token range, output token if available, activation summary, attribution/confidence, and visualization mode.
- Hover layer: tooltip shows layer index/name, type, activation density, top active groups, batch heat summary.
- Hover node/group: tooltip shows group name, source indices, activation value, attribution/contribution score, batch participation, exact/sampled/clustered status.
- Click layer: update Streamlit selection state and open expanded layer pane.
- Click group: drill into the group.
- Click highlighted node/group: drill further toward exact activated node/feature when available.

The highlighted path must remain visible across overview, expanded layer, and drilldown panes.

## Layout

The Map tab contains:

1. Top overview activation guitar-neck graph.
2. Middle expanded selected layer pane.
3. Lower drilldown/details pane.
4. Final diagnostic pane.

The diagnostic pane shows:

- selected batch
- selected layer
- selected node/group
- activation value
- attribution score
- confidence / visualization mode
- source token
- destination token
- top contributing heads/features
- model metadata snapshot
- capture timestamp

## Rendering And Performance

Use Canvas 2D with browser-side level-of-detail:

- Render only high-value or visible groups in overview.
- Keep inactive nodes dim.
- Use opacity and line width for heatmap strength.
- Precompute payload coordinates in JS after resize.
- Use requestAnimationFrame for hover redraws.
- Use spatial hit targets for lines, layers, and nodes to avoid scanning all raw dimensions.
- Cap visible overview groups based on viewport width and height.

The initial implementation does not add WebGL. The data model should remain renderer-agnostic so WebGL can replace Canvas later if large models need it.

## Testing

Use the repo's current script-based convention:

1. Add failing `smoke_check.py` assertions first for the normalized payload.
2. Verify the test fails before implementation.
3. Implement `activation_map.py` until smoke assertions pass.
4. Add renderer smoke checks that the HTML includes a canvas root, payload JSON, tooltip support, and selection hooks.
5. Run:
   - `PYTHONPYCACHEPREFIX=/tmp/glass-skull-pycache python -m compileall main.py glass_skull scripts smoke_check.py`
   - `python smoke_check.py`

Manual Streamlit verification is still required for visual behavior:

- `streamlit run main.py`
- Send a single message with tracing enabled.
- Run a batch with tracing enabled.
- Confirm the Map tab remaps, hover tooltips work, click drilldown updates, and unavailable modes are explicit.

## Worker Plan

After this spec is approved, use agents with disjoint ownership:

- Worker 1: normalized activation-map data model and smoke assertions.
- Worker 2: Canvas renderer module and reference-image visual styling.
- Worker 3: Map tab integration in `main.py`.
- Debugger: run compile/smoke checks, inspect failures, and report concrete fixes.
- Designator: review diffs against this spec and acceptance criteria, then close gaps.

Workers must not revert existing dirty changes. They must list changed files and coordinate around the current uncommitted work in `main.py`, `visuals.py`, `ui_local.py`, `run_artifacts.py`, and related modules.

## Acceptance Criteria

- Map remaps after every single message or batch because it consumes the latest run artifact.
- Layer count matches the loaded model metadata.
- Node/group counts derive from hidden size or available trace dimensions.
- Hover tooltips work for lines, layers, nodes, and groups.
- Clicking layers/groups drills down while preserving the active path.
- Multiple batches render together as a heatmap.
- Diagnostic pane updates from the selected graph element.
- No hardcoded model architecture values.
- UI remains responsive for large model metadata.
- Exact activation data is never faked; unavailable or approximate modes are explicit in payload, tooltip, and diagnostics.
