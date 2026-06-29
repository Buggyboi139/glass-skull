from __future__ import annotations

import html
import json
from typing import Any


def _json_script_payload(payload: dict[str, Any]) -> str:
    return html.escape(
        json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"),
        quote=False,
    )


def activation_map_html(payload: dict, height: int = 1920) -> str:
    data = _json_script_payload(payload)
    return f"""
<div class="gs-map-shell" style="height:{int(height)}px">
  <style>
    .gs-map-shell {{
      position: relative;
      width: 100%;
      min-height: 320px;
      border-radius: 18px;
      overflow: hidden;
      background:
        radial-gradient(circle at top, rgba(87, 204, 255, 0.20), transparent 28%),
        linear-gradient(180deg, rgba(12, 17, 31, 0.98), rgba(8, 11, 22, 0.98));
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.08),
        inset 0 -1px 0 rgba(111, 164, 255, 0.08),
        0 24px 64px rgba(0, 0, 0, 0.35);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #gs-activation-map {{
      width: 100%;
      height: 100%;
      display: block;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.00));
    }}
    .gs-map-controls {{
      position: absolute;
      z-index: 4;
      left: 12px;
      right: 12px;
      top: 10px;
      display: grid;
      grid-template-columns: repeat(5, minmax(112px, 1fr));
      gap: 8px;
      padding: 8px;
      border: 1px solid rgba(146, 188, 255, 0.20);
      border-radius: 8px;
      background: rgba(6, 10, 20, 0.76);
      backdrop-filter: blur(14px);
      color: rgba(228, 238, 255, 0.92);
      font-size: 11px;
    }}
    .gs-map-controls label {{
      display: grid;
      gap: 3px;
      min-width: 0;
    }}
    .gs-map-controls span {{
      color: rgba(169, 185, 214, 0.82);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .gs-map-controls select,
    .gs-map-controls input {{
      min-width: 0;
      height: 24px;
      border: 1px solid rgba(146, 188, 255, 0.28);
      border-radius: 6px;
      background: rgba(10, 16, 29, 0.92);
      color: rgba(241, 246, 255, 0.96);
      font: inherit;
    }}
    .gs-map-controls .gs-check {{
      grid-template-columns: 18px minmax(0, 1fr);
      align-items: center;
      gap: 6px;
    }}
    .gs-map-controls .gs-check input {{
      width: 16px;
      height: 16px;
    }}
    .gs-map-tooltip {{
      position: absolute;
      z-index: 5;
      display: none;
      min-width: 160px;
      max-width: 280px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(146, 188, 255, 0.28);
      background: rgba(7, 11, 22, 0.92);
      color: rgba(238, 245, 255, 0.96);
      backdrop-filter: blur(16px);
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.42);
      pointer-events: none;
      font-size: 12px;
      line-height: 1.45;
    }}
    .gs-map-tooltip .gs-tip-title {{
      margin-bottom: 6px;
      color: rgba(142, 219, 255, 0.95);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .gs-map-tooltip .gs-tip-row {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      white-space: normal;
    }}
    .gs-map-tooltip .gs-tip-key {{
      color: rgba(167, 180, 208, 0.80);
    }}
    .gs-map-tooltip .gs-tip-value {{
      color: rgba(244, 248, 255, 0.96);
      text-align: right;
      overflow-wrap: anywhere;
    }}
  </style>
  <div class="gs-map-controls">
    <label><span>Mode</span><select id="visualization-mode"></select></label>
    <label><span>Prompt</span><select id="selected-prompt"></select></label>
    <label><span>Token</span><select id="selected-token"></select></label>
    <label><span>Top K</span><input id="top-k" type="range" min="1" max="32" step="1"></label>
    <label><span>Background</span><input id="background-opacity" type="range" min="0" max="1" step="0.05"></label>
    <label><span>Edge Threshold</span><input id="edge-threshold" type="range" min="0" max="1" step="0.05"></label>
    <label class="gs-check"><input id="show-aggregate" type="checkbox"><span>Show aggregate heatmap</span></label>
    <label class="gs-check"><input id="show-secondary" type="checkbox"><span>Show secondary branches</span></label>
    <label class="gs-check"><input id="developer-diagnostics" type="checkbox"><span>Developer diagnostics</span></label>
  </div>
  <canvas id="gs-activation-map"></canvas>
  <div id="gs-map-tooltip" class="gs-map-tooltip"></div>
  <script id="gs-map-data" type="application/json">{data}</script>
  <script>
  const payload = JSON.parse(document.getElementById('gs-map-data').textContent);
  const canvas = document.getElementById('gs-activation-map');
  const tooltip = document.getElementById('gs-map-tooltip');
  const ctx = canvas.getContext('2d');
  const shell = canvas.parentElement;
  const controls = {{
    mode: document.getElementById('visualization-mode'),
    prompt: document.getElementById('selected-prompt'),
    token: document.getElementById('selected-token'),
    topK: document.getElementById('top-k'),
    backgroundOpacity: document.getElementById('background-opacity'),
    edgeThreshold: document.getElementById('edge-threshold'),
    showAggregate: document.getElementById('show-aggregate'),
    showSecondary: document.getElementById('show-secondary'),
    developerDiagnostics: document.getElementById('developer-diagnostics'),
  }};
  const rendererOptions = {{
    visualizationMode: payload.rendererOptions?.visualizationMode || payload.visualizationMode || 'aggregate_heatmap',
    selectedPromptId: payload.rendererOptions?.selectedPromptId ?? null,
    selectedTokenId: payload.rendererOptions?.selectedTokenId ?? null,
    topK: Number(payload.rendererOptions?.topK ?? 8),
    backgroundOpacity: Number(payload.rendererOptions?.backgroundOpacity ?? 0.24),
    edgeThreshold: Number(payload.rendererOptions?.edgeThreshold ?? 0),
    showAggregateHeatmap: Boolean(payload.rendererOptions?.showAggregateHeatmap),
    showSecondaryBranches: payload.rendererOptions?.showSecondaryBranches !== false,
    developerDiagnostics: Boolean(payload.rendererOptions?.developerDiagnostics),
  }};
  let hoverTarget = null;
  let selected = {{
    layerId: payload.diagnostics?.selectedLayer?.layerId || payload.layers?.[0]?.layerId || null,
    groupId: payload.diagnostics?.selectedGroup?.groupId || payload.nodeGroups?.[0]?.groupId || null,
    batchId: payload.diagnostics?.selectedBatch?.batchId || payload.activationPaths?.[0]?.batchId || payload.batches?.[0]?.batchId || null,
    edgeId: null
  }};
  let hitTargets = [];

  function option(select, value, label) {{
    const node = document.createElement('option');
    node.value = String(value ?? '');
    node.textContent = String(label ?? value ?? '');
    select.appendChild(node);
  }}

  function initControls() {{
    ['single_prompt', 'batch_overlay', 'aggregate_heatmap', 'compare_prompts'].forEach((mode) => option(controls.mode, mode, mode));
    controls.mode.value = rendererOptions.visualizationMode;
    const promptIds = [...new Set((payload.activationPaths || []).map((path) => path.promptId).filter((value) => value !== null && value !== undefined))];
    if (!promptIds.length) option(controls.prompt, '', 'unavailable');
    promptIds.forEach((promptId) => option(controls.prompt, promptId, promptId));
    controls.prompt.value = rendererOptions.selectedPromptId ?? promptIds[0] ?? '';
    const tokenIds = [...new Set((payload.activationPaths || []).map((path) => path.tokenIndex).filter((value) => value !== null && value !== undefined))];
    if (!tokenIds.length) option(controls.token, '', 'unavailable');
    tokenIds.forEach((tokenId) => option(controls.token, tokenId, tokenId));
    controls.token.value = rendererOptions.selectedTokenId ?? tokenIds[0] ?? '';
    controls.topK.value = rendererOptions.topK;
    controls.backgroundOpacity.value = rendererOptions.backgroundOpacity;
    controls.edgeThreshold.value = rendererOptions.edgeThreshold;
    controls.showAggregate.checked = rendererOptions.showAggregateHeatmap;
    controls.showSecondary.checked = rendererOptions.showSecondaryBranches;
    controls.developerDiagnostics.checked = rendererOptions.developerDiagnostics;
    Object.values(controls).forEach((control) => control.addEventListener('input', () => {{
      rendererOptions.visualizationMode = controls.mode.value;
      rendererOptions.selectedPromptId = controls.prompt.value;
      rendererOptions.selectedTokenId = controls.token.value === '' ? null : Number(controls.token.value);
      rendererOptions.topK = Number(controls.topK.value);
      rendererOptions.backgroundOpacity = Number(controls.backgroundOpacity.value);
      rendererOptions.edgeThreshold = Number(controls.edgeThreshold.value);
      rendererOptions.showAggregateHeatmap = controls.showAggregate.checked;
      rendererOptions.showSecondaryBranches = controls.showSecondary.checked;
      rendererOptions.developerDiagnostics = controls.developerDiagnostics.checked;
      requestAnimationFrame(drawAll);
    }}));
  }}

  function roundedPath(x, y, w, h, r) {{
    const radius = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + w - radius, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
    ctx.lineTo(x + w, y + h - radius);
    ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
    ctx.lineTo(x + radius, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }}

  function fmtNumber(value) {{
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '0.00';
    return Number(value).toFixed(2);
  }}

  function escapeHtml(value) {{
    return String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }}

  function trimText(value, limit = 28) {{
    const text = String(value || '');
    const compact = text.replace(/\\s+/g, ' ').trim();
    return compact.length <= limit ? compact : compact.slice(0, limit - 1) + '…';
  }}

  function promptText(detail) {{
    const text = trimText(detail?.promptText || detail?.promptPreview || '', 260);
    return text || 'unavailable';
  }}

  function promptPreviewList(detail) {{
    const values = Array.isArray(detail?.promptPreviewList) ? detail.promptPreviewList : [];
    if (values.length) return values.map((value) => trimText(value, 260)).filter(Boolean);
    const single = promptText(detail);
    return single === 'unavailable' ? [] : [single];
  }}

  function promptRows(detail) {{
    const prompts = promptPreviewList(detail);
    if (prompts.length > 1) {{
      const rows = prompts.slice(0, 5).map((value, index) => [`prompt ${{index + 1}}`, value]);
      if (prompts.length > 5) {{
        rows.push(['all prompts', `click for details (${{prompts.length}} total)`]);
      }}
      return rows;
    }}
    return [['prompt', promptText(detail)]];
  }}

  function hexToRgb(hex) {{
    const text = String(hex || '').replace('#', '').trim();
    if (text.length !== 6) return [98, 228, 255];
    const value = Number.parseInt(text, 16);
    if (!Number.isFinite(value)) return [98, 228, 255];
    return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
  }}

  function rgba(rgb, alpha) {{
    const values = Array.isArray(rgb) && rgb.length >= 3 ? rgb : [98, 228, 255];
    return `rgba(${{values[0]}}, ${{values[1]}}, ${{values[2]}}, ${{Math.max(0, Math.min(1, alpha))}})`;
  }}

  function promptStyle(detail = null) {{
    const rgb = Array.isArray(detail?.promptRgb) ? detail.promptRgb : hexToRgb(detail?.promptColor || '#62E4FF');
    const opacity = Number(detail?.promptOpacity ?? 1);
    const dash = Array.isArray(detail?.promptDash) ? detail.promptDash : [];
    return {{
      color: detail?.promptColor || '#62E4FF',
      rgb,
      opacity: Number.isFinite(opacity) ? opacity : 1,
      dash,
    }};
  }}

  function annotationRows(detail) {{
    const tags = Array.isArray(detail?.annotationTags) ? detail.annotationTags.filter(Boolean) : [];
    const note = trimText(detail?.annotationNote || '', 150);
    const match = detail?.annotationMatchType || 'none';
    const rows = [['annotation match', match]];
    if (tags.length) rows.push(['tags', tags.join(', ')]);
    if (note) rows.push(['note', note]);
    return rows;
  }}

  function layerById(layerId) {{
    return (payload.layers || []).find((layer) => layer.layerId === layerId) || null;
  }}

  function validLayerIds() {{
    return (payload.layers || []).map((layer) => layer.layerId);
  }}

  function layerOrdinal(layerId) {{
    const layer = layerById(layerId);
    if (layer && Number.isFinite(Number(layer.renderIndex))) return Number(layer.renderIndex);
    if (layer && Number.isFinite(Number(layer.ordinal))) return Number(layer.ordinal);
    return validLayerIds().indexOf(layerId);
  }}

  function nearestValidLayerId(layerId) {{
    const layers = payload.layers || [];
    if (!layers.length) return null;
    if (layerById(layerId)) return layerId;
    const target = Number(String(layerId || '').replace('L', ''));
    if (!Number.isFinite(target)) return layers[0].layerId;
    return layers
      .slice()
      .sort((left, right) => {{
        const leftIndex = Number(left.index ?? String(left.layerId || '').replace('L', ''));
        const rightIndex = Number(right.index ?? String(right.layerId || '').replace('L', ''));
        return Math.abs(leftIndex - target) - Math.abs(rightIndex - target) || leftIndex - rightIndex;
      }})[0]?.layerId || layers[0].layerId;
  }}

  function groupsForLayer(layerId) {{
    return (payload.nodeGroups || []).filter((group) => group.layerId === layerId);
  }}

  function groupById(groupId) {{
    return (payload.nodeGroups || []).find((group) => group.groupId === groupId) || null;
  }}

  function batchById(batchId) {{
    return (payload.batches || []).find((batch) => batch.batchId === batchId) || null;
  }}

  function pathByBatchId(batchId) {{
    return filteredPaths().find((path) => path.batchId === batchId) || null;
  }}

  function edgeById(edgeId) {{
    return (payload.activationEdges || []).find((edge) => edge.edgeId === edgeId) || null;
  }}

  function syncSelection() {{
    if (!layerById(selected.layerId) && payload.layers?.length) {{
      selected.layerId = nearestValidLayerId(selected.layerId);
    }}
    const layerGroups = groupsForLayer(selected.layerId);
    if (!groupById(selected.groupId) || (groupById(selected.groupId) && groupById(selected.groupId).layerId !== selected.layerId)) {{
      selected.groupId = layerGroups[0]?.groupId || null;
    }}
    if (!batchById(selected.batchId) && payload.batches?.length) {{
      selected.batchId = payload.batches[0].batchId;
    }}
  }}

  function tooltipHtml(title, rows) {{
    const body = (rows || [])
      .filter((row) => row && row[1] !== undefined && row[1] !== null && String(row[1]) !== '')
      .map((row) => `<div class="gs-tip-row"><span class="gs-tip-key">${{escapeHtml(row[0])}}</span><span class="gs-tip-value">${{escapeHtml(row[1])}}</span></div>`)
      .join('');
    return `<div class="gs-tip-title">${{escapeHtml(title)}}</div>${{body}}`;
  }}

  function effectiveVisualizationState(detail = null) {{
    const payloadMode = payload.visualizationMode || payload.diagnostics?.visualizationMode || '';
    const payloadReason = payload.unavailableReason || payload.diagnostics?.unavailableReason || '';
    if (payload.visualizationMode === 'unavailable') {{
      return {{
        mode: 'unavailable',
        reason: detail?.unavailableReason || payloadReason,
      }};
    }}
    const detailMode = detail?.visualizationMode || '';
    const mode = detailMode || payloadMode;
    return {{
      mode,
      reason: mode === 'unavailable' ? (detail?.unavailableReason || payloadReason) : '',
    }};
  }}

  function visualizationRows(detail = null) {{
    const state = effectiveVisualizationState(detail);
    const rows = [['mode', state.mode || '']];
    if (state.reason) {{
      rows.push(['unavailableReason', trimText(state.reason, 42)]);
    }}
    return rows;
  }}

  function rangeText(value) {{
    if (!Array.isArray(value) || value.length < 2) return '';
    return `${{value[0]}}..${{value[1]}}`;
  }}

  function addHitTarget(x, y, w, h, tooltipHtmlValue, meta = {{}}) {{
    hitTargets.push({{ x, y, w, h, tooltip: tooltipHtmlValue, ...meta }});
  }}

  function addPathSegmentHitTarget(x1, y1, x2, y2, tooltipHtmlValue, meta = {{}}) {{
    hitTargets.push({{
      type: 'segment',
      x1,
      y1,
      x2,
      y2,
      padding: 10,
      tooltip: tooltipHtmlValue,
      ...meta,
    }});
  }}

  function filteredPaths() {{
    let paths = payload.activationPaths || [];
    if (rendererOptions.visualizationMode === 'single_prompt') {{
      paths = paths.filter((path) => String(path.promptId) === String(rendererOptions.selectedPromptId));
    }}
    if (rendererOptions.selectedTokenId !== null && rendererOptions.selectedTokenId !== undefined && rendererOptions.selectedTokenId !== '') {{
      paths = paths.filter((path) => Number(path.tokenIndex) === Number(rendererOptions.selectedTokenId));
    }}
    return paths;
  }}

  function filteredEdges() {{
    const visiblePaths = filteredPaths();
    const identities = new Set(visiblePaths.map((path) => `${{path.promptId}}:${{path.tokenIndex}}`));
    return (payload.activationEdges || []).filter((edge) => {{
      if (Number(edge.weight || 0) < Number(rendererOptions.edgeThreshold || 0)) return false;
      if (rendererOptions.visualizationMode === 'batch_overlay' && rendererOptions.selectedTokenId === null) return true;
      return identities.has(`${{edge.promptId}}:${{edge.tokenIndex}}`);
    }});
  }}

  function shouldDrawHeatmap() {{
    return rendererOptions.visualizationMode === 'aggregate_heatmap' || rendererOptions.showAggregateHeatmap;
  }}

  function distanceToSegment(px, py, x1, y1, x2, y2) {{
    const dx = x2 - x1;
    const dy = y2 - y1;
    if (!dx && !dy) {{
      return Math.hypot(px - x1, py - y1);
    }}
    const t = Math.max(0, Math.min(1, (((px - x1) * dx) + ((py - y1) * dy)) / ((dx * dx) + (dy * dy))));
    const sx = x1 + t * dx;
    const sy = y1 + t * dy;
    return Math.hypot(px - sx, py - sy);
  }}

  function clamp(value, min, max) {{
    const number = Number(value);
    if (!Number.isFinite(number)) return min;
    return Math.max(min, Math.min(max, number));
  }}

  function clampedPoint(x, y, bounds) {{
    return {{
      x: clamp(x, bounds.left, bounds.right),
      y: clamp(y, bounds.top, bounds.bottom),
    }};
  }}

  function withPanelClip(bounds, draw) {{
    ctx.save();
    ctx.beginPath();
    ctx.rect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
    ctx.clip();
    draw();
    ctx.restore();
  }}

  function panel(x, y, w, h) {{
    ctx.save();
    const fill = ctx.createLinearGradient(x, y, x, y + h);
    fill.addColorStop(0, 'rgba(23, 34, 58, 0.68)');
    fill.addColorStop(1, 'rgba(8, 13, 24, 0.88)');
    const glow = ctx.createLinearGradient(x, y, x + w, y + h);
    glow.addColorStop(0, 'rgba(130, 217, 255, 0.08)');
    glow.addColorStop(1, 'rgba(89, 110, 255, 0.03)');
    roundedPath(x, y, w, h, 18);
    ctx.fillStyle = fill;
    ctx.fill();
    roundedPath(x, y, w, h, 18);
    ctx.fillStyle = glow;
    ctx.fill();
    roundedPath(x, y, w, h, 18);
    ctx.strokeStyle = 'rgba(145, 177, 255, 0.18)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + 14, y + 34);
    ctx.lineTo(x + w - 14, y + 34);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.stroke();
    ctx.restore();
  }}

  function drawLabel(text, x, y, align = 'left', color = 'rgba(224, 233, 251, 0.92)', size = 12, weight = 600) {{
    ctx.save();
    ctx.fillStyle = color;
    ctx.font = `${{weight}} ${{size}}px Inter, sans-serif`;
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x, y);
    ctx.restore();
  }}

  function drawPath(path, left, right, top, bottom, index) {{
    const points = path.points || [];
    if (!points.length) return;
    const style = promptStyle(path);
    const hovered = hoverTarget?.batchId === path.batchId || hoverTarget?.pathId === path.pathId;
    const selectedPath = selected.batchId === path.batchId;
    const isApprox = path.visualizationMode === 'scalar_approx' || path.approximationReason;
    const alphaBase = Number(rendererOptions.backgroundOpacity || 0.24);
    const alpha = selectedPath ? 0.88 : hovered ? 0.72 : isApprox ? Math.max(0.12, alphaBase) : Math.max(0.18, alphaBase + 0.14);
    const finalAlpha = selectedPath ? alpha : alpha * style.opacity;
    const coords = points.map((point) => ({{
      x: left + (point.x || 0) * (right - left),
      y: top + (point.y || 0.5) * (bottom - top),
      point,
    }}));
    ctx.save();
    ctx.beginPath();
    coords.forEach((coord, coordIndex) => {{
      if (coordIndex === 0) {{
        ctx.moveTo(coord.x, coord.y);
      }} else {{
        const prev = coords[coordIndex - 1];
        const midX = (prev.x + coord.x) / 2;
        ctx.bezierCurveTo(midX, prev.y, midX, coord.y, coord.x, coord.y);
      }}
    }});
    ctx.strokeStyle = rgba(style.rgb, finalAlpha);
    ctx.lineWidth = selectedPath ? 4.6 : hovered ? 3.4 : 2.2;
    ctx.setLineDash(isApprox ? [10, 7] : style.dash);
    ctx.shadowColor = rgba(style.rgb, selectedPath ? 0.48 : 0.20 * style.opacity);
    ctx.shadowBlur = selectedPath ? 18 : 10;
    ctx.stroke();
    ctx.restore();

    coords.slice(1).forEach((coord, coordIndex) => {{
      const prev = coords[coordIndex];
      addPathSegmentHitTarget(
        prev.x,
        prev.y,
        coord.x,
        coord.y,
        tooltipHtml('Path', [
          ['type', 'Prompt path'],
          ['batch_id', path.batchId],
          ['prompt_id', path.promptId],
          ['token_id', path.tokenIndex],
          ...promptRows(path),
          ['layers', path.frequency || points.length],
          ['method', path.pathMethod || ''],
          ['tokenRange', trimText(path.tokenRange || '', 24)],
          ['output', trimText(path.outputToken || '', 24)],
          ['summary', trimText(path.activationSummary || '', 24)],
          ['strength', fmtNumber(path.strength)],
          ['attr', fmtNumber(path.attributionScore)],
          ['confidence', fmtNumber(path.confidence)],
          ['reason', trimText(path.approximationReason || '', 36)],
          ...visualizationRows(path),
        ]),
        {{
          batchId: path.batchId,
          pathId: path.pathId,
        }},
      );
    }});

    coords.forEach((coord) => {{
      const isFocusedPoint = coord.point.groupId === selected.groupId || coord.point.layerId === selected.layerId;
      const pointStyle = promptStyle(coord.point);
      ctx.save();
      ctx.beginPath();
      ctx.arc(coord.x, coord.y, isFocusedPoint ? 4.8 : 3.4, 0, Math.PI * 2);
      ctx.fillStyle = rgba(pointStyle.rgb, isFocusedPoint ? 0.95 : 0.82 * pointStyle.opacity);
      ctx.shadowColor = rgba(pointStyle.rgb, isFocusedPoint ? 0.58 : 0.38 * pointStyle.opacity);
      ctx.shadowBlur = isFocusedPoint ? 14 : 8;
      ctx.fill();
      ctx.restore();
      addHitTarget(
        coord.x - 10,
        coord.y - 10,
        20,
        20,
        tooltipHtml('Path', [
          ['type', 'Prompt path'],
          ['batch_id', path.batchId],
          ['prompt_id', path.promptId],
          ['tokenIndex', coord.point.tokenIndex],
          ['token_id', coord.point.tokenId],
          ...promptRows(coord.point),
          ['layer', coord.point.layerId],
          ['group', coord.point.groupId],
          ['token', trimText(coord.point.token || '')],
          ['strength', fmtNumber(path.strength)],
          ['method', path.pathMethod || ''],
          ...visualizationRows(path),
        ]),
        {{
          batchId: path.batchId,
          pathId: path.pathId,
          groupId: coord.point.groupId,
          layerId: coord.point.layerId,
        }},
      );
    }});

    if (rendererOptions.showSecondaryBranches) {{
      (path.branches || []).slice(0, rendererOptions.topK * 4).forEach((branch) => {{
        const bx = left + (branch.x || 0) * (right - left);
        const by = top + (branch.y || 0.5) * (bottom - top);
        ctx.save();
        ctx.beginPath();
        ctx.arc(bx, by, 2.4, 0, Math.PI * 2);
        ctx.fillStyle = rgba(promptStyle(branch).rgb, 0.34 * style.opacity);
        ctx.fill();
        ctx.restore();
      }});
    }}
  }}

  function layerX(layerId, left, right) {{
    const layers = payload.layers || [];
    const ordinal = layerOrdinal(layerId);
    const index = ordinal >= 0 ? ordinal : 0;
    return layers.length > 1 ? left + (index / (layers.length - 1)) * (right - left) : left + (right - left) / 2;
  }}

  function groupY(group, top, bottom) {{
    const y = Number(group?.yPosition);
    return top + (Number.isFinite(y) ? y : 0.5) * (bottom - top);
  }}

  function drawEdge(edge, left, right, top, bottom) {{
    const from = groupById(edge.fromNodeId || edge.fromGroupId);
    const to = groupById(edge.toNodeId || edge.toGroupId);
    if (!from || !to) return;
    const style = promptStyle(edge);
    const x1 = layerX(from.layerId, left, right);
    const y1 = groupY(from, top, bottom);
    const x2 = layerX(to.layerId, left, right);
    const y2 = groupY(to, top, bottom);
    const selectedEdge = selected.edgeId === edge.edgeId;
    const connected = selected.groupId && (selected.groupId === from.groupId || selected.groupId === to.groupId);
    const hovered = hoverTarget?.edgeId === edge.edgeId;
    const isApprox = edge.visualizationMode === 'scalar_approx' || edge.approximationReason;
    const weight = Math.max(0, Math.min(1, Number(edge.weight || 0)));
    const confidence = Math.max(0, Math.min(1, Number(edge.confidence || 0)));
    const alpha = selectedEdge ? 0.92 : hovered ? 0.78 : connected ? 0.58 : Math.max(0.10, weight * (isApprox ? 0.42 : 0.72));
    const finalAlpha = selectedEdge ? alpha : alpha * style.opacity;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    const midX = (x1 + x2) / 2;
    ctx.bezierCurveTo(midX, y1, midX, y2, x2, y2);
    ctx.strokeStyle = isApprox
      ? rgba(style.rgb, Math.max(0.16, finalAlpha * 0.72))
      : rgba(style.rgb, finalAlpha);
    ctx.lineWidth = selectedEdge ? 4.2 : Math.max(0.8, 0.8 + weight * 4.8);
    if (isApprox) ctx.setLineDash([5, 6]);
    ctx.shadowColor = isApprox ? 'transparent' : rgba(style.rgb, finalAlpha * 0.38);
    ctx.shadowBlur = selectedEdge ? 16 : 8;
    ctx.stroke();
    ctx.restore();
    addPathSegmentHitTarget(
      x1,
      y1,
      x2,
      y2,
      tooltipHtml('Edge', [
        ['from', from.groupId],
        ['to', to.groupId],
        ['batch_id', edge.batchId],
        ['prompt_id', edge.promptId],
        ['token_id', edge.tokenIndex],
        ...promptRows(edge),
        ['weight', fmtNumber(weight)],
        ['method', edge.method || ''],
        ['confidence', fmtNumber(confidence)],
        ['reason', trimText(edge.approximationReason || '', 36)],
        ...visualizationRows(edge),
      ]),
      {{
        edgeId: edge.edgeId,
        groupId: from.groupId,
        layerId: from.layerId,
      }},
    );
  }}

  function drawHeatmapCloud(left, right, top, bottom) {{
    const cells = payload.heatmap || [];
    cells.forEach((cell) => {{
      const x = layerX(`L${{cell.layer}}`, left, right);
      const y = top + Number(cell.y ?? 0.5) * (bottom - top);
      const intensity = Math.max(0, Math.min(1, Number(cell.normalizedActivation || 0)));
      const r = 3 + Math.sqrt(Math.max(1, Number(cell.count || 1))) * 1.8 + intensity * 6;
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 184, 64, ${{0.08 + intensity * 0.42}})`;
      ctx.shadowColor = `rgba(255, 184, 64, ${{0.18 + intensity * 0.28}})`;
      ctx.shadowBlur = 10 + intensity * 18;
      ctx.fill();
      ctx.restore();
      addHitTarget(
        x - r - 4,
        y - r - 4,
        (r + 4) * 2,
        (r + 4) * 2,
        tooltipHtml('Aggregate heatmap', [
          ['layer', `L${{cell.layer}}`],
          ['node', cell.nodeId],
          ['count', cell.count],
          ['prompts', cell.promptCount],
          ...promptRows(cell),
          ...annotationRows(cell),
          ['max', fmtNumber(cell.activationMax)],
          ['mean', fmtNumber(cell.activationMean)],
          ['mode', 'aggregate_heatmap'],
        ]),
        {{ layerId: `L${{cell.layer}}`, groupId: cell.nodeId }},
      );
    }});
  }}

  function drawOverviewNode(group, left, right, top, bottom) {{
    const cx = layerX(group.layerId, left, right);
    const cy = groupY(group, top, bottom);
    const style = promptStyle(group);
    const selectedGroup = selected.groupId === group.groupId;
    const connectedEdge = selected.edgeId && (() => {{
      const edge = edgeById(selected.edgeId);
      return edge && (edge.fromNodeId === group.groupId || edge.toNodeId === group.groupId);
    }})();
    const hovered = hoverTarget?.groupId === group.groupId;
    const activation = Math.max(0, Math.min(1, Number(group.normalizedActivation ?? group.activationValue ?? 0)));
    const radius = selectedGroup ? 6.8 : hovered || connectedEdge ? 5.8 : 3.2 + activation * 4.2;
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = selectedGroup
      ? rgba(style.rgb, 0.96)
      : rgba(style.rgb, (0.28 + activation * 0.62) * style.opacity);
    ctx.shadowColor = rgba(style.rgb, 0.42 * style.opacity);
    ctx.shadowBlur = selectedGroup ? 18 : 8 + activation * 10;
    ctx.fill();
    if (selectedGroup || connectedEdge) {{
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 4, 0, Math.PI * 2);
      ctx.strokeStyle = rgba(style.rgb, 0.70);
      ctx.lineWidth = 1.3;
      ctx.stroke();
    }}
    ctx.restore();
    addHitTarget(
      cx - radius - 8,
      cy - radius - 8,
      (radius + 8) * 2,
      (radius + 8) * 2,
      tooltipHtml('Node', [
        ['layer', group.layerId],
        ['node', group.groupId],
        ['batch_id', group.batchId],
        ['prompt_id', group.promptId],
        ['token_id', group.tokenIndex],
        ...promptRows(group),
        ...annotationRows(group),
        ['activation', fmtNumber(group.activationValue || 0)],
        ['normalized', fmtNumber(group.normalizedActivation || 0)],
        ['tokenRange', rangeText(group.tokenRange)],
        ['nodeRange', rangeText(group.nodeRange)],
        ['confidence', fmtNumber(group.confidence || 0)],
        ['source', (group.sourceFields || []).join(', ')],
        ['reason', trimText(group.approximationReason || '', 38)],
        ...visualizationRows(group),
      ]),
      {{
        groupId: group.groupId,
        layerId: group.layerId,
      }},
    );
  }}

  function drawFret(layer, index, total, left, right, top, bottom) {{
    const x = total > 1 ? left + (index / (total - 1)) * (right - left) : left + (right - left) / 2;
    const selectedLayer = selected.layerId === layer.layerId;
    const hovered = hoverTarget?.layerId === layer.layerId;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x, top - 10);
    ctx.lineTo(x, bottom + 12);
    ctx.strokeStyle = selectedLayer
      ? 'rgba(140, 230, 255, 0.86)'
      : hovered
        ? 'rgba(122, 183, 255, 0.66)'
        : 'rgba(138, 150, 186, 0.30)';
    ctx.lineWidth = selectedLayer ? 2.8 : 1.2;
    ctx.shadowColor = selectedLayer ? 'rgba(98, 220, 255, 0.36)' : 'transparent';
    ctx.shadowBlur = selectedLayer ? 14 : 0;
    ctx.stroke();
    ctx.restore();

    drawLabel(layer.name || layer.layerId || `L${{index}}`, x, bottom + 28, 'center', selectedLayer ? 'rgba(234, 247, 255, 0.98)' : 'rgba(179, 190, 221, 0.80)', 11, 700);
    addHitTarget(
      x - 22,
      top - 14,
      44,
      bottom - top + 50,
      tooltipHtml('Layer', [
        ['layer', layer.layerId],
        ['nodes', layer.groupCount || 0],
        ['density', fmtNumber(layer.activationDensity || 0)],
        ...visualizationRows(layer),
      ]),
      {{ layerId: layer.layerId }},
    );
  }}

  function groupActivation(group, layerGroups = []) {{
    if (Object.prototype.hasOwnProperty.call(group || {{}}, 'normalizedActivation')) {{
      const normalized = Number(group.normalizedActivation);
      if (Number.isFinite(normalized)) return Math.max(0, Math.min(1, normalized));
    }}
    const raw = Number(group?.activationValue ?? group?.activation);
    if (!Number.isFinite(raw) || raw <= 0) return 0;
    const maxRaw = Math.max(0, ...layerGroups.map((item) => Number(item?.activationValue ?? item?.activation)).filter((value) => Number.isFinite(value)));
    return maxRaw > 0 ? Math.max(0, Math.min(1, raw / maxRaw)) : 0;
  }}

  function box2Radius(activation, selectedGroup = false, hovered = false) {{
    const base = 2.8 + Math.sqrt(Math.max(0, activation)) * 7.4;
    const hoverLift = hovered ? 1.2 : 0;
    return selectedGroup ? Math.max(base + hoverLift, 6.4) : base + hoverLift;
  }}

  function box2Alpha(activation) {{
    return 0.16 + Math.max(0, Math.min(1, activation)) * 0.72;
  }}

  function selectedGroupOutline(cx, cy, radius) {{
    ctx.beginPath();
    ctx.arc(cx, cy, radius + 4.5, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(218, 249, 255, 0.78)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }}

  function box2NodeRenderDiagnostics(group) {{
    const hasNormalized = Object.prototype.hasOwnProperty.call(group || {{}}, 'normalizedActivation') && Number.isFinite(Number(group?.normalizedActivation));
    const hasRaw = Number.isFinite(Number(group?.activationValue ?? group?.activation));
    if (hasNormalized) return 'normalized_activation';
    if (hasRaw) return 'raw_activation_layer_normalized';
    return 'activation unavailable';
  }}

  function valueList(values) {{
    const list = Array.isArray(values) ? values : [];
    const unique = [];
    list.forEach((value) => {{
      if (value === null || value === undefined || value === '') return;
      const text = String(value);
      if (!unique.includes(text)) unique.push(text);
    }});
    return unique;
  }}

  function box2PromptEntries(group) {{
    const entries = Array.isArray(group?.promptActivations) ? group.promptActivations : [];
    if (entries.length) return entries;
    return [{{
      promptId: group?.promptId,
      batchId: group?.batchId,
      tokenIndex: group?.tokenIndex,
      tokenId: group?.tokenIndex,
      promptText: group?.promptText,
      promptPreview: group?.promptPreview,
      activationValue: group?.activationValue,
      normalizedActivation: group?.normalizedActivation,
      promptColor: group?.promptColor,
      promptRgb: group?.promptRgb,
      promptOpacity: group?.promptOpacity,
      promptDash: group?.promptDash,
    }}];
  }}

  function box2ColorStyle(group, activation) {{
    if (rendererOptions.visualizationMode === 'aggregate_heatmap' || payload.diagnostics?.promptColorMode === 'aggregate') {{
      return {{ color: '#FFB840', rgb: [255, 184, 64], opacity: 1, dash: [] }};
    }}
    if (rendererOptions.visualizationMode === 'batch_overlay') {{
      const entries = box2PromptEntries(group);
      return promptStyle(entries[0] || group);
    }}
    return promptStyle(group);
  }}

  function drawPromptOverlapMarkers(cx, cy, radius, group) {{
    if (rendererOptions.visualizationMode !== 'batch_overlay') return;
    const entries = box2PromptEntries(group);
    const promptIds = valueList(entries.map((entry) => entry.promptId));
    if (promptIds.length <= 1) return;
    const limit = Math.min(entries.length, 5);
    for (let index = 0; index < limit; index++) {{
      const entry = entries[index];
      const style = promptStyle(entry);
      const angle = (-Math.PI / 2) + (index / Math.max(limit, 1)) * Math.PI * 2;
      const mx = cx + Math.cos(angle) * (radius + 6);
      const my = cy + Math.sin(angle) * (radius + 6);
      ctx.save();
      ctx.beginPath();
      ctx.arc(mx, my, 2.4, 0, Math.PI * 2);
      ctx.fillStyle = rgba(style.rgb, 0.94 * style.opacity);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(mx, my, 3.8, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(5, 9, 18, 0.86)';
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.restore();
    }}
  }}

  function drawGroupDot(group, index, total, x, y, w, h, layerGroups = []) {{
    const cols = Math.max(1, Math.min(18, Math.ceil(Math.sqrt(total * 2.2))));
    const rows = Math.max(1, Math.ceil(total / cols));
    const col = index % cols;
    const row = Math.floor(index / cols);
    const cellW = w / cols;
    const cellH = h / rows;
    const cx = x + col * cellW + cellW / 2;
    const cy = y + row * cellH + cellH / 2;
    const selectedGroup = selected.groupId === group.groupId;
    const hovered = hoverTarget?.groupId === group.groupId;
    const activation = groupActivation(group, layerGroups);
    const radius = box2Radius(activation, selectedGroup, hovered);
    const alpha = box2Alpha(activation);
    const style = box2ColorStyle(group, activation);
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = rgba(style.rgb, alpha * style.opacity);
    ctx.shadowColor = rgba(style.rgb, (0.16 + activation * 0.46) * style.opacity);
    ctx.shadowBlur = 5 + activation * 18 + (selectedGroup ? 5 : 0);
    ctx.fill();
    drawPromptOverlapMarkers(cx, cy, radius, group);
    if (selectedGroup) {{
      selectedGroupOutline(cx, cy, radius);
    }}
    ctx.restore();
    const promptEntries = box2PromptEntries(group);
    const promptIds = valueList(group.promptIds || promptEntries.map((entry) => entry.promptId));
    const batchIds = valueList(group.batchIds || promptEntries.map((entry) => entry.batchId));
    const tokenIds = valueList(group.tokenIds || promptEntries.map((entry) => entry.tokenIndex ?? entry.tokenId));

    addHitTarget(
      cx - radius - 8,
      cy - radius - 8,
      (radius + 8) * 2,
      (radius + 8) * 2,
      tooltipHtml('Group', [
        ['node_id', group.nodeId || group.groupId],
        ['cluster_id', group.clusterId],
        ['group', group.groupId],
        ['layer', group.layerId],
        ['batch_id', group.batchId],
        ['prompt_id', group.promptId],
        ['token_id', group.tokenIndex],
        ['prompt_id(s)', promptIds.join(', ')],
        ['batch_id(s)', batchIds.join(', ')],
        ['token_id(s)', tokenIds.join(', ')],
        ['overlap prompts', group.promptOverlapCount || promptIds.length || 0],
        ...promptRows(group),
        ...annotationRows(group),
        ['batches', group.batchParticipation || 0],
        ['activation', fmtNumber(group.activationValue || 0)],
        ['normalized', fmtNumber(group.normalizedActivation || 0)],
        ['box2 activation source', box2NodeRenderDiagnostics(group)],
        ['tokenRange', rangeText(group.tokenRange)],
        ['nodeRange', rangeText(group.nodeRange)],
        ['confidence', fmtNumber(group.confidence || 0)],
        ['source', (group.sourceFields || []).join(', ')],
        ['reason', trimText(group.approximationReason || '', 38)],
        ['attr', fmtNumber(group.attributionScore || 0)],
        ...visualizationRows(group),
      ]),
      {{
        groupId: group.groupId,
        layerId: group.layerId,
      }},
    );
  }}

  function drawSelectedTarget(x, y, groupId, bounds = null) {{
    const selectedGroup = groupById(groupId);
    const selectedLayer = layerById(selected.layerId);
    const selectedPath = rendererOptions.visualizationMode === 'aggregate_heatmap' ? null : pathByBatchId(selected.batchId);
    const selectedEdge = edgeById(selected.edgeId);
    const points = selectedPath?.points || [];
    const currentIndex = points.findIndex((point) => point.layerId === selected.layerId);
    const previewPoints = currentIndex >= 0
      ? points.slice(Math.max(0, currentIndex - 1), currentIndex + 2)
      : points.slice(0, 3);

    if (rendererOptions.visualizationMode !== 'aggregate_heatmap' && previewPoints.length > 1) {{
      const previewBounds = bounds || {{ left: x - 150, right: x + 150, top: y - 58, bottom: y + 58 }};
      const startX = clamp(x - 120, previewBounds.left, previewBounds.right);
      const span = Math.min(240, previewBounds.right - previewBounds.left);
      const baseGroupIndex = Number(previewPoints[0]?.groupIndex);
      const groupIndexBase = Number.isFinite(baseGroupIndex) ? baseGroupIndex : 0;
      withPanelClip(previewBounds, () => {{
        ctx.save();
        ctx.beginPath();
        previewPoints.forEach((point, index) => {{
          const rawGroupIndex = Number(point?.groupIndex);
          const groupIndexOffset = Number.isFinite(rawGroupIndex) ? rawGroupIndex - groupIndexBase : 0;
          const px = startX + (previewPoints.length === 1 ? span / 2 : (index / (previewPoints.length - 1)) * span);
          const py = y + groupIndexOffset * 18;
          const local = clampedPoint(px, py, previewBounds);
          if (index === 0) ctx.moveTo(local.x, local.y);
          else ctx.lineTo(local.x, local.y);
        }});
        ctx.strokeStyle = 'rgba(98, 228, 255, 0.65)';
        ctx.lineWidth = 3;
        ctx.shadowColor = 'rgba(79, 214, 255, 0.30)';
        ctx.shadowBlur = 12;
        ctx.stroke();
        ctx.restore();
      }});
    }}

    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, 28, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(13, 23, 43, 0.92)';
    ctx.shadowColor = 'rgba(87, 227, 255, 0.45)';
    ctx.shadowBlur = 22;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, 28, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(120, 230, 255, 0.72)';
    ctx.lineWidth = 2.4;
    ctx.stroke();
    ctx.restore();

    drawLabel(selectedGroup?.groupId || 'No group', x, y - 8, 'center', 'rgba(236, 246, 255, 0.96)', 13, 700);
    if (selectedLayer) {{
      drawLabel(selectedLayer.name || selectedLayer.layerId, x - 132, y, 'left', 'rgba(158, 173, 205, 0.72)', 11, 600);
    }}
  }}

  function selectedDiagnostics() {{
    const base = payload.diagnostics || {{}};
    const selectedLayer = layerById(selected.layerId) || base.selectedLayer || null;
    const selectedGroup = groupById(selected.groupId) || base.selectedGroup || null;
    const selectedBatch = batchById(selected.batchId) || base.selectedBatch || null;
    const selectedPath = pathByBatchId(selected.batchId) || null;
    const selectedEdge = edgeById(selected.edgeId) || null;
    const pathPoints = selectedPath?.points || [];
    const selectedPoint = pathPoints.find((point) => point.groupId === selected.groupId)
      || pathPoints.find((point) => point.layerId === selected.layerId)
      || pathPoints[0]
      || null;

    return {{
      ...base,
      selectedBatch,
      selectedLayer,
      selectedGroup,
      selectedPath,
      activationValue: selectedPoint?.activationValue ?? selectedGroup?.activationValue ?? base.activationValue ?? 0,
      attributionScore: selectedGroup?.attributionScore ?? selectedPath?.attributionScore ?? base.attributionScore ?? 0,
      confidence: selectedPath?.confidence ?? base.confidence ?? 0,
      visualizationMode: selectedPoint?.visualizationMode || selectedGroup?.visualizationMode || selectedLayer?.visualizationMode || selectedPath?.visualizationMode || base.visualizationMode || payload.visualizationMode || '',
      selectedEdge,
      promptText: selectedEdge?.promptText || selectedPoint?.promptText || selectedPath?.promptText || selectedGroup?.promptText || '',
      promptPreviewList: selectedEdge?.promptPreviewList || selectedPoint?.promptPreviewList || selectedPath?.promptPreviewList || selectedGroup?.promptPreviewList || [],
      annotationTags: selectedGroup?.annotationTags || [],
      annotationNote: selectedGroup?.annotationNote || '',
      annotationMatchType: selectedGroup?.annotationMatchType || 'none',
      sourceToken: selectedPoint?.token || base.sourceToken || '',
      destinationToken: selectedBatch?.outputToken || base.destinationToken || '',
    }};
  }}

  function drawDiagnosticText(x, y, d) {{
    const state = effectiveVisualizationState(d);
    const rows = [
      ['visualizationMode', state.mode || ''],
      ['dataMode', payload.dataMode || d.dataMode || ''],
      ['promptColorMode', d.promptColorMode || ''],
      ['promptColorCount', d.promptColorCount ?? ''],
      ['paletteSize', d.paletteSize ?? ''],
      ['colorsReused', d.colorsReused ?? false],
      ['selectedBatch', d.selectedBatch?.batchId || selected.batchId || ''],
      ['selectedPrompt', rendererOptions.selectedPromptId ?? d.selectedPromptId ?? ''],
      ['selectedToken', rendererOptions.selectedTokenId ?? d.selectedTokenId ?? ''],
      ['selectedLayer', d.selectedLayer?.layerId || selected.layerId || ''],
      ['selectedGroup', d.selectedGroup?.groupId || selected.groupId || ''],
      ['promptText', trimText(d.promptText || '', 72)],
      ['annotationMatch', d.annotationMatchType || 'none'],
      ['annotationTags', Array.isArray(d.annotationTags) ? d.annotationTags.join(', ') : ''],
      ['annotationNote', trimText(d.annotationNote || '', 72)],
      ['activationValue', fmtNumber(d.activationValue || 0)],
      ['attributionScore', fmtNumber(d.attributionScore || 0)],
      ['confidence', fmtNumber(d.confidence || 0)],
      ['selectedEdge', d.selectedEdge?.edgeId || selected.edgeId || ''],
      ['edgeWeight', d.selectedEdge ? fmtNumber(d.selectedEdge.weight || 0) : ''],
	      ['edgeMethod', d.selectedEdge?.method || ''],
	      ['sourceToken', trimText(d.sourceToken || '', 24)],
	      ['destinationToken', trimText(d.destinationToken || '', 24)],
	      ['model', trimText(d.modelMeta?.modelName || '', 24)],
	      ['uiSelectedModelPath', trimText(d.uiSelectedModelPath || '', 58)],
	      ['uiSelectedModelName', trimText(d.uiSelectedModelName || '', 36)],
	      ['backendProcessModelPath', trimText(d.backendProcessModelPath || '', 58)],
	      ['backendInfoModelPath', trimText(d.backendInfoModelPath || '', 58)],
	      ['backendInfoModelName', trimText(d.backendInfoModelName || '', 36)],
	      ['activeModelFingerprint', trimText(d.activeModelFingerprint || '', 36)],
	      ['ggufArchitecture', d.ggufArchitecture || ''],
	      ['ggufLayerCount', d.ggufLayerCount ?? ''],
	      ['backendInfoLayerCount', d.backendInfoLayerCount ?? ''],
	      ['traceRequested', `${{d.traceRequestedMinLayer ?? ''}}..${{d.traceRequestedMaxLayer ?? ''}} (${{d.traceRequestedLayerCount ?? 0}})`],
	      ['traceReturned', `${{d.traceReturnedMinLayer ?? ''}}..${{d.traceReturnedMaxLayer ?? ''}} (${{d.traceReturnedLayerCount ?? 0}})`],
	      ['rendererLayers', `${{d.rendererMinLayer ?? ''}}..${{d.rendererMaxLayer ?? ''}} (${{d.rendererLayerCount ?? d.renderLayerCount ?? 0}})`],
	      ['modelIdentityMismatch', d.modelIdentityMismatch ?? false],
	      ['layerMismatchWarning', trimText(d.layerMismatchWarning || '', 58)],
	      ['staleCacheSuspected', d.staleCacheSuspected ?? false],
	    ];
    (payload.diagnostics?.warnings || []).forEach((warning, index) => {{
      rows.push([`warning${{index + 1}}`, trimText(warning, 58)]);
    }});
    if (state.reason) {{
      rows.push(['unavailableReason', trimText(state.reason, 42)]);
    }}
    ctx.save();
    ctx.fillStyle = 'rgba(224, 231, 244, 0.90)';
    ctx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    rows.forEach((row, index) => {{
      ctx.fillText(`${{row[0]}}: ${{row[1]}}`, x, y + index * 18);
    }});
    ctx.restore();
  }}

  function drawOverview(rect) {{
    const x = 12, y = 104, w = rect.width - 24, h = Math.max(480, rect.height * 0.32);
    panel(x, y, w, h);
    const layers = payload.layers || [];
    const edges = filteredEdges();
    const paths = filteredPaths();
    const groups = payload.nodeGroups || [];
    const left = x + 56, right = x + w - 56, top = y + 48, bottom = y + h - 54;

    ctx.save();
    for (let i = 0; i < 5; i++) {{
      const gy = top + (i / 4) * (bottom - top);
      ctx.beginPath();
      ctx.moveTo(left, gy);
      ctx.lineTo(right, gy);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }}
    ctx.restore();

    const overviewBounds = {{ left, right, top, bottom }};
    withPanelClip(overviewBounds, () => {{
      if (shouldDrawHeatmap()) {{
        drawHeatmapCloud(left, right, top, bottom);
      }}
      if (rendererOptions.visualizationMode !== 'aggregate_heatmap') {{
        edges.forEach((edge) => drawEdge(edge, left, right, top, bottom));
        paths.forEach((path, index) => drawPath(path, left, right, top, bottom, index));
      }}
      groups.forEach((group) => drawOverviewNode(group, left, right, top, bottom));
    }});
    layers.forEach((layer, index) => drawFret(layer, index, layers.length, left, right, top, bottom));
  }}

  function drawLayerPane(rect) {{
    const x = 62, y = Math.max(720, rect.height * 0.45), w = rect.width - 124, h = 300;
    panel(x, y, w, h);
    const layer = layerById(selected.layerId) || payload.layers?.[0] || null;
    const groups = (payload.nodeGroups || []).filter((g) => g.layerId === selected.layerId);
    const layerBounds = {{ left: x + 76, right: x + w - 56, top: y + 46, bottom: y + h - 24 }};
    withPanelClip(layerBounds, () => {{
      groups.forEach((group, index) => drawGroupDot(group, index, groups.length, x + 76, y + 46, w - 132, h - 70, groups));
    }});
    if (layer) {{
      drawLabel(layer.name || layer.layerId, x + 18, y + 19, 'left', 'rgba(141, 228, 255, 0.82)', 11, 700);
    }}
    if (selected.groupId) {{
      drawLabel(selected.groupId, x + 18, y + h - 18, 'left', 'rgba(141, 228, 255, 0.82)', 11, 700);
    }}
  }}

  function drawDrilldownPane(rect) {{
    const x = 62, y = Math.max(1060, rect.height * 0.64), w = rect.width - 124, h = 300;
    panel(x, y, w, h);
    const drilldownBounds = {{ left: x + 18, right: x + w - 18, top: y + 42, bottom: y + h - 18 }};
    withPanelClip(drilldownBounds, () => {{
      drawSelectedTarget(x + w / 2, y + h / 2 + 6, selected.groupId, drilldownBounds);
    }});
  }}

  function drawDiagnostics(rect) {{
    const x = 62, y = Math.max(640, rect.height * 0.80), w = rect.width - 124, h = rect.height - y - 16;
    if (!rendererOptions.developerDiagnostics && y < rect.height - 80) return;
    panel(x, y, w, Math.max(110, h));
    drawDiagnosticText(x + 18, y + 28, selectedDiagnostics());
  }}

  function findTarget(x, y) {{
    for (let i = hitTargets.length - 1; i >= 0; i--) {{
      const t = hitTargets[i];
      if (t.type === 'segment') {{
        if (distanceToSegment(x, y, t.x1, t.y1, t.x2, t.y2) <= (t.padding || 0)) return t;
        continue;
      }}
      if (x >= t.x && x <= t.x + t.w && y >= t.y && y <= t.y + t.h) return t;
    }}
    return null;
  }}

  function drawAll() {{
    syncSelection();
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    hitTargets = [];
    drawOverview(rect);
    drawLayerPane(rect);
    drawDrilldownPane(rect);
    drawDiagnostics(rect);
  }}

  function resizeCanvas() {{
    const rect = shell.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawAll();
  }}

  canvas.addEventListener('mousemove', (event) => {{
    const rect = canvas.getBoundingClientRect();
    hoverTarget = findTarget(event.clientX - rect.left, event.clientY - rect.top);
    if (hoverTarget) {{
      tooltip.style.display = 'block';
      tooltip.style.left = Math.min(rect.width - 300, event.clientX - rect.left + 14) + 'px';
      tooltip.style.top = Math.min(rect.height - 110, event.clientY - rect.top + 14) + 'px';
      tooltip.innerHTML = hoverTarget.tooltip;
    }} else {{
      tooltip.style.display = 'none';
    }}
    requestAnimationFrame(drawAll);
  }});

  canvas.addEventListener('mouseleave', () => {{
    hoverTarget = null;
    tooltip.style.display = 'none';
    requestAnimationFrame(drawAll);
  }});

  canvas.addEventListener('click', () => {{
    if (!hoverTarget) return;
    if (hoverTarget.layerId) {{
      selected.layerId = hoverTarget.layerId;
    }}
    if (hoverTarget.groupId) {{
      selected.groupId = hoverTarget.groupId;
      const group = groupById(hoverTarget.groupId);
      if (group?.layerId) {{
        selected.layerId = group.layerId;
      }}
    }}
    if (hoverTarget.batchId) {{
      selected.batchId = hoverTarget.batchId;
    }}
    if (hoverTarget.edgeId) {{
      selected.edgeId = hoverTarget.edgeId;
    }}
    syncSelection();
    requestAnimationFrame(drawAll);
  }});

  window.addEventListener('resize', resizeCanvas);
  initControls();
  resizeCanvas();
  </script>
</div>
"""


def _legend_panel_html(payload: dict) -> str:
    panel = payload.get("promptLegendPanel") if isinstance(payload.get("promptLegendPanel"), dict) else {}
    entries = panel.get("entries") if isinstance(panel.get("entries"), list) else []
    color_mode = (payload.get("diagnostics") or {}).get("promptColorMode")
    if color_mode != "prompt_palette" or len(entries) <= 1:
        return ""
    selected_prompt = panel.get("selectedPromptId")
    rows = []
    for entry in entries:
        color = html.escape(str(entry.get("promptColor") or "#62E4FF"))
        prompt_id = html.escape(str(entry.get("promptId") if entry.get("promptId") is not None else ""))
        label = html.escape(str(entry.get("label") or entry.get("promptPreview") or f"prompt {prompt_id}"))
        selected = str(entry.get("promptId")) == str(selected_prompt) if selected_prompt is not None else False
        rows.append(
            f"<div class='gs-prompt-legend-row{' is-selected' if selected else ''}' data-prompt-id='{prompt_id}'>"
            f"<span class='gs-prompt-legend-swatch' style='background:{color}'></span>"
            f"<span class='gs-prompt-legend-id'>{prompt_id}</span>"
            f"<span class='gs-prompt-legend-label'>{label}</span>"
            "</div>"
        )
    more_count = int(panel.get("moreCount") or 0)
    if more_count:
        rows.append(f"<div class='gs-prompt-legend-more'>+ {more_count} more prompts</div>")
    warning = ""
    if panel.get("colorsReused"):
        warning = "<div class='gs-prompt-legend-warning'>Palette colors are reused; prompt IDs disambiguate paths.</div>"
    return """
<style>
  .gs-prompt-legend-panel {
    margin-top: 10px;
    border: 1px solid rgba(146, 188, 255, 0.20);
    border-radius: 8px;
    background: rgba(6, 10, 20, 0.78);
    color: rgba(226, 239, 255, 0.92);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    padding: 10px 12px;
  }
  .gs-prompt-legend-title {
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 8px;
  }
  .gs-prompt-legend-list {
    display: grid;
    gap: 6px;
    max-height: 180px;
    overflow-y: auto;
  }
  .gs-prompt-legend-row {
    display: grid;
    grid-template-columns: 14px minmax(32px, max-content) minmax(0, 1fr);
    align-items: center;
    gap: 8px;
    width: 100%;
    box-sizing: border-box;
    min-width: 0;
    padding: 5px 6px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.025);
  }
  .gs-prompt-legend-row.is-selected {
    outline: 1px solid rgba(141, 228, 255, 0.62);
    background: rgba(98, 228, 255, 0.08);
  }
  .gs-prompt-legend-swatch {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    box-shadow: 0 0 10px currentColor;
  }
  .gs-prompt-legend-id {
    font-size: 11px;
    font-weight: 700;
    color: rgba(244, 248, 255, 0.94);
  }
  .gs-prompt-legend-label,
  .gs-prompt-legend-more,
  .gs-prompt-legend-warning {
    display: block;
    min-width: 0;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
    color: rgba(180, 193, 222, 0.84);
  }
  .gs-prompt-legend-warning {
    margin-top: 8px;
    color: rgba(255, 226, 122, 0.90);
  }
</style>
<div class="gs-prompt-legend-panel">
  <div class="gs-prompt-legend-title">Prompt paths</div>
  <div class="gs-prompt-legend-list">
    """ + "\n".join(rows) + """
  </div>
  """ + warning + """
</div>
"""


def render_activation_map(payload: dict, key: str = "activation_map_canvas", height: int = 1920) -> None:
    _ = key
    import streamlit as st

    st.iframe(activation_map_html(payload, height=height), height=height, width="stretch")
    legend_html = _legend_panel_html(payload)
    if legend_html:
        st.markdown(legend_html, unsafe_allow_html=True)
