from __future__ import annotations

import html
import json
from typing import Any


def _json_script_payload(payload: dict[str, Any]) -> str:
    return html.escape(
        json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"),
        quote=False,
    )


def activation_map_html(payload: dict, height: int = 960) -> str:
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
      gap: 10px;
      white-space: nowrap;
    }}
    .gs-map-tooltip .gs-tip-key {{
      color: rgba(167, 180, 208, 0.80);
    }}
    .gs-map-tooltip .gs-tip-value {{
      color: rgba(244, 248, 255, 0.96);
      text-align: right;
    }}
  </style>
  <canvas id="gs-activation-map"></canvas>
  <div id="gs-map-tooltip" class="gs-map-tooltip"></div>
  <script id="gs-map-data" type="application/json">{data}</script>
  <script>
  const payload = JSON.parse(document.getElementById('gs-map-data').textContent);
  const canvas = document.getElementById('gs-activation-map');
  const tooltip = document.getElementById('gs-map-tooltip');
  const ctx = canvas.getContext('2d');
  const shell = canvas.parentElement;
  let hoverTarget = null;
  let selected = {{
    layerId: payload.diagnostics?.selectedLayer?.layerId || payload.layers?.[0]?.layerId || null,
    groupId: payload.diagnostics?.selectedGroup?.groupId || payload.nodeGroups?.[0]?.groupId || null,
    batchId: payload.diagnostics?.selectedBatch?.batchId || payload.activationPaths?.[0]?.batchId || payload.batches?.[0]?.batchId || null
  }};
  let hitTargets = [];

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

  function trimText(value, limit = 28) {{
    const text = String(value || '');
    return text.length <= limit ? text : text.slice(0, limit - 1) + '…';
  }}

  function layerById(layerId) {{
    return (payload.layers || []).find((layer) => layer.layerId === layerId) || null;
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
    return (payload.activationPaths || []).find((path) => path.batchId === batchId) || null;
  }}

  function syncSelection() {{
    if (!layerById(selected.layerId) && payload.layers?.length) {{
      selected.layerId = payload.layers[0].layerId;
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
      .map((row) => `<div class="gs-tip-row"><span class="gs-tip-key">${{row[0]}}</span><span class="gs-tip-value">${{row[1]}}</span></div>`)
      .join('');
    return `<div class="gs-tip-title">${{title}}</div>${{body}}`;
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
    const hovered = hoverTarget?.batchId === path.batchId || hoverTarget?.pathId === path.pathId;
    const selectedPath = selected.batchId === path.batchId;
    const alpha = selectedPath ? 0.92 : hovered ? 0.80 : 0.34;
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
    ctx.strokeStyle = `rgba(92, 235, 255, ${{alpha}})`;
    ctx.lineWidth = selectedPath ? 4.2 : hovered ? 3.0 : 2.0;
    ctx.shadowColor = `rgba(84, 223, 255, ${{selectedPath ? 0.55 : 0.25}})`;
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
          ['batch', path.batchId],
          ['prompt', path.promptId],
          ['layers', path.frequency || points.length],
          ['tokenRange', trimText(path.tokenRange || '', 24)],
          ['output', trimText(path.outputToken || '', 24)],
          ['summary', trimText(path.activationSummary || '', 24)],
          ['strength', fmtNumber(path.strength)],
          ['attr', fmtNumber(path.attributionScore)],
          ['confidence', fmtNumber(path.confidence)],
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
      ctx.save();
      ctx.beginPath();
      ctx.arc(coord.x, coord.y, isFocusedPoint ? 4.8 : 3.4, 0, Math.PI * 2);
      ctx.fillStyle = isFocusedPoint ? 'rgba(213, 250, 255, 0.95)' : 'rgba(160, 233, 255, 0.82)';
      ctx.shadowColor = 'rgba(94, 225, 255, 0.55)';
      ctx.shadowBlur = isFocusedPoint ? 14 : 8;
      ctx.fill();
      ctx.restore();
      addHitTarget(
        coord.x - 10,
        coord.y - 10,
        20,
        20,
        tooltipHtml('Path', [
          ['batch', path.batchId],
          ['prompt', path.promptId],
          ['layer', coord.point.layerId],
          ['group', coord.point.groupId],
          ['token', trimText(coord.point.token || '')],
          ['strength', fmtNumber(path.strength)],
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

    const dotCount = Math.max(1, Math.min(layer.groupCount || 1, 42));
    for (let dotIndex = 0; dotIndex < dotCount; dotIndex++) {{
      const y = top + ((dotIndex + 0.5) / dotCount) * (bottom - top);
      const isTopActive = (layer.topActiveGroups || []).some((groupId) => {{
        const suffix = `-G${{dotIndex}}`;
        return groupId.endsWith(suffix);
      }});
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, isTopActive ? 2.8 : 1.7, 0, Math.PI * 2);
      ctx.fillStyle = isTopActive
        ? 'rgba(171, 247, 255, 0.84)'
        : 'rgba(168, 179, 212, 0.22)';
      ctx.shadowColor = isTopActive ? 'rgba(91, 232, 255, 0.42)' : 'transparent';
      ctx.shadowBlur = isTopActive ? 9 : 0;
      ctx.fill();
      ctx.restore();
    }}

    drawLabel(layer.name || layer.layerId || `L${{index}}`, x, bottom + 28, 'center', selectedLayer ? 'rgba(234, 247, 255, 0.98)' : 'rgba(179, 190, 221, 0.80)', 11, 700);
    addHitTarget(
      x - 22,
      top - 14,
      44,
      bottom - top + 50,
      tooltipHtml('Layer', [
        ['layer', layer.layerId],
        ['groups', layer.groupCount || 0],
        ['density', fmtNumber(layer.activationDensity || 0)],
        ...visualizationRows(layer),
      ]),
      {{ layerId: layer.layerId }},
    );
  }}

  function drawGroupDot(group, index, total, x, y, w, h) {{
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
    const radius = selectedGroup ? 7 : hovered ? 6 : 4.8;
    const glow = Math.min(1, Math.max(group.activationValue || 0, group.attributionScore || 0) / 4);
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = selectedGroup
      ? 'rgba(218, 251, 255, 0.94)'
      : `rgba(120, 224, 255, ${{0.30 + glow * 0.45}})`;
    ctx.shadowColor = selectedGroup ? 'rgba(103, 234, 255, 0.65)' : 'rgba(72, 201, 255, 0.28)';
    ctx.shadowBlur = selectedGroup ? 18 : 10;
    ctx.fill();
    if (selectedGroup) {{
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 4, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(125, 211, 252, 0.72)';
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }}
    ctx.restore();

    addHitTarget(
      cx - radius - 8,
      cy - radius - 8,
      (radius + 8) * 2,
      (radius + 8) * 2,
      tooltipHtml('Group', [
        ['group', group.groupId],
        ['layer', group.layerId],
        ['batches', group.batchParticipation || 0],
        ['activation', fmtNumber(group.activationValue || 0)],
        ['attr', fmtNumber(group.attributionScore || 0)],
        ...visualizationRows(group),
      ]),
      {{
        groupId: group.groupId,
        layerId: group.layerId,
      }},
    );
  }}

  function drawSelectedTarget(x, y, groupId) {{
    const selectedGroup = groupById(groupId);
    const selectedLayer = layerById(selected.layerId);
    const selectedPath = pathByBatchId(selected.batchId);
    const points = selectedPath?.points || [];
    const currentIndex = points.findIndex((point) => point.layerId === selected.layerId);
    const previewPoints = currentIndex >= 0
      ? points.slice(Math.max(0, currentIndex - 1), currentIndex + 2)
      : points.slice(0, 3);

    if (previewPoints.length > 1) {{
      const startX = x - 120;
      const span = 240;
      ctx.save();
      ctx.beginPath();
      previewPoints.forEach((point, index) => {{
        const px = startX + (previewPoints.length === 1 ? span / 2 : (index / (previewPoints.length - 1)) * span);
        const py = y + (point.groupIndex - (previewPoints[0].groupIndex || 0)) * 18;
        if (index === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }});
      ctx.strokeStyle = 'rgba(98, 228, 255, 0.65)';
      ctx.lineWidth = 3;
      ctx.shadowColor = 'rgba(79, 214, 255, 0.30)';
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.restore();
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
      activationValue: selectedPoint?.activationValue ?? selectedGroup?.activationValue ?? base.activationValue ?? 0,
      attributionScore: selectedGroup?.attributionScore ?? selectedPath?.attributionScore ?? base.attributionScore ?? 0,
      confidence: selectedPath?.confidence ?? base.confidence ?? 0,
      visualizationMode: selectedPoint?.visualizationMode || selectedGroup?.visualizationMode || selectedLayer?.visualizationMode || selectedPath?.visualizationMode || base.visualizationMode || payload.visualizationMode || '',
      sourceToken: selectedPoint?.token || base.sourceToken || '',
      destinationToken: selectedBatch?.outputToken || base.destinationToken || '',
    }};
  }}

  function drawDiagnosticText(x, y, d) {{
    const state = effectiveVisualizationState(d);
    const rows = [
      ['visualizationMode', state.mode || ''],
      ['selectedBatch', d.selectedBatch?.batchId || selected.batchId || ''],
      ['selectedLayer', d.selectedLayer?.layerId || selected.layerId || ''],
      ['selectedGroup', d.selectedGroup?.groupId || selected.groupId || ''],
      ['activationValue', fmtNumber(d.activationValue || 0)],
      ['attributionScore', fmtNumber(d.attributionScore || 0)],
      ['confidence', fmtNumber(d.confidence || 0)],
      ['sourceToken', trimText(d.sourceToken || '', 24)],
      ['destinationToken', trimText(d.destinationToken || '', 24)],
      ['model', trimText(d.modelMeta?.modelName || '', 24)],
    ];
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
    const x = 12, y = 10, w = rect.width - 24, h = Math.max(260, rect.height * 0.36);
    panel(x, y, w, h);
    const layers = payload.layers || [];
    const paths = payload.activationPaths || [];
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

    paths.forEach((path, index) => drawPath(path, left, right, top, bottom, index));
    layers.forEach((layer, index) => drawFret(layer, index, layers.length, left, right, top, bottom));
  }}

  function drawLayerPane(rect) {{
    const x = 62, y = Math.max(300, rect.height * 0.41), w = rect.width - 124, h = 150;
    panel(x, y, w, h);
    const layer = layerById(selected.layerId) || payload.layers?.[0] || null;
    const groups = (payload.nodeGroups || []).filter((g) => g.layerId === selected.layerId).slice(0, 120);
    groups.forEach((group, index) => drawGroupDot(group, index, groups.length, x + 76, y + 46, w - 132, h - 70));
    if (layer) {{
      drawLabel(layer.name || layer.layerId, x + 18, y + 19, 'left', 'rgba(141, 228, 255, 0.82)', 11, 700);
    }}
    if (selected.groupId) {{
      drawLabel(selected.groupId, x + 18, y + h - 18, 'left', 'rgba(141, 228, 255, 0.82)', 11, 700);
    }}
  }}

  function drawDrilldownPane(rect) {{
    const x = 62, y = Math.max(470, rect.height * 0.61), w = rect.width - 124, h = 150;
    panel(x, y, w, h);
    drawSelectedTarget(x + w / 2, y + h / 2 + 6, selected.groupId);
  }}

  function drawDiagnostics(rect) {{
    const x = 62, y = Math.max(640, rect.height * 0.80), w = rect.width - 124, h = rect.height - y - 16;
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
    syncSelection();
    requestAnimationFrame(drawAll);
  }});

  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();
  </script>
</div>
"""


def render_activation_map(payload: dict, key: str = "activation_map_canvas", height: int = 960) -> None:
    _ = key
    import streamlit.components.v1 as components

    components.html(activation_map_html(payload, height=height), height=height, scrolling=False)
