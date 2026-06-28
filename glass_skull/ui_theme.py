"""Presentation layer for Operation Glass Skull.

Holds the dark "scientific instrument" theme, reusable HTML components, and
small render helpers. Importing this module keeps main.py focused on behavior
while all visual styling lives here.
"""
from __future__ import annotations

import html as _html
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG = "#0b0f17"
BG_ELEVATED = "#11161f"
PANEL = "#151b26"
PANEL_2 = "#1b2230"
BORDER = "#27303f"
BORDER_SOFT = "#1f2735"
TEXT = "#e6edf3"
TEXT_MUTED = "#8b97a7"
TEXT_FAINT = "#5d6b7d"

ACCENT = "#38bdf8"          # cyan — primary
ACCENT_DEEP = "#0ea5e9"
TEAL = "#2dd4bf"
PURPLE = "#a371f7"          # steering
AMBER = "#e3b341"           # tracing / warning
GREEN = "#3fb950"           # online / loaded
RED = "#f85149"             # offline / danger
SLATE = "#64748b"


def inject_theme() -> None:
    """Inject the global stylesheet. Call once, right after set_page_config."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {{
            --gs-bg: {BG};
            --gs-panel: {PANEL};
            --gs-panel-2: {PANEL_2};
            --gs-border: {BORDER};
            --gs-text: {TEXT};
            --gs-muted: {TEXT_MUTED};
            --gs-accent: {ACCENT};
            --gs-purple: {PURPLE};
            --gs-amber: {AMBER};
            --gs-green: {GREEN};
            --gs-red: {RED};
        }}

        html, body, [class*="css"], .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(1200px 600px at 12% -8%, rgba(56,189,248,0.08), transparent 55%),
                radial-gradient(900px 500px at 95% 0%, rgba(163,113,247,0.07), transparent 50%),
                {BG};
            color: {TEXT};
        }}

        /* Tighten default top padding so the HUD sits high */
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1600px;
        }}

        #MainMenu, footer {{ visibility: hidden; }}
        [data-testid="stHeader"] {{ background: transparent; }}

        /* ---------- Typography ---------- */
        h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.01em; }}
        p, span, label, li {{ color: {TEXT}; }}
        code {{
            font-family: 'JetBrains Mono', monospace;
            background: rgba(56,189,248,0.10);
            color: {TEAL};
            border-radius: 5px;
            padding: 1px 6px;
            font-size: 0.82em;
        }}

        /* ---------- Sidebar ----------
           Configuration lives in the Settings tab. Keep Streamlit's sidebar
           shell hidden so the app does not present two control surfaces. */
        [data-testid="stSidebar"] {{
            display: none;
        }}
        [data-testid="stSidebarCollapsedControl"] {{
            display: none;
        }}
        [data-testid="collapsedControl"] {{
            display: none;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
            border-radius: 9px;
            border: 1px solid {BORDER};
            background: {PANEL_2};
            color: {TEXT};
            font-weight: 600;
            font-size: 0.86rem;
            padding: 0.46rem 0.95rem;
            transition: all 0.15s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {{
            border-color: {ACCENT};
            color: {ACCENT};
            transform: translateY(-1px);
        }}
        /* Primary CTA */
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DEEP} 100%);
            border: none;
            color: #04121c;
            box-shadow: 0 4px 16px rgba(56,189,248,0.25);
        }}
        .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
            color: #04121c;
            box-shadow: 0 6px 22px rgba(56,189,248,0.40);
            transform: translateY(-1px);
        }}

        /* ---------- Inputs ---------- */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        textarea,
        [data-baseweb="select"] > div {{
            background: {BG_ELEVATED} !important;
            border-radius: 8px !important;
            border: 1px solid {BORDER} !important;
            color: {TEXT} !important;
        }}
        textarea {{ font-family: 'JetBrains Mono', monospace !important; font-size: 0.86rem !important; }}
        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus,
        textarea:focus {{
            border-color: {ACCENT} !important;
            box-shadow: 0 0 0 2px rgba(56,189,248,0.18) !important;
        }}

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            background: {BG_ELEVATED};
            padding: 6px;
            border-radius: 12px;
            border: 1px solid {BORDER_SOFT};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: auto;
            padding: 9px 18px;
            border-radius: 8px;
            background: transparent;
            color: {TEXT_MUTED};
            font-weight: 600;
            font-size: 0.9rem;
            border: 1px solid transparent;
            transition: all 0.15s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{ color: {TEXT}; background: {PANEL_2}; }}
        .stTabs [aria-selected="true"] {{
            background: {PANEL_2} !important;
            color: {TEXT} !important;
            border: 1px solid {BORDER} !important;
            box-shadow: inset 0 -2px 0 {ACCENT};
        }}
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

        /* ---------- Chat messages ---------- */
        [data-testid="stChatMessage"] {{
            background: {PANEL};
            border: 1px solid {BORDER_SOFT};
            border-radius: 14px;
            padding: 0.65rem 0.9rem;
            margin-bottom: 0.55rem;
        }}

        /* ---------- DataFrame ---------- */
        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 10px;
            overflow: hidden;
        }}

        /* ---------- Expander ---------- */
        [data-testid="stExpander"] {{
            border: 1px solid {BORDER_SOFT};
            border-radius: 10px;
            background: {PANEL};
            overflow: hidden;
        }}
        [data-testid="stExpander"] summary {{ font-weight: 600; }}

        /* ---------- Toggles / sliders accent ---------- */
        [data-testid="stSlider"] [role="slider"] {{ background: {ACCENT} !important; }}

        /* ---------- Alerts ---------- */
        [data-testid="stAlert"] {{ border-radius: 10px; }}

        /* =====================================================
           Custom Glass Skull components
           ===================================================== */
        .gs-hud {{
            background: linear-gradient(135deg, {PANEL} 0%, {BG_ELEVATED} 100%);
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 18px 22px;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 14px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        }}
        .gs-hud-left {{ display: flex; align-items: center; gap: 16px; }}
        .gs-brand-mark {{
            width: 44px; height: 44px;
            border-radius: 12px;
            background: linear-gradient(135deg, {PURPLE}, {ACCENT});
            display: flex; align-items: center; justify-content: center;
            font-size: 24px;
            box-shadow: 0 4px 18px rgba(163,113,247,0.35);
        }}
        .gs-title {{ font-size: 1.18rem; font-weight: 700; line-height: 1.1; }}
        .gs-subtitle {{ font-size: 0.78rem; color: {TEXT_MUTED}; margin-top: 2px; }}
        .gs-hud-right {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}

        .gs-stat {{
            background: {BG_ELEVATED};
            border: 1px solid {BORDER_SOFT};
            border-radius: 10px;
            padding: 8px 14px;
            min-width: 84px;
            text-align: center;
        }}
        .gs-stat-label {{ font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase; color: {TEXT_FAINT}; }}
        .gs-stat-value {{ font-size: 0.98rem; font-weight: 700; color: {TEXT}; font-family: 'JetBrains Mono', monospace; }}

        .gs-pill {{
            display: inline-flex; align-items: center; gap: 7px;
            border-radius: 999px;
            padding: 6px 13px;
            font-size: 0.76rem;
            font-weight: 600;
            border: 1px solid {BORDER};
            background: {BG_ELEVATED};
        }}
        .gs-dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
        .gs-dot-pulse {{ box-shadow: 0 0 0 0 currentColor; animation: gs-pulse 2s infinite; }}
        @keyframes gs-pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(63,185,80,0.45); }}
            70% {{ box-shadow: 0 0 0 7px rgba(63,185,80,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(63,185,80,0); }}
        }}

        .gs-badge {{
            display: inline-flex; align-items: center; gap: 6px;
            border-radius: 8px;
            padding: 5px 11px;
            font-size: 0.74rem;
            font-weight: 600;
            border: 1px solid transparent;
        }}

        .gs-section {{
            display: flex; align-items: center; gap: 11px;
            margin: 4px 0 14px 0;
            padding-bottom: 11px;
            border-bottom: 1px solid {BORDER_SOFT};
        }}
        .gs-section-bar {{
            width: 4px; height: 30px; border-radius: 3px;
            background: linear-gradient(180deg, {ACCENT}, {PURPLE});
            flex-shrink: 0;
        }}
        .gs-section-title {{ font-size: 1.05rem; font-weight: 700; line-height: 1.05; }}
        .gs-section-sub {{ font-size: 0.76rem; color: {TEXT_MUTED}; margin-top: 1px; }}

        .gs-sidebar-head {{
            display: flex; align-items: center; gap: 10px;
            margin: 2px 0 16px 0;
        }}
        .gs-sidebar-mark {{
            width: 34px; height: 34px; border-radius: 9px;
            background: linear-gradient(135deg, {PURPLE}, {ACCENT});
            display:flex; align-items:center; justify-content:center; font-size:18px;
        }}
        .gs-sidebar-name {{ font-weight: 700; font-size: 0.98rem; line-height: 1; }}
        .gs-sidebar-ver {{ font-size: 0.68rem; color: {TEXT_MUTED}; }}

        .gs-sec-label {{
            font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
            color: {TEXT_FAINT}; font-weight: 700;
            margin: 16px 0 8px 0;
            display: flex; align-items: center; gap: 7px;
        }}

        .gs-prop {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 7px 0;
            border-bottom: 1px dashed {BORDER_SOFT};
        }}
        .gs-prop:last-child {{ border-bottom: none; }}
        .gs-prop-k {{ font-size: 0.78rem; color: {TEXT_MUTED}; }}
        .gs-prop-v {{ font-size: 0.82rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; color: {TEXT}; }}

        .gs-server-row {{
            display: flex; align-items: center; gap: 8px;
            padding: 6px 10px;
            background: {BG_ELEVATED};
            border: 1px solid {BORDER_SOFT};
            border-radius: 8px;
            margin-bottom: 6px;
            font-size: 0.76rem;
        }}
        .gs-server-url {{ font-family: 'JetBrains Mono', monospace; color: {TEXT_MUTED}; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

        .gs-empty {{
            border: 1.5px dashed {BORDER};
            border-radius: 16px;
            padding: 46px 28px;
            text-align: center;
            background: {PANEL};
        }}
        .gs-empty-icon {{ opacity: 0.6; display: flex; justify-content: center; }}

        .gs-net {{
            background: {PANEL};
            border: 1px solid {BORDER_SOFT};
            border-radius: 14px;
            padding: 14px 16px 8px 16px;
            margin-bottom: 14px;
        }}
        .gs-net svg {{ width: 100%; height: auto; display: block; }}
        .gs-chip-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
        .gs-chip {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
            background: {BG_ELEVATED}; border: 1px solid {BORDER_SOFT};
            border-radius: 7px; padding: 4px 9px; color: {TEXT_MUTED};
        }}
        .gs-chip b {{ color: {TEXT}; font-weight: 700; }}
        .gs-empty-title {{ font-size: 1.05rem; font-weight: 700; margin-top: 10px; }}
        .gs-empty-sub {{ font-size: 0.84rem; color: {TEXT_MUTED}; margin-top: 5px; }}

        .gs-card {{
            background: {PANEL};
            border: 1px solid {BORDER_SOFT};
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 14px;
        }}

        .gs-purpose {{
            font-size: 0.8rem; color: {TEXT_MUTED};
            background: {BG_ELEVATED};
            border-left: 3px solid {ACCENT};
            border-radius: 0 8px 8px 0;
            padding: 9px 13px;
            margin-bottom: 14px;
        }}

        .gs-timeline {{ display: flex; align-items: center; gap: 4px; flex-wrap: wrap; margin: 6px 0 14px 0; }}
        .gs-step {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            padding: 4px 9px;
            border-radius: 7px;
            background: {BG_ELEVATED};
            border: 1px solid {BORDER_SOFT};
            color: {TEXT_MUTED};
        }}
        .gs-step-cur {{ background: rgba(56,189,248,0.14); border-color: {ACCENT}; color: {ACCENT}; }}
        .gs-step-arrow {{ color: {TEXT_FAINT}; font-size: 0.7rem; }}

        .gs-term {{
            background: #070a10;
            border: 1px solid {BORDER};
            border-radius: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            line-height: 1.55;
            padding: 0;
            overflow: hidden;
        }}
        .gs-term-bar {{
            background: {PANEL_2};
            border-bottom: 1px solid {BORDER};
            padding: 7px 13px;
            display: flex; align-items: center; gap: 7px;
            font-size: 0.72rem; color: {TEXT_MUTED};
        }}
        .gs-term-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
        .gs-term-body {{ padding: 12px 15px; max-height: 560px; overflow-y: auto; }}
        .gs-log-line {{ display: flex; gap: 12px; padding: 2px 0; white-space: pre-wrap; }}
        .gs-log-ts {{ color: {TEAL}; flex-shrink: 0; }}
        .gs-log-tag {{ color: {PURPLE}; flex-shrink: 0; min-width: 118px; }}
        .gs-log-msg {{ color: {TEXT}; word-break: break-word; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Component builders (return HTML strings)
# ---------------------------------------------------------------------------
def _esc(text) -> str:
    return _html.escape(str(text))


def dot(color: str, pulse: bool = False) -> str:
    cls = "gs-dot gs-dot-pulse" if pulse else "gs-dot"
    return f'<span class="{cls}" style="background:{color};color:{color};"></span>'


def pill(label: str, color: str, pulse: bool = False) -> str:
    return (
        f'<span class="gs-pill" style="border-color:{color}33;">'
        f'{dot(color, pulse)}<span style="color:{TEXT};">{_esc(label)}</span></span>'
    )


def badge(label: str, color: str, active: bool = True) -> str:
    if active:
        style = f"background:{color}1f;border-color:{color}66;color:{color};"
    else:
        style = f"background:{BG};border-color:{BORDER};color:{TEXT_FAINT};"
    return f'<span class="gs-badge" style="{style}">{_esc(label)}</span>'


def section_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="gs-section-sub">{_esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="gs-section">'
        f'<div class="gs-section-bar"></div>'
        f'<div><div class="gs-section-title">{_esc(title)}</div>{sub}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def purpose(text: str) -> None:
    st.markdown(f'<div class="gs-purpose">{_esc(text)}</div>', unsafe_allow_html=True)


def sec_label(text: str) -> None:
    st.markdown(f'<div class="gs-sec-label">{_esc(text)}</div>', unsafe_allow_html=True)


_EMPTY_SVG = (
    '<svg width="46" height="46" viewBox="0 0 46 46" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="23" cy="23" r="20" stroke="#5d6b7d" stroke-width="1.5" stroke-dasharray="4 5" opacity="0.7"/>'
    '<path d="M15 23h16M23 15v16" stroke="#5d6b7d" stroke-width="1.5" stroke-linecap="round" opacity="0.55"/>'
    "</svg>"
)


def empty_state(title: str, subtitle: str = "") -> None:
    sub = f'<div class="gs-empty-sub">{_esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="gs-empty"><div class="gs-empty-icon">{_EMPTY_SVG}</div>'
        f'<div class="gs-empty-title">{_esc(title)}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def property_list(items: list[tuple[str, str]]) -> None:
    rows = "".join(
        f'<div class="gs-prop"><span class="gs-prop-k">{_esc(k)}</span>'
        f'<span class="gs-prop-v">{_esc(v)}</span></div>'
        for k, v in items
    )
    st.markdown(f'<div class="gs-card">{rows}</div>', unsafe_allow_html=True)


def hud(title: str, subtitle: str, stats: list[tuple[str, str]], pills_html: str) -> None:
    stat_html = "".join(
        f'<div class="gs-stat"><div class="gs-stat-label">{_esc(label)}</div>'
        f'<div class="gs-stat-value">{_esc(value)}</div></div>'
        for label, value in stats
    )
    st.markdown(
        f'<div class="gs-hud">'
        f'<div class="gs-hud-left">'
        f'<div class="gs-brand-mark"><span style="font-weight:800;font-size:16px;color:#04121c;letter-spacing:0.04em;">GS</span></div>'
        f'<div><div class="gs-title">{_esc(title)}</div>'
        f'<div class="gs-subtitle">{_esc(subtitle)}</div></div>'
        f"</div>"
        f'<div class="gs-hud-right">{stat_html}{pills_html}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def server_health_inline(label: str, url: str, status) -> str:
    if status is None:
        color, text = SLATE, "unchecked"
    elif status.online:
        color = GREEN
        latency = f"{status.latency_ms:.0f}ms" if status.latency_ms is not None else "online"
        text = f"online · {latency}"
    else:
        color, text = RED, "offline"
    return (
        f'<div class="gs-server-row">{dot(color)}'
        f'<span style="font-weight:600;min-width:48px;">{_esc(label)}</span>'
        f'<span class="gs-server-url">{_esc(url)}</span>'
        f'<span style="color:{color};font-weight:600;">{_esc(text)}</span></div>'
    )


def server_status_color(status) -> str:
    if status is None:
        return SLATE
    return GREEN if status.online else RED


def timeline(steps: list[str], current: Optional[int] = None) -> None:
    chunks = []
    for i, label in enumerate(steps):
        cls = "gs-step gs-step-cur" if current is not None and i == current else "gs-step"
        chunks.append(f'<span class="{cls}">{_esc(label)}</span>')
        if i < len(steps) - 1:
            chunks.append('<span class="gs-step-arrow">&rsaquo;</span>')
    st.markdown(f'<div class="gs-timeline">{"".join(chunks)}</div>', unsafe_allow_html=True)


def terminal(title: str, lines: list[tuple[str, str, str]]) -> None:
    """Render a terminal-style viewer. lines = list of (timestamp, tag, message)."""
    body = ""
    if not lines:
        body = f'<div style="color:{TEXT_FAINT};">— no log output yet —</div>'
    else:
        for ts, tag, msg in lines:
            body += (
                f'<div class="gs-log-line">'
                f'<span class="gs-log-ts">{_esc(ts)}</span>'
                f'<span class="gs-log-tag">{_esc(tag)}</span>'
                f'<span class="gs-log-msg">{_esc(msg)}</span></div>'
            )
    st.markdown(
        f'<div class="gs-term">'
        f'<div class="gs-term-bar">'
        f'<span class="gs-term-dot" style="background:{RED};"></span>'
        f'<span class="gs-term-dot" style="background:{AMBER};"></span>'
        f'<span class="gs-term-dot" style="background:{GREEN};"></span>'
        f'<span style="margin-left:8px;">{_esc(title)}</span></div>'
        f'<div class="gs-term-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


def render_network(summary: dict) -> None:
    """Animated SVG of the model graph, drawn accurately from the model spec.

    Columns = embedding -> one node-stack per model block -> unembedding.
    Animated signal dots and a flowing dashed line illustrate activations /
    weights propagating through the layers. All counts come from model.cfg.
    """
    n_layers = int(summary.get("layers", 0))
    d_model = int(summary.get("d_model", 0))
    heads = int(summary.get("heads", 0))
    d_head = int(summary.get("d_head", 0))
    d_mlp = int(summary.get("d_mlp", 0))
    vocab = int(summary.get("vocab_size", 0))
    params = int(summary.get("parameters", 0))

    width = 1100
    height = 300
    pad_x = 80
    usable = width - 2 * pad_x
    draw_layers = min(max(n_layers, 0), 18)
    truncated = draw_layers < n_layers
    cols = max(draw_layers + 2, 2)
    xs = [pad_x + (usable * i / (cols - 1)) for i in range(cols)]
    cy = height / 2
    neurons = 5
    spread = 96.0
    ys = [cy - spread / 2 + spread * j / (neurons - 1) for j in range(neurons)]

    # faint full connections between consecutive columns
    lines = []
    for c in range(cols - 1):
        for a in ys:
            for b in ys:
                lines.append(
                    f'<line x1="{xs[c]:.1f}" y1="{a:.1f}" x2="{xs[c + 1]:.1f}" y2="{b:.1f}" '
                    f'stroke="{ACCENT}" stroke-width="0.6" opacity="0.10"/>'
                )
    lines_svg = "".join(lines)

    # neuron nodes
    nodes = []
    for c in range(cols):
        for y in ys:
            nodes.append(
                f'<circle cx="{xs[c]:.1f}" cy="{y:.1f}" r="4.5" fill="{BG_ELEVATED}" '
                f'stroke="{ACCENT}" stroke-width="1.3"/>'
            )
    nodes_svg = "".join(nodes)

    # column labels
    def col_label(c: int) -> str:
        if c == 0:
            return "embed"
        if c == cols - 1:
            return "unembed"
        idx = c - 1
        if truncated and idx == draw_layers - 1:
            return f"L{n_layers - 1}"
        return f"L{idx}"

    labels = [
        f'<text x="{xs[c]:.1f}" y="{cy + spread / 2 + 28:.1f}" fill="{TEXT_MUTED}" '
        f'font-size="11" font-family="JetBrains Mono, monospace" text-anchor="middle">{col_label(c)}</text>'
        for c in range(cols)
    ]
    if truncated and cols >= 3:
        mid_x = (xs[cols - 2] + xs[cols - 3]) / 2
        labels.append(
            f'<text x="{mid_x:.1f}" y="{cy - spread / 2 - 12:.1f}" fill="{TEXT_FAINT}" '
            f'font-size="15" text-anchor="middle">… {n_layers} blocks total …</text>'
        )
    labels_svg = "".join(labels)

    # backbone path through column centers
    backbone = "M " + " L ".join(f"{x:.1f},{cy:.1f}" for x in xs)

    flow = (
        f'<path d="{backbone}" stroke="url(#gsflow)" stroke-width="2.6" fill="none" '
        f'stroke-dasharray="16 12" stroke-linecap="round">'
        f'<animate attributeName="stroke-dashoffset" from="0" to="-56" dur="1.1s" repeatCount="indefinite"/>'
        f"</path>"
    )

    dots = ""
    for begin in ("0s", "0.9s", "1.8s"):
        dots += (
            f'<circle r="5" fill="{TEAL}">'
            f'<animateMotion dur="2.7s" begin="{begin}" repeatCount="indefinite" path="{backbone}"/>'
            f'<animate attributeName="opacity" values="0;1;1;0" dur="2.7s" begin="{begin}" repeatCount="indefinite"/>'
            f"</circle>"
        )

    defs = (
        f'<defs><linearGradient id="gsflow" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0%" stop-color="{PURPLE}"/>'
        f'<stop offset="100%" stop-color="{ACCENT}"/></linearGradient></defs>'
    )

    chips = (
        '<div class="gs-chip-row">'
        f'<span class="gs-chip">layers <b>{n_layers}</b></span>'
        f'<span class="gs-chip">d_model <b>{d_model}</b></span>'
        f'<span class="gs-chip">heads <b>{heads}</b></span>'
        f'<span class="gs-chip">d_head <b>{d_head}</b></span>'
        f'<span class="gs-chip">d_mlp <b>{d_mlp}</b></span>'
        f'<span class="gs-chip">vocab <b>{vocab:,}</b></span>'
        f'<span class="gs-chip">params <b>{params:,}</b></span>'
        "</div>"
    )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg">{defs}{lines_svg}{flow}{nodes_svg}{dots}{labels_svg}</svg>'
    )
    st.markdown(f'<div class="gs-net">{chips}{svg}</div>', unsafe_allow_html=True)
