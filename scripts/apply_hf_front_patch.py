from __future__ import annotations

from pathlib import Path


MAIN = Path("main.py")


def replace_once(text: str, old: str, new: str, required: bool = True) -> str:
    if old not in text:
        if required:
            raise SystemExit(f"Patch anchor not found:\n{old[:300]}")
        return text
    return text.replace(old, new, 1)


def insert_after(text: str, anchor: str, insertion: str, required: bool = True) -> str:
    if insertion.strip() and insertion.strip() in text:
        return text
    if anchor not in text:
        if required:
            raise SystemExit(f"Patch anchor not found:\n{anchor[:300]}")
        return text
    return text.replace(anchor, anchor + insertion, 1)


def ensure_import_after(text: str, anchor: str, import_line: str) -> str:
    if import_line in text:
        return text
    return insert_after(text, anchor, import_line)


HELPER_ANCHOR = '''def feature_names_from_rows(rows: list[dict]) -> list[str]:
    return [str(f["name"]) for f in rows if f.get("name")]

'''


SET_DASHBOARD_TRACE = '''

def set_dashboard_trace(trace, prompt: str, backend: str, trace_model: str) -> None:
    st.session_state.dashboard_trace = trace
    st.session_state.dashboard_trace_counter = int(st.session_state.get("dashboard_trace_counter", 0)) + 1
    st.session_state.dashboard_trace_meta = {
        "prompt": prompt,
        "backend": backend,
        "trace_model": trace_model,
        "updated_at": datetime.now().strftime("%H:%M:%S"),
        "token_count": len(getattr(trace, "tokens", []) or []),
        "run": st.session_state.dashboard_trace_counter,
    }

'''


CAPABILITY_HELPER = '''

def render_capability_warning(chat_backend: str) -> dict[str, bool]:
    caps = capabilities_for_backend(chat_backend)
    if "llama.cpp" in chat_backend.lower():
        st.warning(
            "llama.cpp backend is chat/output-only right now. Activation Trace, Logit Lens, Attention, Map, Steer, and activation Compare are disabled until llama.cpp-glass exposes real hooks."
        )
    return caps

'''


HF_PANEL_HELPER = '''

def render_hf_catalog_panel() -> None:
    st.markdown("### Hugging Face Access")
    token = st.text_input(
        "HF token",
        value=st.session_state.get("hf_token", ""),
        type="password",
        help=HELP["hf_token"],
    )
    st.session_state.hf_token = token

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Validate token", width="stretch"):
            st.session_state.hf_token_status = validate_token(token)
    with c2:
        if st.button("Clear token", width="stretch"):
            st.session_state.hf_token = ""
            st.session_state.hf_token_status = None
            st.session_state.hf_model_access_cache = {}
            st.session_state.hf_last_load_plan = None

    token_status = st.session_state.get("hf_token_status")
    token_valid = bool(token_status and token_status.valid)
    if token_status is None:
        st.caption("HF token: not checked")
    elif token_status.valid:
        st.success(token_status.label())
    else:
        st.error(token_status.label())

    st.markdown("### Official model catalog")
    st.caption(HELP["hf_catalog"])
    fam_options = ["All"] + families()
    fam = st.selectbox(
        "Family",
        fam_options,
        index=fam_options.index(st.session_state.get("hf_selected_family", "All")) if st.session_state.get("hf_selected_family", "All") in fam_options else 0,
    )
    st.session_state.hf_selected_family = fam
    recommended_only = st.toggle("Recommended practical first", value=bool(st.session_state.get("hf_recommended_only", False)))
    st.session_state.hf_recommended_only = recommended_only

    rows = visible_models(fam, recommended_only=recommended_only)
    if not rows:
        st.info("No models match the current filters.")
        return

    labels = [f"{m.family} · {m.display_name} · {m.repo_id}" for m in rows]
    selected_label = st.selectbox("Model", labels)
    selected = rows[labels.index(selected_label)]
    st.session_state.hf_selected_repo = selected.repo_id

    cache = st.session_state.setdefault("hf_model_access_cache", {})
    access = cache.get(selected.repo_id)
    access_status = access.get("status") if isinstance(access, dict) else None
    state, reason, enabled = model_state(selected, token_valid, access_status)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Params", "?" if selected.params_b is None else f"{selected.params_b:g}B")
    with col_b:
        st.metric("Trace", selected.trace_level)
    with col_c:
        st.metric("Access", state)

    st.code(selected.repo_id)
    st.caption(reason)
    st.caption(selected.notes)

    a1, a2 = st.columns(2)
    with a1:
        if st.button("Check model access", width="stretch"):
            checked = check_model_access(selected.repo_id, token=token if token_valid else None)
            cache[selected.repo_id] = checked.__dict__
            st.session_state.hf_model_access_cache = cache
            st.rerun()
    with a2:
        if st.button("Plan HF load", width="stretch", disabled=not enabled):
            plan = build_hf_load_plan(selected.repo_id, token=token if token_valid else None)
            st.session_state.hf_last_load_plan = plan.__dict__

    if access:
        st.info(f"Hub access: {access_badge_text(type('HFModelAccessView', (), access)())}")
    if st.session_state.get("hf_last_load_plan"):
        st.json(st.session_state.hf_last_load_plan)

    st.dataframe(pd.DataFrame(registry_as_dicts()), width="stretch", height=220, hide_index=True)

'''


def main() -> None:
    if not MAIN.exists():
        raise SystemExit("Run this from the glass-skull repo root")

    text = MAIN.read_text(encoding="utf-8")

    text = text.replace("use_container_width=True", 'width="stretch"')
    text = text.replace("use_container_width=False", 'width="content"')

    if "PLOT_COUNTER = 0" not in text:
        text = replace_once(text, "ui.inject_theme()\n\nHELP =", "ui.inject_theme()\n\nPLOT_COUNTER = 0\n\nHELP =")

    if "def plot_if_present(fig, key_hint: str = \"plot\")" not in text:
        old = '''def plot_if_present(fig) -> None:
    if fig is not None:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=ui.TEXT, family="Inter, sans-serif"),
            title_font=dict(color=ui.TEXT, size=15),
        )
        st.plotly_chart(fig, width="stretch")
'''
        new = '''def plot_if_present(fig, key_hint: str = "plot") -> None:
    global PLOT_COUNTER
    if fig is not None:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=ui.TEXT, family="Inter, sans-serif"),
            title_font=dict(color=ui.TEXT, size=15),
        )
        PLOT_COUNTER += 1
        st.plotly_chart(fig, width="stretch", key=f"{key_hint}_{PLOT_COUNTER}")
'''
        text = replace_once(text, old, new, required=False)

    import_anchor = "from glass_skull.feature_store import compatible_features, list_features, load_feature, save_feature\n"
    text = ensure_import_after(text, import_anchor, "from glass_skull.hf_access import access_badge_text, check_model_access, validate_token\n")
    text = ensure_import_after(text, import_anchor, "from glass_skull.hf_loader import build_hf_load_plan\n")
    text = ensure_import_after(text, import_anchor, "from glass_skull.hf_registry import capabilities_for_backend, families, model_state, registry_as_dicts, visible_models\n")

    if '"hf_token": "Optional Hugging Face read token' not in text:
        text = replace_once(
            text,
            '    "compare": "Runs the same prompt normally and with steering, then shows text and activation differences.",\n}',
            '    "compare": "Runs the same prompt normally and with steering, then shows text and activation differences.",\n'
            '    "hf_token": "Optional Hugging Face read token. Required for gated models after you accept the model license/access terms on Hugging Face.",\n'
            '    "hf_catalog": "Official HF model catalog. Visible does not mean loadable; loadability depends on token, access, hardware, and adapter support.",\n'
            '}',
        )

    defaults_slice = text[text.find("defaults = {"):text.find("defaults = {") + 1600]
    if '"dashboard_trace"' not in defaults_slice:
        text = replace_once(
            text,
            '        "last_comparison": None,\n',
            '        "last_comparison": None,\n'
            '        "dashboard_trace": None,\n'
            '        "dashboard_trace_meta": {},\n'
            '        "dashboard_trace_counter": 0,\n',
        )
    defaults_slice = text[text.find("defaults = {"):text.find("defaults = {") + 1600]
    if '"hf_token"' not in defaults_slice:
        text = replace_once(
            text,
            '        "poke_strength": 1.5,\n',
            '        "poke_strength": 1.5,\n'
            '        "hf_token": "",\n'
            '        "hf_token_status": None,\n'
            '        "hf_model_access_cache": {},\n'
            '        "hf_selected_family": "All",\n'
            '        "hf_recommended_only": False,\n'
            '        "hf_selected_repo": "",\n'
            '        "hf_last_load_plan": None,\n',
        )

    text = text.replace(
        '            st.session_state.last_comparison = None\n            load_hooked_model.clear()\n',
        '            st.session_state.last_comparison = None\n            st.session_state.dashboard_trace = None\n            st.session_state.dashboard_trace_meta = {}\n            load_hooked_model.clear()\n',
    )
    text = text.replace(
        '        if st.button("Clear trace", width="stretch", help="Clears the currently cached activations."):\n            st.session_state.trace = None\n',
        '        if st.button("Clear trace", width="stretch", help="Clears the currently cached activations."):\n            st.session_state.trace = None\n            st.session_state.dashboard_trace = None\n            st.session_state.dashboard_trace_meta = {}\n',
    )

    if "def set_dashboard_trace(" not in text:
        text = insert_after(text, HELPER_ANCHOR, SET_DASHBOARD_TRACE)
    if "def render_capability_warning(" not in text:
        text = insert_after(text, HELPER_ANCHOR, CAPABILITY_HELPER)
    if "def render_hf_catalog_panel(" not in text:
        text = insert_after(text, HELPER_ANCHOR, HF_PANEL_HELPER)

    sidebar_anchor = '''        elif glass_status is not None:
            st.caption(f"Glass error: {glass_status.error or 'no details'}")

    with st.expander("Session", expanded=False):
'''
    sidebar_add = '''
    with st.expander("Hugging Face", expanded=False):
        render_hf_catalog_panel()

'''
    text = insert_after(text, sidebar_anchor, sidebar_add)

    if 'ui.pill("HF token"' not in text:
        text = replace_once(
            text,
            '    ui.pill("Glass server", ui.server_status_color(st.session_state.llama_glass_status)),\n',
            '    ui.pill("Glass server", ui.server_status_color(st.session_state.llama_glass_status)),\n'
            '    ui.pill("HF token", ui.GREEN if (st.session_state.get("hf_token_status") and st.session_state.hf_token_status.valid) else ui.SLATE),\n',
        )

    text = text.replace(
        '    trace = st.session_state.trace\n    if trace is None:\n        ui.empty_state("No trace captured yet", "Send a message in the Chat tab with tracing enabled to populate these graphs.")\n    else:\n        st.code(trace.prompt)\n',
        '    trace = st.session_state.get("dashboard_trace") or st.session_state.get("trace")\n    dash_meta = st.session_state.get("dashboard_trace_meta", {}) or {}\n    if trace is None:\n        ui.empty_state("No trace captured yet", "Send a message in the Chat tab with tracing enabled to populate these graphs.")\n    else:\n        ui.property_list([\n            ("updated", str(dash_meta.get("updated_at", "current run"))),\n            ("backend", str(dash_meta.get("backend", "unknown"))),\n            ("trace_model", str(dash_meta.get("trace_model", summary["model_name"]))),\n            ("tokens", str(dash_meta.get("token_count", len(trace.tokens)))),\n            ("run", str(dash_meta.get("run", "-"))),\n        ])\n        st.code(trace.prompt)\n',
    )

    chat_anchor = '''    with cfg3:
        temperature = st.slider("Temperature", 0.01, 1.5, 0.8, 0.05, help=HELP["temperature"], key="chat_temp")

    tog1, tog2 = st.columns(2)
'''
    chat_add = '''
    caps = render_capability_warning(chat_backend)

'''
    text = insert_after(text, chat_anchor, chat_add)

    text = text.replace(
        '        use_steering = st.toggle("Use steering", value=False, help="Only applies when the chat backend is TransformerLens. Stock llama.cpp cannot be activation-steered yet.")',
        '        use_steering = st.toggle("Use steering", value=False, disabled=not caps["activation_steering"], help="Only applies when the chat backend is TransformerLens. Stock llama.cpp cannot be activation-steered yet.")',
    )

    text = text.replace(
        '                trace = trace_prompt(model, prompt)\n                st.session_state.trace = trace\n                run_id = log_run(model_name=model_name, mode="chat_trace", prompt=prompt, metadata={"tokens": trace.tokens, "summary": summary})\n',
        '                trace = trace_prompt(model, prompt)\n                st.session_state.trace = trace\n                set_dashboard_trace(trace, prompt, chat_backend, str(summary["model_name"]))\n                run_id = log_run(model_name=model_name, mode="chat_trace", prompt=prompt, metadata={"tokens": trace.tokens, "summary": summary, "chat_backend": chat_backend})\n',
    )

    trace_anchor = '''with tab_trace:
    ui.section_header("Trace / Lens", "Visualize model internals captured from the latest traced prompt.")
'''
    trace_add = '''    if 'chat_backend' in locals() and "llama.cpp" in chat_backend.lower():
        ui.empty_state("Trace source is not llama.cpp", "The current llama.cpp backend can chat, but it does not expose activations. Switch to TransformerLens/HF trace mode for Logit Lens, Attention, and steering.")

'''
    text = insert_after(text, trace_anchor, trace_add)

    poke_anchor = '''with tab_poke:
    ui.section_header("Poke / Compare / Fuzz", "Probe and stress-test model behavior.")
'''
    poke_add = '''    if 'chat_backend' in locals() and "llama.cpp" in chat_backend.lower():
        st.info("llama.cpp selected: chat and fuzz output are available, but activation Map/Steer/Compare controls target TransformerLens only until llama.cpp-glass exposes hooks.")

'''
    text = insert_after(text, poke_anchor, poke_add)

    text = text.replace(
        '["Anatomy", "Hooks", "Parameters", "Experiments", "Features", "Logs"]',
        '["Anatomy", "Hooks", "Parameters", "Experiments", "Features", "HF Catalog", "Logs"]',
    )
    anatomy_logs_anchor = '''    elif panel == "Features":
        st.caption(f"Compatible with current d_model {expected_dim}: {len(compatible_feature_names)} / {len(all_feature_names)}")
        st.dataframe(pd.DataFrame(all_features), width="stretch", height=590, hide_index=True)
    elif panel == "Logs":
'''
    hf_catalog_panel = '''    elif panel == "Features":
        st.caption(f"Compatible with current d_model {expected_dim}: {len(compatible_feature_names)} / {len(all_feature_names)}")
        st.dataframe(pd.DataFrame(all_features), width="stretch", height=590, hide_index=True)
    elif panel == "HF Catalog":
        st.dataframe(pd.DataFrame(registry_as_dicts()), width="stretch", height=590, hide_index=True)
    elif panel == "Logs":
'''
    if anatomy_logs_anchor in text:
        text = replace_once(text, anatomy_logs_anchor, hf_catalog_panel)

    MAIN.write_text(text, encoding="utf-8")
    print("HF front patch applied to main.py")
    print("Dashboard live trace snapshot enabled")


if __name__ == "__main__":
    main()
