✨Vibe-Code✨

# Operation Glass Skull

A local interpretability cockpit for chatting with llama.cpp, TransformerLens, and planned Hugging Face model backends while inspecting, mapping, fuzzing, comparing, and steering transformer activations.

## Current status

Glass Skull now separates three worlds:

```text
TransformerLens:
  Full local interpretability path.
  Activation traces, attention, Logit Lens, feature mapping, steering, comparison.

llama.cpp:
  Fast GGUF chat path.
  Chat, server checks, metadata, fuzz output.
  No activation hooks until llama.cpp-glass grows real endpoints.

Hugging Face front:
  Official model catalog, token validation, gated model access checks, and future generic HF loading.
```

Visible does not mean loadable. Loadable does not mean traceable. Traceable does not mean steerable. This is the app refusing to lie, a bold innovation in software.

## HF front

New modules:

```text
glass_skull/hf_registry.py
  Official model registry under 70B.
  Gemma, Qwen, Llama, Mistral, Phi, DeepSeek, and TransformerLens baselines.

glass_skull/hf_access.py
  Hugging Face token validation.
  Model access checks.
  Gated/private/public status helpers.

glass_skull/hf_loader.py
  Generic HF loading scaffold.
  Not yet a full activation hook adapter.

scripts/apply_hf_front_patch.py
  Local UI patcher for main.py.
  Used to preserve local UI polish without overwriting your uncommitted Streamlit edits.
```

## Applying the HF UI patch

From the repo root:

```bash
python scripts/apply_hf_front_patch.py
python -m compileall .
python smoke_check.py
streamlit run main.py
```

The patch adds:

```text
Sidebar:
  Hugging Face token input
  Validate token
  Clear token
  Official model catalog
  Family filter
  Recommended-only toggle
  Model access check
  HF load plan preview

HUD:
  HF token status pill

Chat:
  llama.cpp capability warning
  steering toggle disabled when llama.cpp is selected

Trace / Lens:
  warning when llama.cpp is selected because stock llama.cpp does not expose activations

Poke / Compare / Fuzz:
  warning that activation controls target TransformerLens only unless llama.cpp-glass grows hooks

Anatomy / Logs:
  HF Catalog table
```

## Initial official model registry

Families included:

```text
TransformerLens:
  pythia-70m-deduped
  pythia-160m-deduped
  pythia-410m-deduped
  pythia-1b-deduped

Gemma:
  google/gemma-4-E2B
  google/gemma-4-E2B-it
  google/gemma-4-E4B
  google/gemma-4-E4B-it
  google/gemma-4-12B
  google/gemma-4-12B-it
  google/gemma-4-26B-A4B-it
  google/gemma-4-31B-it

Qwen:
  Qwen/Qwen3-0.6B
  Qwen/Qwen3-1.7B
  Qwen/Qwen3-4B
  Qwen/Qwen3-8B
  Qwen/Qwen3-14B
  Qwen/Qwen3-30B-A3B
  Qwen/Qwen3-32B
  Qwen/Qwen3.6-27B        pending Hub validation
  Qwen/Qwen3.6-35B-A3B    pending Hub validation

Llama:
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  meta-llama/Llama-3.1-70B-Instruct
  meta-llama/Llama-3.3-70B-Instruct

Mistral:
  mistralai/Mistral-7B-Instruct-v0.3
  mistralai/Mistral-Small-24B-Instruct-2501
  mistralai/Mistral-Small-3.1-24B-Instruct-2503
  mistralai/Mistral-Small-3.2-24B-Instruct-2506

Phi:
  microsoft/Phi-3.5-mini-instruct
  microsoft/Phi-3.5-MoE-instruct
  microsoft/Phi-4
  microsoft/Phi-4-mini-instruct
  microsoft/Phi-4-reasoning
  microsoft/Phi-4-mini-reasoning

DeepSeek:
  deepseek-ai/deepseek-coder-33b-instruct
  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
  deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  deepseek-ai/DeepSeek-R1-Distill-Llama-70B
```

## Backend capability rules

When chat backend is `Local GGUF normal (llama.cpp)` or `Local GGUF steered (llama.cpp)`:

```text
Enabled:
  Chat
  Server metadata
  Fuzz outputs
  Control-vector steering through llama.cpp startup flags

Disabled / warning-gated:
  Activation Trace
  Logit Lens
  Attention
  Map
  TransformerLens-style activation Steer
  Tensor-level Activation Compare
```

When backend is TransformerLens:

```text
Enabled:
  Activation Trace
  Logit Lens
  Attention
  Map
  Steer
  Compare
  Fuzz trace
```

## Quick start

```bash
cd ~/repos/glass-skull
source .venv/bin/activate
pip install -r requirements.txt
python scripts/apply_hf_front_patch.py
python -m compileall .
python smoke_check.py
streamlit run main.py
```

On first boot, Glass Skull opens a workflow setup dialog. Pick the model sources
for the session:

- Local GGUF (llama.cpp)
- Hugging Face models
- Trace model (TransformerLens)

Each source includes an info tooltip with its runtime requirements. Successful
setup adds the configured sources to the `Models` tab, which appears after
`Anatomy / Logs`.

Runtime configuration lives in the `Settings` tab, not the sidebar. `Settings`
is organized as `Local`, `HF`, `Trace`, and `Session`; start in `Local` to set
the router model alias, GGUF path, llama.cpp server URLs, and local tool paths.
Chat, fuzzing, Local Alter launch commands, and local model flags all consume
that same Local configuration.

The Chat tab includes `Send message`, `Cancel chat`, `New chat`, and `Load chat`
controls beside the message box. `New chat` archives the current transcript
under `data/chats/`; `Load chat` restores the most recent saved transcript.

## llama.cpp reminder

Glass Skull can now stage local GGUF control-vector runs around upstream
llama.cpp flags. Put positive/negative prompt files under
`data/control_sets/`, generate vectors under `data/control_vectors/`, then
use the Local Alter tab to preflight the model and build launch commands. The
generator omits `-ngl` by default so llama.cpp can auto-fit GPU layers; set it
only from Advanced generator options when you need an explicit value.

Launch a normal server and a steered server on separate ports, then compare
`Local GGUF normal (llama.cpp)` and `Local GGUF steered (llama.cpp)` in Chat:

```bash
/home/dsmason321/llama.cpp/build/bin/llama-server \
  -m /path/to/qwen3.6-35b.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -ngl 999 \
  --alias qwen3.6-35b-mtp-q4-ks-vision

/home/dsmason321/llama.cpp/build/bin/llama-server \
  -m /path/to/qwen3.6-35b.gguf \
  --host 127.0.0.1 \
  --port 8088 \
  -ngl 999 \
  --alias qwen3.6-35b-mtp-q4-ks-vision \
  --control-vector-scaled data/control_vectors/my_behavior.gguf:1.25 \
  --control-vector-layer-range 20 60
```

Local Alter also preserves full stdout/stderr for failed generator attempts and
classifies common failures, including explicit `-ngl 999` auto-fit conflicts and
the likely Qwen3.6 MoE/MTP `diff_filtered.size() == n_layers - 1` assertion.

For Gemma 4 models, disable reasoning if the chat endpoint returns empty visible content:

```bash
~/repos/llama.cpp-glass/build-glass/bin/llama-server \
  -m /path/to/model.gguf \
  --alias default \
  --host 127.0.0.1 \
  --port 8088 \
  -ngl 999 \
  -c 32768 \
  -b 2048 \
  -ub 2048 \
  --jinja \
  --reasoning-budget 0 \
  --flash-attn \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --no-mmap
```

## Accuracy scope

The anatomy view is grounded in the loaded TransformerLens model. The llama.cpp model card can expose GGUF metadata, but not real attention/MLP/residual internals until llama.cpp-glass gets dedicated trace endpoints.
