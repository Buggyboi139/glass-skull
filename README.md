# Operation Glass Skull

A local interpretability cockpit for chatting with llama.cpp or TransformerLens while inspecting, mapping, fuzzing, and steering transformer activations.

Version `v0.6` adds:

- llama.cpp backend support
- normal llama.cpp server checks
- future patched llama.cpp glass-server checks
- chat routing through TransformerLens or llama.cpp
- prompt-file fuzzing
- batch trace aggregation
- saved experiment folders
- activation pulse visuals
- active contribution constellation visuals
- fuzz heatmaps across prompts, labels, layers, and recurring dimensions

## Layout

The app uses a four-panel cockpit:

- 1/4 screen: chat UI
- 1/4 screen: live trace view
- 1/4 screen: poke, map, edges, and fuzz controls
- 1/4 screen: anatomy, hook points, parameters, experiments, and logs

## Backend model

There are now two separate ideas:

```text
Chat backend:
  Where generated replies come from.

Trace model:
  The TransformerLens model used for local activation tracing, feature mapping, and steering.
```

Supported chat backends:

```text
TransformerLens
llama.cpp normal
llama.cpp glass
```

The `llama.cpp glass` option is for a future patched llama.cpp lab server running on a nonstandard port such as `8088`.

Stock llama.cpp can provide chat outputs. It does not expose activation traces or activation injection yet. The app can still compare llama.cpp outputs against local TransformerLens traces.

## Accuracy scope

The app does not draw a fake classic neural-network cartoon.

The anatomy view is grounded in the loaded TransformerLens model:

- model config
- parameter names
- tensor shapes
- layer count
- hidden size
- attention heads
- MLP size
- discovered hook points
- expected block components

The trace view shows cached activations from actual hook points.

The active edge view currently supports selected MLP matrix contribution edges. It does not yet fully visualize every attention subcomponent, every normalization operation, embeddings, unembedding, or every residual addition as animated edges.

So the honest status is:

```text
Current GUI:
  accurate partial anatomy
  accurate cached TransformerLens activations
  accurate selected MLP contribution edges
  llama.cpp chat availability
  prompt fuzzing and aggregation

Not yet:
  complete animated rendering of every transformer operation
  stock llama.cpp activation traces
  stock llama.cpp activation poking
```

## Quick start

```bash
cd ~/repos/glass-skull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m compileall .
streamlit run main.py
```

The default trace model is:

```text
pythia-70m-deduped
```

Once the app works, try:

```text
pythia-160m-deduped
pythia-410m-deduped
pythia-1b-deduped
```

## llama.cpp setup

Run your normal llama.cpp server separately, for example:

```bash
/path/to/llama-server \
  -m /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080
```

In Glass Skull:

```text
Normal server URL:
  http://127.0.0.1:8080
```

For the future patched lab build:

```text
Glass server URL:
  http://127.0.0.1:8088
```

Use `Check normal` or `Check glass` in the sidebar to test availability.

## Fuzzing

The Fuzz tab can run a file of prompts through a selected chat backend and optionally trace each prompt with the local TransformerLens model.

Supported prompt files:

### TXT

```text
The cat sat on the
Explain what a mammal is.
Write a Python function.
```

### JSONL

```jsonl
{"label":"animal","prompt":"Explain what a mouse is."}
{"label":"vehicle","prompt":"Explain what a car is."}
```

### CSV

```csv
label,prompt
animal,"Explain what a mouse is."
vehicle,"Explain what a car is."
```

Fuzz experiments are saved under:

```text
data/experiments/
```

Each run can produce:

- `config.json`
- `prompts.json`
- `outputs.jsonl`
- `summary.json`
- `prompt_layer_heatmap.csv`
- `label_layer_heatmap.csv`
- `top_recurring_dimensions.csv`

## Cockpit workflow

### Chat

Use the left panel to talk to the selected backend. Enable `Trace every message` to capture activations with the local trace model. Enable `Use steering` to apply the selected feature vector when the backend is TransformerLens.

### Trace

The trace panel shows:

- prompt tokens
- activation pulse visual
- activation heatmap
- selected layer/token activation dimensions
- next-token probability table

### Poke / Fuzz

The poke panel has four tabs:

- `Steer`: load a saved feature and set strength
- `Map`: build a new feature from positive and negative examples
- `Edges`: show selected active MLP contribution edges from the current trace
- `Fuzz`: rapid-fire prompt files and generate heatmaps

### Anatomy / Logs

The right panel shows:

- model config
- expected block components
- discovered hook points
- parameter tensors
- saved experiments
- recent SQLite logs

## Directory layout

```text
glass-skull/
  main.py
  glass_skull/
    __init__.py
    aggregation.py
    anatomy.py
    config.py
    contribution.py
    experiment_store.py
    feature_store.py
    fuzzing.py
    llama_client.py
    logger.py
    model_loader.py
    prompt_loader.py
    steering.py
    tracer.py
    visuals.py
  data/
    experiments/
    features/
    logs/
    prompt_sets/
  requirements.txt
```

## Concepts

Weights are fixed learned tensors. Activations are the live values that move through the model during a prompt.

This app visualizes the active path by showing:

- model anatomy
- available hook points
- layer activation norms
- top active residual dimensions
- selected active contribution edges
- saved steering vectors
- normal vs steered output
- fuzzed activation patterns over many prompts

The visual edges are not literal wires. They are top computed contribution paths derived from real activations and real weight matrices.
