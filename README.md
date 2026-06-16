# Operation Glass Skull

A local interpretability lab for chatting with a transformer while inspecting and steering its activations.

Version `v0.5` uses a chat-centered cockpit layout:

- 1/4 screen: chat UI
- 1/4 screen: live trace view
- 1/4 screen: poke/map/edge controls
- 1/4 screen: anatomy, hook points, parameters, and logs

This is not a llama.cpp/GGUF project. It uses PyTorch + TransformerLens so the app can see and modify internal activations directly.

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
  accurate partial anatomy + accurate cached activations + accurate selected MLP contribution edges

Not yet:
  complete animated rendering of every transformer operation
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

The default model is:

```text
pythia-70m-deduped
```

Once the app works, try:

```text
pythia-160m-deduped
pythia-410m-deduped
pythia-1b-deduped
```

## Cockpit workflow

### Chat

Use the left panel to talk to the model. Enable `Trace every message` to capture activations for each prompt. Enable `Use steering` to apply the selected feature vector from the Poke panel.

### Trace

The trace panel shows:

- prompt tokens
- activation heatmap
- selected layer/token activation dimensions
- next-token probability table

### Poke

The poke panel has three tabs:

- `Steer`: load a saved feature and set strength
- `Map`: build a new feature from positive and negative examples
- `Edges`: show selected active MLP contribution edges from the current trace

### Anatomy / Logs

The right panel shows:

- model config
- expected block components
- discovered hook points
- parameter tensors
- recent SQLite logs

## Directory layout

```text
glass-skull/
  main.py
  glass_skull/
    __init__.py
    anatomy.py
    config.py
    model_loader.py
    tracer.py
    contribution.py
    steering.py
    feature_store.py
    logger.py
  data/
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

The visual edges are not literal wires. They are top computed contribution paths derived from real activations and real weight matrices.
