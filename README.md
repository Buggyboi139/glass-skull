# Operation Glass Skull

A local interpretability lab for inspecting and steering transformer activations.

Version `v0.4` focuses on a practical local workflow:

- Load a TransformerLens-supported model
- Inspect model anatomy from actual config, parameters, and hook points
- Run prompts through the model
- Capture activation caches
- Display token/layer activation heatmaps
- Inspect top active dimensions
- Build contrast vectors from positive/negative prompt sets
- Save and reload feature vectors
- Steer generation with activation hooks
- Log runs to SQLite

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

The animated/visual edges are not literal wires. They are top computed contribution paths derived from real activations and real weight matrices.

## Run levels

### v0.1
Capture activations.

### v0.2
Inspect top active dimensions and contribution edges.

### v0.3
Build feature vectors from prompt contrasts.

### v0.4
Inject saved vectors during generation and compare output.
