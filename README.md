# Operation Glass Skull

Glass Skull is a local Streamlit cockpit for running GGUF models through llama.cpp and visualizing the activity returned by the local Glass Skull trace endpoint.

The app is intentionally local-only:

- chat goes to a local llama.cpp server
- model metadata comes from the configured GGUF file
- trace data comes from `/glass-skull/trace` when the managed llama.cpp patch is present
- control vectors are generated and launched with local llama.cpp tools
- run artifacts, chats, logs, and control data stay under `data/`

## Quick Start

```bash
cd ~/repos/glass-skull
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall main.py glass_skull scripts smoke_check.py
python smoke_check.py
streamlit run main.py
```

Or use the launcher:

```bash
START_LLAMA_CPP=0 ./run_glass_skull.sh
```

Set `START_LLAMA_CPP=1` and configure these environment variables when you want the launcher to start llama.cpp too:

```bash
LLAMA_MODEL_PATH=/path/to/model.gguf
LLAMA_MODEL_ALIAS=local-model
LLAMA_PORT=8080
./run_glass_skull.sh
```

By default, the launcher starts the repo-managed patched server at
`managed/llama.cpp-glass/build/bin/llama-server`. Set `LLAMA_SERVER_BIN=/path/to/llama-server`
only when intentionally overriding that managed binary.

## Local llama.cpp

The managed local source lives at:

```text
managed/llama.cpp-glass/
```

The local patch file lives at:

```text
patches/llama.cpp-glass/0001-glass-skull-per-request-steering.patch
```

Use the setup helper when the managed checkout needs to be prepared:

```bash
python scripts/setup_llama_cpp_glass.py
```

Expected local binaries:

```text
managed/llama.cpp-glass/build/bin/llama-server
managed/llama.cpp-glass/build/bin/llama-cvector-generator
```

The app settings default to these managed binary paths. External local builds can still be used by explicitly setting `LLAMA_SERVER_BIN` or editing the app settings fields.

## App Workflow

1. Configure the GGUF path, model alias, and llama.cpp URLs in `Settings`.
2. Use `Model` to inspect GGUF metadata and check the normal and steered servers.
3. Send a prompt in `Run`.
4. If the local server exposes `/glass-skull/trace`, Glass Skull stores prompt tokens and layer input summaries.
5. The app generates a run artifact containing prompt, output, trace rows, behavior scores, and diagnostics.
6. `Map` renders the activation canvas and supporting tables.
7. `Steer` creates control sets, generates local control vectors, and shows server launch commands.
8. `Timeline` compares behavior scores across local runs.

## Data Layout

```text
data/chats/              saved local transcripts
data/control_sets/       positive/negative prompt sets
data/control_vectors/    generated GGUF control vectors and metadata
data/experiments/        batch run artifacts and CSV summaries
data/logs/               SQLite run log and llama.cpp log output
data/prompt_sets/        tracked sample prompt sets
```

Runtime data is ignored by git except for tracked sample prompts.

## Validation

```bash
python -m compileall main.py glass_skull scripts smoke_check.py
python smoke_check.py
```

The smoke check stubs local llama.cpp HTTP calls, verifies run artifacts and activation-map payloads, checks GGUF tensor parsing, and exercises local batch execution.
