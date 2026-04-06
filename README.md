# Cascade

**Prompt-driven generation of executable visual node graphs in volumetric space via agentic chain-of-influence synthesis.**

![Screenshot 2026-04-05 170044](https://github.com/user-attachments/assets/fde0562f-8b8d-4ad2-ad31-73d4b1d1e7d3)

![Screenshot 2026-04-05 171033](https://github.com/user-attachments/assets/aea4c1e5-8a08-4835-98ca-361149b26169)


![Screenshot 2026-04-05 174639](https://github.com/user-attachments/assets/59d037f0-f5cb-46ec-992d-0c131cc123d6)



---

## What it does

Type a natural-language brief → Cascade generates a fully executable, browser-rendered visual node graph — no shader authoring, no signal routing, no code written by you.

A two-phase agentic pipeline handles everything:

| Phase | What happens |
|---|---|
| **1 — Semantic Divergence** | CLIP extracts brand/emotion from reference images; per-level divergence scores and a RAG knowledge graph produce a structured creative brief |
| **2 — Chain-of-Influence** | Reasoner designs a typed DAG; Influence Compiler lays it out deterministically; Mason generates polyglot code (GLSL / p5.js / Three.js / WebAudio) per node with headless validation |

A **RuntimeInspector** agent intercepts browser errors and hot-reloads LLM-generated fixes autonomously. All edits write back to a canonical session JSON.

---

## Project structure

```
phase1/          Semantic divergence engine (CLIP, RAG, brand extraction)
phase2/          Chain-of-influence pipeline (Reasoner, Compiler, Mason, RuntimeInspector)
static/js/       Browser runtime (WebGL2 executor pipeline, texture hub, grid renderer)
templates/       Flask HTML templates
app_web.py       Flask entry point
config.py        LLM model selection (local Ollama or cloud)
session_manager.py  Session JSON persistence
setup.sh         One-shot environment setup (WSL/Linux)
```

---

## Setup

Requires Python 3.10+, [Ollama](https://ollama.com) (for local inference), and Node.js (headless validation).

```bash
bash setup.sh        # installs Python deps, pulls Ollama models, builds RAG graph
python app_web.py    # starts the Flask dev server at http://localhost:5000
```

**Cloud LLM mode** (Qwen3-Coder 480B / Qwen3.5 for reasoning):

```bash
USE_CLOUD_LLM=1 python app_web.py
```

---

## Models used

| Role | Local | Cloud |
|---|---|---|
| Reasoning / planning | `llama3.2:latest` | `qwen3.5:cloud` |
| Code generation (Mason) | `qwen2.5-coder:7b` | `qwen3-coder:480b-cloud` |
| Review / fallback | `llama3.2:latest` | — |

---

## Browser runtime engines

The generated node graph runs live in the browser on a shared **WebGL2** context:

`GLSL` · `p5.js` · `Three.js` · `Canvas 2D` · `WebAudio` · `html_video` · `ml5.js`



## License

See [LICENSE](LICENSE).
