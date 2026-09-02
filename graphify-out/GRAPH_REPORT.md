# Graph Report - parakeet-server  (2026-09-02)

## Corpus Check
- 2 files · ~3,370 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 52 nodes · 75 edges · 6 communities (5 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d3e58cf4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_BatchProcessor|BatchProcessor]]
- [[_COMMUNITY_app.py|app.py]]
- [[_COMMUNITY_load_model|load_model]]
- [[_COMMUNITY_.submit|.submit]]
- [[_COMMUNITY__materialize_items|_materialize_items]]
- [[_COMMUNITY_README|README.md]]

## God Nodes (most connected - your core abstractions)
1. `BatchProcessor` - 12 edges
2. `load_model()` - 9 edges
3. `lifespan()` - 7 edges
4. `_extract_text()` - 6 edges
5. `transcribe_rest()` - 6 edges
6. `_materialize_items()` - 5 edges
7. `_result_to_segments()` - 5 edges
8. `_BatchItem` - 4 edges
9. `health_check()` - 4 edges
10. `_log_active_providers()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `lifespan()` --calls--> `load_model()`  [EXTRACTED]
  app.py → app.py  _Bridges community 0 → community 2_
- `lifespan()` --references--> `FastAPI`  [EXTRACTED]
  app.py →   _Bridges community 0 → community 1_
- `transcribe_rest()` --calls--> `load_model()`  [EXTRACTED]
  app.py → app.py  _Bridges community 2 → community 1_
- `_extract_text()` --calls--> `_materialize_items()`  [EXTRACTED]
  app.py → app.py  _Bridges community 4 → community 2_

## Import Cycles
- None detected.

## Communities (6 total, 1 thin omitted)

### Community 0 - "BatchProcessor"
Cohesion: 0.21
Nodes (6): BatchProcessor, lifespan(), Collects concurrent transcription requests and processes them with     controlle, Wait for at least one item, then collect up to max_batch_size         within the, Background loop: collect batches and process items., Startup and shutdown logic

### Community 1 - "app.py"
Cohesion: 0.18
Nodes (11): _ensure_response_segments(), _ensure_tensorrt_on_ld_path(), main(), Readiness probe: 503 until the ASR model is actually loaded.      /health answer, Handles audio transcription via REST API (OpenAI compatible).     Requests are q, Main application entry point., Make the pip-installed TensorRT (tensorrt-cu12-libs) visible to     onnxruntime., readiness() (+3 more)

### Community 2 - "load_model"
Cohesion: 0.18
Nodes (12): _build_providers(), _ensure_onnx_export(), _extract_text(), health_check(), load_model(), _log_active_providers(), Log the execution providers each ONNX Runtime session actually got.      onnx-as, Build ONNX Runtime provider list based on configuration. (+4 more)

### Community 3 - ".submit"
Cohesion: 0.29
Nodes (5): _BatchItem, A single queued transcription request., Submit audio for transcription.  Blocks until result is ready.         Raises as, Blocking inference — called inside the thread-pool executor., ndarray

### Community 4 - "_materialize_items"
Cohesion: 0.33
Nodes (6): _avg_logprob(), _materialize_items(), Normalize any onnx-asr recognize() return value into a list of result     items, Mean token log-probability for a result item, or None when unavailable., Convert an onnx-asr result into OpenAI verbose_json compatible segments.      Ma, _result_to_segments()

## Knowledge Gaps
- **1 isolated node(s):** `parakeet-server`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BatchProcessor` connect `BatchProcessor` to `app.py`, `.submit`?**
  _High betweenness centrality (0.300) - this node is a cross-community bridge._
- **Why does `transcribe_rest()` connect `app.py` to `load_model`, `.submit`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `load_model()` connect `load_model` to `BatchProcessor`, `app.py`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **What connects `Startup and shutdown logic`, `Make the pip-installed TensorRT (tensorrt-cu12-libs) visible to     onnxruntime.`, `A single queued transcription request.` to the rest of the system?**
  _21 weakly-connected nodes found - possible documentation gaps or missing edges._