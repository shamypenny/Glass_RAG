# GLASS-RAG

LangChain-based Python implementation for the `GLASS-RAG` project described in `GLASS_RAG_产品详细说明文档.md`.

## What Is Included

- Configurable multi-stage LLM setup from YAML, with no model hardcoding in the business pipeline
- MongoDB-backed document storage, chunk storage, audit logs, and session history
- Ingestion flow: parsing, OCR abstraction, structured extraction, chunking, embeddings, vector indexing
- Query flow: intent routing, keyword/vector/hybrid retrieval, reranking, prompt assembly, cited answering
- Langfuse callback integration for tracing
- FastAPI endpoints for document ingestion and query serving

## Project Structure

```text
glass_rag/
  api/main.py
  config.py
  factories.py
  governance.py
  observability.py
  schemas.py
  ingestion/
    parser.py
    chunker.py
    pipeline.py
  pipelines/
    query_pipeline.py
  prompts/
    templates.py
  retrieval/
    retrievers.py
    reranker.py
  services/
    application.py
  storage/
    mongo.py
config/
  app.example.yaml
```

## Setup

1. Create a virtual environment and install dependencies.

```bash
pip install -e .
```

2. Copy `.env.example` to `.env` and fill in provider keys.

3. Adjust `config/app.example.yaml` or point `GLASS_RAG_CONFIG` to your own YAML file.

4. Prepare MongoDB:

- The code creates standard Mongo indexes automatically.
- For production vector search on MongoDB Atlas, create the Atlas Vector Search index whose name matches `mongodb.vector_index_name`.

## Run

```bash
uvicorn glass_rag.api.main:app --reload
```

## Notes

- `component_bindings` maps each pipeline stage to a named model profile.
- `llm_profiles` and `embedding_profiles` are the only places where provider/model names are set.
- If you want to add Anthropic, Gemini, DashScope, or other providers, extend `glass_rag/factories.py`.
