# Arogya AI Backend

FastAPI backend for Arogya AI. The repository currently contains a rules-based symptom-processing pipeline and condition-specific data modules. Phase 1 introduces a provider-routing API layer while retaining the existing Flutter client's /chat endpoint.

## Phase 1 API layer

- Backward-compatible POST /chat and versioned POST /v1/chat
- Liveness endpoints: GET /health and GET /v1/health
- Readiness endpoints: GET /ready and GET /v1/ready
- Configurable provider routing: legacy (default) or local Ollama
- Request IDs, structured errors, configurable CORS, input bounds, and timeout handling
- Precautionary urgent-symptom keyword gate (not a medically validated triage system)

## Run locally

Create a Python virtual environment, activate it, then run:

    pip install -r requirements.txt
    # Copy .env.example to .env (Windows PowerShell: Copy-Item .env.example .env)
    uvicorn api.main:app --reload

Open http://127.0.0.1:8000/docs for the API schema. With the default AI_PROVIDER=legacy, the current rules-based engine is used.

## Optional local Ollama provider

Install Ollama and pull a model suitable for your hardware, then set the following values in .env:

    AI_PROVIDER=ollama
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=llama3.2:3b
    OLLAMA_TIMEOUT_SECONDS=45

Restart the API and check /ready. The adapter uses Ollama's non-streaming /api/chat endpoint. This does not fine-tune the model. A medical JSON dataset may be better suited to retrieval-augmented generation (RAG); fine-tuning requires a separate training and evaluation pipeline.

## Run tests

    pip install pytest
    pytest -q

## Limitations before a public launch

- Session state is in process memory: it is not durable, cross-worker safe, authenticated, or appropriate for storing real patient histories.
- The urgent-symptom gate is intentionally small and can miss emergencies or flag educational questions. It must not be treated as a clinical triage tool.
- Neither model outputs nor rule-based heuristic scores are validated diagnoses or calibrated probabilities.
- Do not commit patient data, credentials, .env files, or private datasets.
- A public healthcare release requires authentication and authorization, explicit consent, secure storage and transport, retention/deletion controls, auditability, clinical evaluation, and a security/privacy review.
