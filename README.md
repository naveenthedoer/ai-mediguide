# {PROJECT_NAME} — Production Multi-Agent RAG System on GCP

> A production-grade multi-agent AI system built with Google ADK, Vertex AI Search, and Agent Runtime. Routes user queries through specialist AI agents grounded in a custom document corpus, then synthesizes a unified, cited response.

Built with `google-adk`, `agents-cli v0.3.1`, Vertex AI Search, and Vertex AI Agent Runtime.

---

## How it works

```
User query
    ↓
{orchestrator_agent} (SequentialAgent)
    ↓
parallel_retrieval (ParallelAgent) — all specialists fire simultaneously
    ├── {specialist_agent_1}   → VertexAiSearchTool → corpus
    ├── {specialist_agent_2}   → VertexAiSearchTool → corpus
    ├── {specialist_agent_3}   → VertexAiSearchTool → corpus
    └── {specialist_agent_4}   → VertexAiSearchTool → corpus
    ↓
synthesizer_agent — citations + confidence score + domain-specific flags
    ↓
Unified response
```

**Why this pattern?**

- `ParallelAgent` fires all specialists simultaneously — reduces latency from N×T to ~T
- `SequentialAgent` wraps parallel retrieval + synthesis — ensures retrieval is complete before synthesis begins
- One shared data store per project — domain separation through system prompt instructions, not separate indexes

---

## Stack

| Component       | Technology                                                   |
| --------------- | ------------------------------------------------------------ |
| Agent framework | `google-adk ≥1.15`, `agents-cli v0.3.1`                      |
| Model           | `gemini-2.5-flash` (configurable via `GEMINI_MODEL` env var) |
| RAG retrieval   | Vertex AI Search (`discoveryengine.googleapis.com`)          |
| Deployment      | Vertex AI Agent Runtime                                      |
| Memory          | `VertexAiMemoryBankService` — cross-session context          |
| Observability   | Cloud Trace · Cloud Monitoring · GCS telemetry logs          |
| Infrastructure  | Terraform (`deployment/terraform/single-project/`)           |
| Python tooling  | `uv` package manager                                         |

---

## Project structure

```
{project_name}/
├── app/
│   ├── agent.py                  # Specialist agents + orchestrator definition
│   ├── agent_runtime_app.py      # AgentEngineApp + telemetry + Memory Bank
│   ├── retrievers.py             # VertexAiSearchTool factory
│   └── app_utils/
│       ├── telemetry.py          # OpenTelemetry + GCS log upload (scaffold)
│       ├── observability.py      # Custom domain metrics + trace helpers
│       └── typing.py             # Feedback model
├── tests/
│   └── eval/
│       ├── eval_config.yaml      # Custom metrics (LLM judge + Python functions)
│       └── datasets/
│           └── {project}-dataset.json   # Eval cases with reference answers
├── deployment/
│   ├── terraform/single-project/ # Vertex AI Search infrastructure (Terraform)
│   └── monitoring_alerts.json    # Cloud Monitoring alerting policies
├── sample_data/                  # Source documents for corpus ingestion
├── agents-cli-manifest.yaml      # Project config (read by all agents-cli commands)
└── pyproject.toml
```

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) — authenticated with a GCP project that has billing enabled
- [agents-cli](https://pypi.org/project/google-agents-cli/) — `uv tool install google-agents-cli`
- [Terraform](https://developer.hashicorp.com/terraform/install) — for corpus infrastructure provisioning

---

## Setup from scratch

Follow these steps in order. Each step depends on the previous one.

### Step 1 — Authenticate

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
gcloud config set project YOUR_PROJECT_ID
```

> **Why three auth commands?** `gcloud auth login` is for the CLI. `application-default login` is for Python SDKs. `set-quota-project` tells the Discovery Engine API which project to bill — without it you get a 403.

### Step 2 — Enable APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  discoveryengine.googleapis.com \
  run.googleapis.com cloudbuild.googleapis.com \
  cloudtrace.googleapis.com monitoring.googleapis.com \
  logging.googleapis.com secretmanager.googleapis.com \
  iam.googleapis.com iamcredentials.googleapis.com \
  storage.googleapis.com artifactregistry.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project=YOUR_PROJECT_ID
```

> ⚠️ Wait **2 minutes** after enabling `discoveryengine.googleapis.com` before the next step. The API propagates asynchronously and returns 503 if called too soon.

### Step 3 — Create service account

```bash
# Create SA
gcloud iam service-accounts create YOUR_SA_NAME \
  --display-name="Agent Runtime Service Account"

# Grant least-privilege roles
for ROLE in \
  roles/aiplatform.user \
  roles/discoveryengine.editor \
  roles/storage.objectViewer \
  roles/logging.logWriter \
  roles/cloudtrace.agent \
  roles/monitoring.metricWriter \
  roles/secretmanager.secretAccessor \
  roles/run.invoker \
  roles/artifactregistry.reader; do
  gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:YOUR_SA_NAME@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="$ROLE" --condition=None
done
```

### Step 4 — Provision Vertex AI service agents (Workload Identity)

```bash
# Force-provision service agents (created lazily by GCP — do not poll for them)
gcloud beta services identity create \
  --service=aiplatform.googleapis.com --project=YOUR_PROJECT_ID

gcloud beta services identity create \
  --service=discoveryengine.googleapis.com --project=YOUR_PROJECT_ID

# Bind Workload Identity
gcloud iam service-accounts add-iam-policy-binding \
  YOUR_SA_NAME@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --member="serviceAccount:service-PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator"
```

> Get your project number: `gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)'`

### Step 5 — Scaffold the project

```bash
# Install agents-cli and coding skills
uv tool install google-agents-cli --upgrade
uvx google-agents-cli setup

# Scaffold with RAG template
agents-cli scaffold create YOUR_PROJECT_NAME \
  --agent agentic_rag \
  --datastore agent_platform_search \
  --yes

cd YOUR_PROJECT_NAME

# Add Agent Runtime deployment infrastructure
agents-cli scaffold enhance --deployment-target agent_runtime --yes

# Install Python dependencies
agents-cli install

# Verify
uv run python -c "import app.agent; print('Import OK')"
```

### Step 6 — Configure environment

Create a `.env` file (never committed):

```bash
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="YOUR_REGION"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export DATA_STORE_ID="YOUR_DATASTORE_ID"
export DATA_STORE_REGION="global"
export GEMINI_MODEL="gemini-2.5-flash"
export LOGS_BUCKET_NAME="YOUR_LOGS_BUCKET"
export AGENT_ENGINE_ID=""                      # Fill after Step 8 deploy
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa-key.json"  # Local dev only
```

```bash
source .env
```

### Step 7 — Provision Vertex AI Search corpus

```bash
# Provision data store infrastructure (runs Terraform internally)
agents-cli infra datastore \
  --project YOUR_PROJECT_ID \
  --region YOUR_REGION

# Upload source documents to corpus bucket
gsutil -m cp -r sample_data/* gs://YOUR_DOCS_BUCKET/

# Grant Discovery Engine SA access to the bucket
gcloud storage buckets add-iam-policy-binding gs://YOUR_DOCS_BUCKET \
  --member="serviceAccount:service-PROJECT_NUMBER@gcp-sa-discoveryengine.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

# Trigger document indexing (REST API — use project NUMBER not ID)
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "x-goog-user-project: YOUR_PROJECT_ID" \
  -H "Content-Type: application/json" \
  -d '{"gcsSource":{"inputUris":["gs://YOUR_DOCS_BUCKET/folder1/*",
    "gs://YOUR_DOCS_BUCKET/folder2/*"],"dataSchema":"content"},
    "reconciliationMode":"FULL"}' \
  "https://discoveryengine.googleapis.com/v1/projects/PROJECT_NUMBER/locations/global/\
collections/default_collection/dataStores/YOUR_DATASTORE_ID/\
branches/default_branch/documents:import"
```

> ⚠️ Always use the project **NUMBER** (not ID string) in Discovery Engine URLs. Use subfolder paths in `inputUris` — the root wildcard `*` does not recurse into subdirectories.

### Step 8 — Test locally

```bash
agents-cli run "Your test query here"
```

All specialist agents should appear in the output. The synthesizer should return a confidence score.

---

## Evaluation

Write eval cases in `tests/eval/datasets/{project}-dataset.json` with a `prompt` and `reference_response` for each case. Configure metrics in `tests/eval/eval_config.yaml`.

```bash
# Always set MEDIGUIDE_EVAL_MODE (or your project's equivalent) before running
# Without it the eval SDK crashes on SequentialAgent introspection
MEDIGUIDE_EVAL_MODE=true agents-cli eval run \
  --dataset tests/eval/datasets/{project}-dataset.json \
  --config tests/eval/eval_config.yaml \
  --project YOUR_PROJECT_ID \
  --region YOUR_REGION
```

### Recommended metrics

| Metric                     | Type                  | Measures                                        |
| -------------------------- | --------------------- | ----------------------------------------------- |
| `custom_response_quality`  | LLM judge (1–5)       | Accuracy, grounding, structure, safety          |
| `domain_safety_compliance` | LLM judge (0–1)       | Named citations present + disclaimer present    |
| `red_flag_detection`       | LLM judge (0–1)       | Emergency/critical scenarios correctly surfaced |
| `tool_trajectory_score`    | Python function (0–1) | Fraction of specialist agents that fired        |

Compare runs after prompt iterations:

```bash
agents-cli eval compare \
  artifacts/grade_results/results_RUN1.json \
  artifacts/grade_results/results_RUN2.json
```

---

## Deploy to Agent Runtime

```bash
# Upgrade scaffold if CLI version changed since scaffolding
agents-cli scaffold upgrade

# Grant your Gmail permission to use the SA as runtime identity
gcloud iam service-accounts add-iam-policy-binding \
  YOUR_SA_NAME@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --member="user:YOUR_GMAIL@gmail.com" \
  --role="roles/iam.serviceAccountUser"

# CRITICAL: deploy must run as your Gmail, not the SA key file
unset GOOGLE_APPLICATION_CREDENTIALS

# Deploy
agents-cli deploy \
  --project YOUR_PROJECT_ID \
  --region YOUR_REGION \
  --service-account YOUR_SA_NAME@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --update-env-vars "DATA_STORE_ID=YOUR_DATASTORE_ID,\
DATA_STORE_REGION=global,\
GEMINI_MODEL=gemini-2.5-flash,\
LOGS_BUCKET_NAME=YOUR_LOGS_BUCKET,\
GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true,\
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT" \
  --no-wait

# Poll status
agents-cli deploy --status --project YOUR_PROJECT_ID --region YOUR_REGION

# Query the live endpoint
agents-cli run \
  --url "https://YOUR_REGION-aiplatform.googleapis.com/v1/projects/PROJECT_NUMBER/\
locations/YOUR_REGION/reasoningEngines/YOUR_AGENT_RUNTIME_ID" \
  --mode adk \
  "Your test query"
```

> ⚠️ `unset GOOGLE_APPLICATION_CREDENTIALS` is the most common deploy failure. If the SA key file is the active credential, the API call runs as the SA — which cannot deploy itself (400 error).

### Add Memory Bank (optional)

After deploy, add `AGENT_ENGINE_ID=YOUR_AGENT_RUNTIME_ID` to `--update-env-vars` and redeploy. The `VertexAiMemoryBankService` in `agent_runtime_app.py` will activate automatically and persist context across sessions.

---

## Observability

The scaffold pre-wires OpenTelemetry via `app/app_utils/telemetry.py`. After deployment:

- **Cloud Trace** → every query appears as a waterfall of agent spans
- **GCS logs** → GenAI completion metadata at `gs://YOUR_LOGS_BUCKET/completions/` (JSONL, HIPAA-safe `NO_CONTENT` mode)
- **Cloud Monitoring** → custom metrics under `custom.googleapis.com/{project}/`

Add domain-specific metrics in `app/app_utils/observability.py` using the OTEL meter and tracer — they export automatically via Agent Runtime's injected exporters.

```bash
# View traces
# console.cloud.google.com → Trace → Trace list → Filter: service.namespace=YOUR_PROJECT

# View custom metrics
# console.cloud.google.com → Monitoring → Metrics Explorer → custom.googleapis.com/
```

---

## Common errors and fixes

| Error                                                | Root cause                                                  | Fix                                                                            |
| ---------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `503` on data connector creation                     | `discoveryengine` API propagation lag                       | `sleep 120` after `gcloud services enable`                                     |
| `403` ADC quota project                              | Discovery Engine bills Google's internal project by default | `gcloud auth application-default set-quota-project YOUR_PROJECT_ID`            |
| `INVALID_ARGUMENT` — SA does not exist               | Vertex AI service agent is created lazily                   | `gcloud beta services identity create --service=aiplatform.googleapis.com`     |
| `404` on document import                             | Project ID used instead of project number in URL            | Use `PROJECT_NUMBER` (from `gcloud projects describe`) not `PROJECT_ID` string |
| Only N/total docs indexed                            | Root wildcard `*` does not recurse subdirectories           | Specify each subfolder explicitly: `gs://bucket/folder/*`                      |
| `SequentialAgent has no .tools` (eval)               | Eval SDK cannot introspect container agents                 | Export a leaf `Agent` when `YOUR_EVAL_MODE_ENV=true`                           |
| `name 'Optional' is not defined` (eval, Python 3.13) | Stricter `get_type_hints()` in Python 3.13                  | Wrap `VertexAiSearchTool` in a plain Python function with concrete annotations |
| `400` cannot act as service account (deploy)         | SA key is active ADC credential — SA cannot deploy itself   | `unset GOOGLE_APPLICATION_CREDENTIALS` then redeploy                           |
| Version mismatch warning on deploy                   | Scaffold version ≠ CLI version                              | `agents-cli scaffold upgrade`                                                  |

---

## What is and is not committed

### Committed ✓

- All source code (`app/`)
- Terraform infrastructure definitions (`deployment/terraform/**/*.tf`)
- Terraform lock file (`deployment/terraform/**/.terraform.lock.hcl`) — pins provider versions
- Terraform variable values (`deployment/terraform/**/vars/env.tfvars`) — project name, region, no secrets
- Eval datasets and config (`tests/eval/`)
- Sample corpus documents (`sample_data/`)
- Monitoring alert definitions (`deployment/monitoring_alerts.json`)

### Never committed ✗

| File/pattern                      | Why excluded                                                     |
| --------------------------------- | ---------------------------------------------------------------- |
| `.env`, `.env.*`                  | Contains credentials and resource IDs                            |
| `*-key.json`, `*-sa-key.json`     | GCP service account key files                                    |
| `*.tfstate`, `*.tfstate.backup`   | Terraform state — contains internal GCP resource metadata        |
| `deployment_metadata.json`        | Agent Runtime ID and project number — document in README instead |
| `artifacts/`                      | Auto-generated eval traces and grade results                     |
| `.venv/`                          | Python virtual environment                                       |
| `app/app_utils/.requirements.txt` | Auto-generated by `agents-cli deploy`                            |

In production, Agent Runtime uses **Workload Identity** — no key files are ever deployed or needed.

---

## Adapting this template for your project

1. Replace all `YOUR_*` placeholders with your actual values
2. Replace `{specialist_agent_*}` in the architecture diagram with your agent names
3. Update `sample_data/` with your domain documents
4. Rewrite the system prompts in `app/agent.py` for your domain
5. Update `tests/eval/datasets/` with domain-appropriate eval cases
6. Add domain-specific custom metrics to `app/app_utils/observability.py`
7. Update `deployment/terraform/single-project/vars/env.tfvars` with your project values

---

## Disclaimer

This system is for decision support only. Always verify outputs with qualified professionals and current authoritative sources. The agent's responses are grounded in the indexed corpus — corpus quality directly determines response quality.
