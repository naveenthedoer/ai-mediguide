# MediGuide — Production Multi-Agent Clinical Intelligence System

A production-grade healthcare RAG platform on GCP. Routes clinical queries through 4 specialist AI agents grounded in a corpus of medical literature, drug interactions, clinical guidelines, and patient education — then synthesizes a unified, cited, safety-checked response.

Built with `google-adk`, `agents-cli v0.3.1`, Vertex AI Search, and Vertex AI Agent Runtime.

---

## Architecture

```
User query
    ↓
mediguide_orchestrator (SequentialAgent)
    ↓
parallel_retrieval (ParallelAgent) — all 4 fire simultaneously
    ├── diagnosis_agent          → VertexAiSearchTool → corpus
    ├── drug_interaction_agent   → VertexAiSearchTool → corpus
    ├── treatment_plan_agent     → VertexAiSearchTool → corpus
    └── patient_education_agent  → VertexAiSearchTool → corpus
    ↓
synthesizer_agent — citations + confidence score + red flags
    ↓
Unified clinical response
```

---

## Stack

| Component       | Technology                                                   |
| --------------- | ------------------------------------------------------------ |
| Agent framework | `google-adk ≥1.15`, `agents-cli v0.3.1`                      |
| Model           | `gemini-2.5-flash`                                           |
| RAG retrieval   | Vertex AI Search (`discoveryengine.googleapis.com`)          |
| Deployment      | Vertex AI Agent Runtime                                      |
| Memory          | `VertexAiMemoryBankService` — cross-session clinical context |
| Observability   | Cloud Trace, Cloud Monitoring, GCS telemetry logs            |
| Infrastructure  | Terraform (`deployment/terraform/single-project/`)           |
| Region          | `us-east1` (agents), `global` (Vertex AI Search)             |

---

## Project Structure

```
mediguide/
├── app/
│   ├── agent.py                  # 4 specialists + ParallelAgent + synthesizer + SequentialAgent
│   ├── agent_runtime_app.py      # AgentEngineApp + telemetry + Memory Bank
│   ├── retrievers.py             # VertexAiSearchTool factory
│   └── app_utils/
│       ├── telemetry.py          # OpenTelemetry + GCS log upload
│       ├── observability.py      # Custom metrics + trace helpers
│       └── typing.py             # Feedback model
├── tests/
│   └── eval/
│       ├── eval_config.yaml      # 4 custom metrics (LLM judge + Python)
│       └── datasets/
│           └── mediguide-clinical-dataset.json  # 8 clinical eval cases
├── deployment/
│   ├── terraform/single-project/ # Vertex AI Search infrastructure
│   └── monitoring_alerts.json    # 3 Cloud Monitoring alerting policies
├── sample_data/                  # 6 clinical documents (corpus source)
├── agents-cli-manifest.yaml
└── pyproject.toml
```

---

## Production Deployment

**Agent Runtime ID:** `8864385888432422912`  
**Project:** `adk-learning-494613` (project number: `1082951374997`)  
**Region:** `us-east1`

```bash
# Query the live production endpoint
agents-cli run \
  --url "https://us-east1-aiplatform.googleapis.com/v1/projects/1082951374997/locations/us-east1/reasoningEngines/8864385888432422912" \
  --mode adk \
  "What is the first-line treatment for hypertension in a diabetic patient with CKD?"
```

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- GCP project with billing enabled

### 1. Clone and install

```bash
git clone <repo-url>
cd mediguide
agents-cli install
```

### 2. Authenticate

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth application-default set-quota-project adk-learning-494613
```

### 3. Set environment variables

Create a `.env` file (never committed — see `.gitignore`):

```bash
export GOOGLE_CLOUD_PROJECT="adk-learning-494613"
export GOOGLE_CLOUD_LOCATION="us-east1"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export DATA_STORE_ID="mediguide-collection_documents"
export DATA_STORE_REGION="global"
export GEMINI_MODEL="gemini-2.5-flash"
export LOGS_BUCKET_NAME="adk-learning-494613-mediguide-logs"
export AGENT_ENGINE_ID="8864385888432422912"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/mediguide-sa-key.json"
```

Then source it:

```bash
source .env
```

### 4. Run locally

```bash
agents-cli run "What is the interaction between warfarin and amiodarone?"
```

---

## Evaluation

```bash
MEDIGUIDE_EVAL_MODE=true agents-cli eval run \
  --dataset tests/eval/datasets/mediguide-clinical-dataset.json \
  --config tests/eval/eval_config.yaml \
  --project adk-learning-494613 \
  --region us-east1
```

### Eval results (latest run)

| Metric                       | Score          | Notes                                          |
| ---------------------------- | -------------- | ---------------------------------------------- |
| `red_flag_detection`         | **1.0 / 1.0**  | Perfect — every emergency correctly identified |
| `custom_response_quality`    | **3.5 / 5.0**  | Good — citation specificity gap identified     |
| `clinical_safety_compliance` | **0.58 / 1.0** | Fix: add named guideline citations to prompts  |
| `tool_trajectory_score`      | **0.0**        | Expected in eval mode — meaningful post-deploy |

---

## Deploying

```bash
# Critical: must deploy as personal Gmail ADC, not SA key file
unset GOOGLE_APPLICATION_CREDENTIALS

agents-cli deploy \
  --project adk-learning-494613 \
  --region us-east1 \
  --service-account mediguide-agent-sa@adk-learning-494613.iam.gserviceaccount.com \
  --update-env-vars "DATA_STORE_ID=mediguide-collection_documents,DATA_STORE_REGION=global,GEMINI_MODEL=gemini-2.5-flash,LOGS_BUCKET_NAME=adk-learning-494613-mediguide-logs,GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true,OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT,AGENT_ENGINE_ID=8864385888432422912" \
  --no-wait

# Check status
agents-cli deploy --status --project adk-learning-494613 --region us-east1
```

---

## Corpus — 6 Indexed Documents

| File                            | Folder                 | Contents                                       |
| ------------------------------- | ---------------------- | ---------------------------------------------- |
| `hypertension-review.txt`       | `pubmed/`              | ACEi, ARBs, CCBs, ONTARGET trial, BP targets   |
| `diabetes-management.txt`       | `pubmed/`              | Metformin, GLP-1, SGLT2, ADA 2024              |
| `warfarin-interactions.txt`     | `drug-interactions/`   | CYP2C9 inhibitors, INR monitoring, amiodarone  |
| `common-drug-interactions.txt`  | `drug-interactions/`   | Statins, SSRIs, ACEi+ARB danger                |
| `chest-pain-assessment.txt`     | `clinical-guidelines/` | HEART score, STEMI, NICE NG185, Hour-1 bundle  |
| `understanding-medications.txt` | `patient-faqs/`        | Ramipril, metformin, amlodipine patient guides |

---

## Key Design Decisions

**Why one shared datastore, not four separate ones?**  
Domain separation is achieved through system prompt instructions, not separate indexes. Each agent's instruction tells it exactly which domain to focus on — so the drug interaction agent retrieves drug content even though all documents share one store.

**Why SequentialAgent wrapping ParallelAgent?**  
The retrieve phase (parallel across 4 agents) must complete entirely before synthesis begins. SequentialAgent enforces this ordering. ParallelAgent handles the fan-out within the retrieve phase. This is the retrieve-then-synthesize RAG pattern.

**Why `MEDIGUIDE_EVAL_MODE=true` for evals?**  
The Vertex AI eval SDK calls `AgentConfig.from_agent(agent).tools` — SequentialAgent has no `.tools` attribute. Eval mode exports a leaf `Agent` that the SDK can introspect. The full orchestrator runs normally in all other contexts.

**Why `unset GOOGLE_APPLICATION_CREDENTIALS` before deploy?**  
If the SA key file is the active ADC credential, the deploy API call runs as the SA — which cannot deploy itself (400 error). Deploy must run as your personal Gmail account which has owner role.

**Why use project NUMBER not ID in Discovery Engine URLs?**  
The Discovery Engine API resource names are keyed on project number (`1082951374997`), not project ID string. Using the ID returns 404 on import and list operations.

---

## Errors Fixed During Build

| Error                                   | Root cause                                           | Fix                                                 |
| --------------------------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| `INVALID_ARGUMENT` — SA does not exist  | Vertex AI service agent is created lazily            | `gcloud beta services identity create`              |
| `503` on data connector                 | API propagation lag after enablement                 | `sleep 120` after enabling discoveryengine API      |
| Wrong Terraform path                    | Expected `dev/` — actual is `single-project/`        | Read scaffold output before assuming paths          |
| `403` ADC quota project                 | Discovery Engine billing defaults to Google internal | `gcloud auth application-default set-quota-project` |
| Only 1 doc indexed instead of 6         | Root wildcard `*` skips subdirectories               | Specify each subfolder explicitly in `inputUris`    |
| `SequentialAgent has no .tools` (eval)  | Eval SDK cannot introspect container agents          | `MEDIGUIDE_EVAL_MODE=true` + leaf `eval_agent`      |
| `name 'Optional' is not defined` (eval) | Python 3.13 stricter `get_type_hints()`              | Wrap `VertexAiSearchTool` in plain Python function  |
| `400` — cannot act as SA (deploy)       | SA key was active ADC credential                     | `unset GOOGLE_APPLICATION_CREDENTIALS`              |

---

## GCP Resources

| Resource          | Name / ID                                                        |
| ----------------- | ---------------------------------------------------------------- |
| Project           | `adk-learning-494613`                                            |
| Project number    | `1082951374997`                                                  |
| Service account   | `mediguide-agent-sa@adk-learning-494613.iam.gserviceaccount.com` |
| Corpus bucket     | `adk-learning-494613-mediguide-docs`                             |
| Logs bucket       | `adk-learning-494613-mediguide-logs`                             |
| Artifact Registry | `mediguide-images` (us-east1)                                    |
| Data store        | `mediguide-collection_documents`                                 |
| Search engine     | `mediguide-search`                                               |
| Agent Runtime ID  | `8864385888432422912`                                            |

---

## Security Notes

The following are excluded from version control via `.gitignore` and must **never** be committed:

- `.env` — environment variables including credentials
- `*-key.json`, `*-sa-key.json` — GCP service account key files
- `deployment/terraform/**/*.tfstate` — Terraform state (contains resource metadata)
- `deployment_metadata.json` — Agent Runtime ID and project number
- `artifacts/` — eval traces and grade results
- `.venv/` — Python virtual environment

In production, Agent Runtime uses **Workload Identity** — no key files are deployed or needed.

---

## Disclaimer

This system is for clinical decision support only and does not replace professional medical judgement. Always verify clinical information with qualified healthcare professionals and current guidelines.
