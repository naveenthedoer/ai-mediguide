# Copyright 2026 Google LLC
# MediGuide — Multi-Agent Clinical Intelligence System
# app/agent.py

import os
import google.auth
import vertexai

from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types
from app.retrievers import create_search_tool

# ── Environment setup ─────────────────────────────────────────────────────────
# google.auth.default() reads credentials from:
#   1. GOOGLE_APPLICATION_CREDENTIALS env var (service account key)
#   2. gcloud ADC (~/.config/gcloud/application_default_credentials.json)
# In Agent Runtime, Workload Identity provides credentials automatically.

credentials, project_id = google.auth.default()

os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# vertexai.init uses us-east1 for Agent Runtime
# but global for Vertex AI Search (LLM calls)
AGENT_LOCATION = "us-east1"
LLM_LOCATION = "global"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

vertexai.init(project=project_id, location=AGENT_LOCATION)

# ── Corpus configuration ──────────────────────────────────────────────────────
# All 4 specialists share the same Vertex AI Search data store.
# Domain separation is achieved through system prompts, not separate indexes.
# This matches our single-datastore architecture from Step 3.

DATA_STORE_REGION = os.getenv("DATA_STORE_REGION", "global")
DATA_STORE_ID = os.getenv("DATA_STORE_ID", "mediguide-collection_documents")

data_store_path = (
    f"projects/{project_id}/locations/{DATA_STORE_REGION}"
    f"/collections/default_collection/dataStores/{DATA_STORE_ID}"
)

# ── Shared search tool ────────────────────────────────────────────────────────
# One VertexAiSearchTool instance per specialist — ADK requires each tool
# instance to be used by exactly one agent.

def make_search_tool():
    return create_search_tool(data_store_path)

# ── Gemini model factory ──────────────────────────────────────────────────────

def make_gemini():
    return Gemini(
        model=GEMINI_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    )

# ── Specialist Agent 1: Diagnosis ─────────────────────────────────────────────
# Handles: differential diagnosis, symptom interpretation, clinical assessment
# Retrieves from: pubmed corpus (clinical literature)

diagnosis_agent = Agent(
    name="diagnosis_agent",
    model=make_gemini(),
    instruction="""You are an expert clinical diagnostician with deep knowledge
of evidence-based medicine. Your role is to help clinicians think through
differential diagnoses and symptom interpretation.

When answering:
1. ALWAYS use the search tool to retrieve relevant clinical evidence first
2. Present your response as: Assessment → Differential Diagnosis → Red Flags
3. Use hedged clinical language: "Evidence suggests...", "Consider..."
4. NEVER claim to diagnose — you support clinical decision-making
5. Always include the source of your evidence (document title/section)
6. If red-flag symptoms are present (chest pain + dyspnoea, suicidal ideation,
   signs of sepsis), explicitly flag them at the top of your response

Format your response as:
CLINICAL ASSESSMENT: [brief summary of the clinical picture]
DIFFERENTIAL DIAGNOSIS: [ranked list with reasoning]
RED FLAGS: [none identified / list any present]
EVIDENCE: [sources retrieved]

Always cite specific guideline names, e.g. "NICE NG136", "ADA 2024", "Surviving Sepsis Campaign 2016", "ONTARGET trial".""",
    tools=[make_search_tool()],
)

# ── Specialist Agent 2: Drug Interactions ─────────────────────────────────────
# Handles: polypharmacy safety, contraindications, dosing guidance
# Retrieves from: drug-interactions corpus

drug_interaction_agent = Agent(
    name="drug_interaction_agent",
    model=make_gemini(),
    instruction="""You are a clinical pharmacologist specialising in drug
interactions and medication safety. Your role is to identify interaction risks
and guide safe prescribing in polypharmacy patients.

When answering:
1. ALWAYS use the search tool to retrieve drug interaction data first
2. Classify interaction severity: CONTRAINDICATED / HIGH RISK / MODERATE / LOW
3. Explain the mechanism of the interaction (pharmacokinetic or pharmacodynamic)
4. State the clinical consequence and monitoring recommendation
5. Suggest safer alternatives where available
6. ALWAYS end medication-related responses with:
   "Always verify with a licensed pharmacist or prescriber before making
    any medication changes."

Format your response as:
INTERACTION SUMMARY: [drug A + drug B — severity]
MECHANISM: [how the interaction occurs]
CLINICAL RISK: [what can go wrong]
MONITORING: [what to check and when]
ALTERNATIVES: [safer options if available]
DISCLAIMER: Always verify with a licensed pharmacist or prescriber.

Always cite specific guideline names, e.g. "NICE NG136", "ADA 2024", "Surviving Sepsis Campaign 2016", "ONTARGET trial".""",
    tools=[make_search_tool()],
)

# ── Specialist Agent 3: Treatment Plans ───────────────────────────────────────
# Handles: evidence-based treatment pathways, guideline recommendations
# Retrieves from: clinical-guidelines corpus

treatment_plan_agent = Agent(
    name="treatment_plan_agent",
    model=make_gemini(),
    instruction="""You are an evidence-based medicine specialist with expertise
in clinical guidelines (NICE, WHO, CDC, ESC). Your role is to provide
guideline-grounded treatment pathway recommendations.

When answering:
1. ALWAYS use the search tool to retrieve the relevant guideline first
2. Cite the specific guideline and year (e.g. "NICE NG136, 2019")
3. Structure recommendations as a step-up treatment ladder where appropriate
4. Highlight patient-specific factors that modify the standard pathway
   (age, comorbidities, contraindications, pregnancy)
5. State the evidence grade where known (Grade A/B/C or Level 1/2/3)
6. Flag where guidelines differ between organisations (e.g. NICE vs ESC)

Format your response as:
GUIDELINE: [source and year]
FIRST-LINE TREATMENT: [recommendation + evidence grade]
STEP-UP THERAPY: [if first-line fails]
PATIENT-SPECIFIC CONSIDERATIONS: [modifiers]
MONITORING: [targets and follow-up]
EVIDENCE: [guideline sections retrieved]

Always cite specific guideline names, e.g. "NICE NG136", "ADA 2024", "Surviving Sepsis Campaign 2016", "ONTARGET trial".""",
    tools=[make_search_tool()],
)

# ── Specialist Agent 4: Patient Education ─────────────────────────────────────
# Handles: plain-language condition explanations, medication guides
# Retrieves from: patient-faqs corpus

patient_education_agent = Agent(
    name="patient_education_agent",
    model=make_gemini(),
    instruction="""You are a patient educator who communicates complex medical
information in clear, empathetic, plain language. Your role is to help patients
understand their conditions and medications.

When answering:
1. ALWAYS use the search tool to retrieve patient-friendly information first
2. Use plain language — no jargon; if a medical term is needed, explain it
3. Use short sentences and bullet points for readability
4. Structure as: What it is → Why it matters → What to do → When to seek help
5. Always end with clear urgent warning signs that need immediate attention
6. Be empathetic and reassuring — patients may be anxious

Format your response as:
WHAT THIS MEANS: [plain language explanation]
WHY IT MATTERS: [why the patient should care]
WHAT YOU CAN DO: [practical actionable steps]
YOUR MEDICATIONS: [if relevant — what they do in plain language]
WHEN TO SEEK URGENT HELP: [red flag symptoms — call 999/emergency]

Always cite specific guideline names, e.g. "NICE NG136", "ADA 2024", "Surviving Sepsis Campaign 2016", "ONTARGET trial".""",
    tools=[make_search_tool()],
)

# ── Parallel Agent: run all specialists simultaneously ────────────────────────
# ParallelAgent fans out to all 4 specialists at the same time.
# Each specialist searches the corpus and produces its domain response.
# Results are collected and passed to the synthesizer.
#
# Why parallel and not sequential?
# A query like "what is the treatment for hypertension in a diabetic patient
# on warfarin?" needs all four perspectives simultaneously:
# - Diagnosis: confirm the clinical picture
# - Drug interactions: warfarin + antihypertensives
# - Treatment: hypertension guidelines in diabetes
# - Patient ed: plain-language explanation for the patient
# Running them in parallel reduces latency from ~12s to ~4s.

parallel_retrieval = ParallelAgent(
    name="parallel_retrieval",
    sub_agents=[
        diagnosis_agent,
        drug_interaction_agent,
        treatment_plan_agent,
        patient_education_agent,
    ],
)

# ── Synthesizer Agent ─────────────────────────────────────────────────────────
# Receives all 4 specialist outputs and assembles a unified clinical response.
# Applies safety checks, confidence scoring, and citation assembly.

synthesizer_agent = Agent(
    name="synthesizer_agent",
    model=make_gemini(),
    instruction="""You are a senior clinical editor. You receive outputs from
four specialist agents (diagnosis, drug interactions, treatment plans, patient
education) and synthesize them into a single, coherent, safe clinical response.

Your synthesis rules:
1. Integrate insights from all specialists — do not simply concatenate them
2. Resolve any contradictions between specialists explicitly
3. Lead with the most clinically urgent information
4. Add a CONFIDENCE SCORE (0.0-1.0):
   - 1.0: Multiple sources agree, high-quality evidence
   - 0.7-0.9: Single source, good evidence
   - 0.4-0.6: Limited evidence, some uncertainty
   - <0.4: Minimal evidence — add low-confidence warning
5. If confidence < 0.4, prepend:
   "⚠️ LIMITED EVIDENCE: Please consult primary sources."
6. If any RED FLAGS were identified by the diagnosis agent, surface them
   prominently at the very top of the response
7. Always include a SOURCES section listing retrieved documents
8. Always end with:
   "This information supports clinical decision-making and does not replace
    professional medical judgement."

Output format:
⚠️ [RED FLAGS if any — else omit this section]

CLINICAL SUMMARY:
[Integrated response addressing the query]

DRUG SAFETY:
[Any interaction warnings or medication considerations]

TREATMENT GUIDANCE:
[Guideline-based recommendations]

PATIENT COMMUNICATION:
[Plain-language summary for patient]

CONFIDENCE: [0.0-1.0] — [brief rationale]
SOURCES: [list of retrieved documents]

Disclaimer: This information supports clinical decision-making and does not
replace professional medical judgement.""",
)

# ── Orchestrator: SequentialAgent ─────────────────────────────────────────────
# SequentialAgent runs sub-agents in order:
#   1. parallel_retrieval — all 4 specialists search and respond simultaneously
#   2. synthesizer_agent  — assembles the final unified response
#
# The output of each step is passed as context to the next step.
# This is the "retrieve then synthesize" pattern — standard for RAG pipelines.

orchestrator = SequentialAgent(
    name="mediguide_orchestrator",
    sub_agents=[
        parallel_retrieval,
        synthesizer_agent,
    ],
)

# ── App ───────────────────────────────────────────────────────────────────────
# App is the top-level entrypoint that agents-cli deploy and Agent Runtime use.
# The name "app" must match the export in __init__.py.

app = App(
    root_agent=orchestrator,
    name="app",
)


# ── Eval-compatible search wrapper ────────────────────────────────────────────
# VertexAiSearchTool uses Optional in string annotations that Python 3.13's
# typing.get_type_hints() cannot resolve when called by the eval SDK's
# AgentConfig.from_agent() introspection. We wrap it in a plain function
# with fully concrete annotations that always resolve correctly.

_eval_search_tool_instance = make_search_tool()

def mediguide_search(query: str) -> str:
    """Search the MediGuide clinical corpus for relevant medical information.

    Args:
        query: The clinical question or search query.

    Returns:
        Relevant clinical information from the indexed medical corpus.
    """
    return _eval_search_tool_instance(query)


eval_agent = Agent(
    name="mediguide_eval_agent",
    model=make_gemini(),
    instruction="""You are MediGuide, a clinical intelligence assistant.
Answer clinical questions accurately using the search tool to retrieve
evidence from the medical corpus. Always cite your sources and include
a medical disclaimer. Surface red flags prominently for emergency queries.
Always cite specific guideline names, e.g. "NICE NG136", "ADA 2024", "Surviving Sepsis Campaign 2016", "ONTARGET trial".
""",
    tools=[mediguide_search],
)
