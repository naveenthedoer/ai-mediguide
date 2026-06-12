# Copyright 2026 Google LLC
import os
from .agent import app, eval_agent
from google.adk.apps import App

if os.getenv("MEDIGUIDE_EVAL_MODE") == "true":
    # Eval harness needs a leaf Agent — SequentialAgent has no .tools attribute
    app = App(root_agent=eval_agent, name="app")

__all__ = ["app"]