"""Web frontend + analysis endpoint for trufflepig.

This package serves a small FastAPI app that lets a user upload a
sample TPM file (or salmon quant) through the browser, runs the full
``trufflepig run`` pipeline as a background subprocess, streams
per-stage progress back via server-sent events, and renders the
summary / analysis / brief markdown as HTML once the run completes.

Multi-sample longitudinal comparison is exposed via the same UI by
selecting two or more prior runs.

Architecture is the scaffolded shape of trufflepig#16 — the analysis
runs in-process here, but each run is isolated to a workspace
directory so a serverless deployment can keep the same shape with the
subprocess swapped for a remote-job submission.
"""

from .app import create_app, WebSettings

__all__ = ["create_app", "WebSettings"]
