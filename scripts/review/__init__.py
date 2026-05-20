"""Distributed pseudo-label review pipeline (Phase 1 core).

Backend-independent, dependency-light modules that pin the JSON schemas,
diff/IRR semantics, provenance transitions, and the double-review +
adjudication state machine. The HF-Space review service (Phase 2) and the
reviewtool client (Phase 3) build against this contract.
"""
