"""Pathoconnect Copilot - the CRM chat assistant.

Served by copilot_app.py as its own gunicorn process, separate from app.py,
so a slow LLM reply never holds up the route/delay/no-show endpoints.
"""
