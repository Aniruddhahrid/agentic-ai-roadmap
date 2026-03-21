# Week 1 Complete — Python Fundamentals for Agent Codebases

## Completed: March 2026

## What I Can Now Do
- Set up a clean Python environment with pyenv and venv from scratch
- Write typed, validated, production-style Python code
- Build and stack decorators for real use cases (timer, retry, @tool preview)
- Write async functions and run concurrent tasks with asyncio.gather()
- Make real LLM API calls with system prompts and multi-turn chat history
- Parse and validate LLM responses into structured Pydantic models
- Read and write JSON, manage files with pathlib
- Understand and navigate professional Python codebases

## Files Built
- `test_setup.py` — environment verification
- `type_hints_practice.py` — basic types, containers, Optional
- `pydantic_practice.py` — BaseModel, Field, validation, nested models
- `decorators_practice.py` — wrapper pattern, timer, retry, stacking, @tool
- `async_practice.py` — asyncio, gather, httpx, concurrent tool calls
- `first_llm_call.py` — Gemini API, system prompts, multi-turn chat
- `json_practice.py` — JSON I/O, pathlib, file analysis
- `week1_project.py` — research agent combining all week 1 concepts

## Stack
Python 3.11 · pyenv · venv · Pydantic v2 · google-genai · httpx · python-dotenv

## Key Mental Models Locked In
- venv = isolated apartment per project, activate before every session
- Type hints = contract between functions, read by VS Code and Pydantic
- Pydantic = type hints with teeth — validates at runtime not just statically
- Decorators = wrappers that run at call time not definition time
- Async = good chef — starts all tasks, checks back when each needs attention
- LLM = just a function: text in, text out, Pydantic structures the output

## Next: Week 2 — Git & GitHub