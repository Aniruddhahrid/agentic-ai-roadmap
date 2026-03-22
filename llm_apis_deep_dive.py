# llm_apis_deep_dive.py
# LLM API parameters — understanding what each one does to output.

import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash-lite"


# # ============================================================
# # SECTION 1: Temperature — randomness control
# # ============================================================

# def call_with_temperature(prompt: str, temperature: float) -> str:
#     """Make a call with a specific temperature and return the text."""
#     response = client.models.generate_content(
#         model=MODEL,
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             temperature=temperature,
#             max_output_tokens=1000,  # keep responses short for comparison
#         )
#     )
#     time.sleep(20)
#     return response.text.strip()


# print("=" * 50)
# print("SECTION 1: Temperature comparison")
# print("=" * 50)

# prompt = "Describe an AI agent in one sentence."

# # Call the same prompt three times at different temperatures
# for temp in [0.0, 0.5, 1.0]:
#     result = call_with_temperature(prompt, temp)
#     print(f"\nTemperature {temp}:")
#     print(f"  {result}")

# # Call temperature 0.0 twice — should be identical or near-identical
# print("\n--- Determinism test (temperature=0.0, called twice) ---")
# result1 = call_with_temperature(prompt, 0.0)
# result2 = call_with_temperature(prompt, 0.0)
# print(f"Call 1: {result1}")
# print(f"Call 2: {result2}")
# print(f"Identical: {result1 == result2}")

# ============================================================
# SECTION 2: Tokens — understanding cost and length
# ============================================================

# print("\n" + "=" * 50)
# print("SECTION 2: Token counting")
# print("=" * 50)

# def analyze_tokens(prompt: str, label: str) -> None:
#     """Call the API and show detailed token usage."""
#     response = client.models.generate_content(
#         model=MODEL,
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             temperature=0.0,
#             max_output_tokens=5000,
#         )
#     )

#     input_tokens = response.usage_metadata.prompt_token_count
#     output_tokens = response.usage_metadata.candidates_token_count
#     total_tokens = response.usage_metadata.total_token_count

#     print(f"\n{label}")
#     print(f"  Input tokens:  {input_tokens}")
#     print(f"  Output tokens: {output_tokens}")
#     print(f"  Total tokens:  {total_tokens}")
#     print(f"  Response preview: {response.text}")

#     time.sleep(15)


# # Compare token usage across different prompt lengths
# analyze_tokens(
#     "What is Python?",
#     "Short prompt"
# )

# analyze_tokens(
#     """You are a senior Python developer. A junior developer has asked you
#     to explain what Python is, its main use cases, its advantages over
#     other languages, and why it's particularly popular for AI development.
#     Please provide a comprehensive answer.""",
#     "Long prompt with context"
# )

# analyze_tokens(
#     "List 10 Python frameworks with one sentence each.",
#     "Request for long output"
# )

# ============================================================
# SECTION 3: max_output_tokens — truncation behaviour
# ============================================================

# print("\n" + "=" * 50)
# print("SECTION 3: max_output_tokens truncation")
# print("=" * 50)

# def call_with_max_tokens(prompt: str, max_tokens: int) -> str:
#     response = client.models.generate_content(
#         model=MODEL,
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             temperature=0.0,
#             max_output_tokens=max_tokens,
#         )
#     )
#     # finish_reason tells you WHY generation stopped
#     # "STOP" = natural end, "MAX_TOKENS" = hit the limit
#     finish_reason = response.candidates[0].finish_reason
#     return response.text.strip(), finish_reason


# prompt = "Write a detailed explanation of how neural networks work."

# for max_t in [20, 100, 500, 1000, 2000]:
#     text, reason = call_with_max_tokens(prompt, max_t)
#     print(f"\nmax_output_tokens={max_t} | finish_reason={reason}")
#     print(f"  Output: {text}")

# print("List of models that support generateContent:\n")
# for m in client.models.list():
#     for action in m.supported_actions:
#         if action == "generateContent":
#             print(m.name)

# model_info = client.models.get(model="gemini-2.0-flash")
# print(model_info)

# ============================================================
# SECTION 4: Model selection — speed vs quality trade-off
# ============================================================

# ============================================================
# SECTION 5: Stop sequences — controlling where generation ends
# ============================================================

print("\n" + "=" * 50)
print("SECTION 5: Stop sequences")
print("=" * 50)

# Without stop sequence — model might ramble after the JSON
response_without_stop = client.models.generate_content(
    model=MODEL,
    contents='Return a JSON object with keys "name" and "score". Nothing else.',
    config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=200,
    )
)
print("Without stop sequence:")
print(response_without_stop.text)

# Clean approach — strip markdown fences if present
response_with_stop = client.models.generate_content(
    model=MODEL,
    contents='Return a JSON object with keys "name" and "score". Nothing else.',
    config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=200,
    )
)

# Strip markdown fences if model adds them despite being told not to
text = response_with_stop.text.strip()
if text.startswith("```"):
    text = text.split("```")[1]
    if text.startswith("json"):
        text = text[4:]
text = text.strip()

print("\nWith markdown stripping:")
print(text)

# ============================================================
# PARAMETER DECISION GUIDE — reference for every API call
# ============================================================
#
# TEMPERATURE:
#   0.0       → data extraction, classification, structured output
#   0.1-0.3   → factual Q&A, summarisation, code generation
#   0.5-0.7   → balanced tasks, analysis, explanations
#   0.8-1.0   → creative writing, brainstorming, varied responses
#
# MAX_OUTPUT_TOKENS:
#   50-100    → single sentence answers, classifications
#   200-500   → paragraphs, short explanations
#   1000-2000 → detailed analysis, structured documents
#   4000+     → long-form content, full code files
#   Rule: set ~20% higher than you expect to need
#
# TOP_P:
#   Leave at default (1.0) unless you know what you're doing
#   Never adjust both temperature AND top_p simultaneously
#
# STOP SEQUENCES:
#   Use when you need clean termination at a specific point
#   Most useful for JSON outputs and structured formats
#
# MODEL SELECTION:
#   gemini-2.5-flash  → best quality, use for complex reasoning
#   gemini-2.0-flash-lite → faster, use for simple/high-volume tasks
#   Rule: prototype with best model, optimise down later

#         models/gemini-2.5-flash
# models/gemini-2.5-pro
# models/gemini-2.0-flash
# x models/gemini-2.0-flash-001 
# models/gemini-2.0-flash-lite-001
# x models/gemini-2.0-flash-lite
#  audio - models/gemini-2.5-flash-preview-tts
# models/gemini-2.5-pro-preview-tts
# models/gemma-3-1b-it
# models/gemma-3-4b-it
# models/gemma-3-12b-it
# models/gemma-3-27b-it
# models/gemma-3n-e4b-it
# models/gemma-3n-e2b-it
# models/gemini-flash-latest
# models/gemini-flash-lite-latest
# models/gemini-pro-latest
# models/gemini-2.5-flash-lite
# models/gemini-2.5-flash-image
# models/gemini-2.5-flash-lite-preview-09-2025
# models/gemini-3-pro-preview
# models/gemini-3-flash-preview
# models/gemini-3.1-pro-preview
# models/gemini-3.1-pro-preview-customtools
# models/gemini-3.1-flash-lite-preview
# models/gemini-3-pro-image-preview
# models/nano-banana-pro-preview
# models/gemini-3.1-flash-image-preview
# models/gemini-robotics-er-1.5-preview
# models/gemini-2.5-computer-use-preview-10-2025
# models/deep-research-pro-preview-12-2025