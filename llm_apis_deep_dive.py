# llm_apis_deep_dive.py
# LLM API parameters — understanding what each one does to output.
# Using OpenAI SDK via Ollama (local, no quota limits)

import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen2.5:7b"


# ============================================================
# SECTION 1: Temperature — randomness control
# ============================================================

def call_with_temperature(prompt: str, temperature: float) -> str:
    """Make a call with a specific temperature and return the text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


print("=" * 50)
print("SECTION 1: Temperature comparison")
print("=" * 50)

prompt = "Describe an AI agent in one sentence."

for temp in [0.0, 0.5, 1.0]:
    result = call_with_temperature(prompt, temp)
    print(f"\nTemperature {temp}:")
    print(f"  {result}")

print("\n--- Determinism test (temperature=0.0, called twice) ---")
result1 = call_with_temperature(prompt, 0.0)
result2 = call_with_temperature(prompt, 0.0)
print(f"Call 1: {result1}")
print(f"Call 2: {result2}")
print(f"Identical: {result1 == result2}")


# ============================================================
# SECTION 2: Tokens — understanding cost and length
# ============================================================

print("\n" + "=" * 50)
print("SECTION 2: Token counting")
print("=" * 50)

def analyze_tokens(prompt: str, label: str) -> None:
    """Call the API and show token usage."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500
    )

    # OpenAI SDK token usage lives in response.usage
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    print(f"\n{label}")
    print(f"  Input tokens:  {input_tokens}")
    print(f"  Output tokens: {output_tokens}")
    print(f"  Total tokens:  {total_tokens}")
    print(f"  Response preview: {response.choices[0].message.content[:100]}...")


analyze_tokens("What is Python?", "Short prompt")

analyze_tokens(
    """You are a senior Python developer. A junior developer has asked you
    to explain what Python is, its main use cases, its advantages over
    other languages, and why it's particularly popular for AI development.
    Please provide a comprehensive answer.""",
    "Long prompt with context"
)

analyze_tokens(
    "List 10 Python frameworks with one sentence each.",
    "Request for long output"
)


# ============================================================
# SECTION 3: max_tokens — truncation behaviour
# ============================================================

print("\n" + "=" * 50)
print("SECTION 3: max_tokens truncation")
print("=" * 50)

def call_with_max_tokens(prompt: str, max_tokens: int) -> tuple[str, str]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens
    )
    # finish_reason: "stop" = natural end, "length" = hit max_tokens
    finish_reason = response.choices[0].finish_reason
    return response.choices[0].message.content.strip(), finish_reason


prompt = "Write a detailed explanation of how neural networks work."

for max_t in [20, 100, 500]:
    text, reason = call_with_max_tokens(prompt, max_t)
    print(f"\nmax_tokens={max_t} | finish_reason={reason}")
    print(f"  Output: {text[:150]}")


# ============================================================
# SECTION 4: Model comparison
# ============================================================

print("\n" + "=" * 50)
print("SECTION 4: Model comparison")
print("=" * 50)

def timed_call(model: str, prompt: str) -> tuple[str, float, int]:
    """Call a model and return text, latency, and total tokens."""
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200
    )
    latency = time.time() - start
    tokens = response.usage.total_tokens
    return response.choices[0].message.content.strip(), latency, tokens


prompt = "Explain what an MCP server is in 3 sentences."

# Only testing models you have downloaded locally
models_to_test = ["qwen2.5:7b"]

for model in models_to_test:
    try:
        text, latency, tokens = timed_call(model, prompt)
        print(f"\nModel: {model}")
        print(f"  Latency: {latency:.2f}s")
        print(f"  Tokens:  {tokens}")
        print(f"  Output:  {text[:150]}")
    except Exception as e:
        print(f"\nModel: {model} — Error: {e}")


# ============================================================
# SECTION 5: Stop sequences
# ============================================================

print("\n" + "=" * 50)
print("SECTION 5: Stop sequences")
print("=" * 50)

# Without stop sequence
response_without = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": 'Return a JSON object with keys "name" and "score". Nothing else.'}],
    temperature=0.0,
    max_tokens=200
)
print("Without stop sequence:")
print(response_without.choices[0].message.content)

# With markdown stripping — cleaner than stop sequences for JSON
response_with = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": 'Return a JSON object with keys "name" and "score". Nothing else.'}],
    temperature=0.0,
    max_tokens=200
)
text = response_with.choices[0].message.content.strip()
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
# MAX_TOKENS:
#   50-100    → single sentence answers, classifications
#   200-500   → paragraphs, short explanations
#   1000-2000 → detailed analysis, structured documents
#   Rule: set ~20% higher than you expect to need
#
# FINISH_REASON:
#   "stop"   → model finished naturally
#   "length" → hit max_tokens limit, response was cut off
#
# STOP SEQUENCES:
#   Use sparingly — markdown stripping is more reliable for JSON
#
# MODEL SELECTION:
#   Prototype with best available, optimise down for production