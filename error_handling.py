# error_handling.py
# Error handling and retries for LLM API calls.
# Resilient agents handle failures gracefully — they don't crash.

import json
import time
import random
from typing import Optional, TypeVar, Type
from openai import OpenAI
from pydantic import BaseModel, ValidationError

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen2.5:7b"


# ============================================================
# SECTION 1: Exception taxonomy
# ============================================================

print("=" * 50)
print("SECTION 1: Exception taxonomy")
print("=" * 50)

# Not all errors should be treated the same way.
# Some should be retried. Some should fail immediately.
# Retrying the wrong ones wastes time and money.

# RETRY-ABLE errors (transient — might work on next attempt):
#   - Network timeout
#   - Rate limit (429)
#   - Server error (500, 502, 503)
#   - Model temporarily unavailable

# FAIL-FAST errors (permanent — retrying won't help):
#   - Invalid API key (401)
#   - Model not found (404)
#   - Malformed request (400)
#   - JSON parse error from bad model output
#   - Pydantic validation error

# Let's see each type in action:

def demonstrate_exceptions():
    """Show the different exception types you'll encounter."""

    # 1. JSON parse error — model returned invalid JSON
    print("\n1. JSON parse error:")
    bad_json = "Here is the data: {name: John, age: not_a_number}"
    try:
        json.loads(bad_json)
    except json.JSONDecodeError as e:
        print(f"   JSONDecodeError: {e.msg} at position {e.pos}")
        print(f"   Action: Log it, return None, don't retry")

    # 2. Pydantic validation error — JSON parsed but wrong shape
    print("\n2. Pydantic validation error:")
    class User(BaseModel):
        name: str
        age: int

    try:
        User(name="John", age="not_a_number")
    except ValidationError as e:
        # e.errors() returns a list of dicts describing each failure
        for error in e.errors():
            print(f"   Field '{error['loc'][0]}': {error['msg']}")
        print(f"   Action: Log it, return None, don't retry")

    # 3. Connection error — Ollama not running
    print("\n3. Connection error (simulated):")
    bad_client = OpenAI(
        base_url="http://localhost:9999/v1",  # wrong port
        api_key="ollama"
    )
    try:
        bad_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10
        )
    except Exception as e:
        print(f"   {type(e).__name__}: {str(e)[:80]}")
        print(f"   Action: Retry with backoff")

demonstrate_exceptions()

# ============================================================
# SECTION 2: Retry with exponential backoff
# ============================================================

print("\n" + "=" * 50)
print("SECTION 2: Retry with exponential backoff")
print("=" * 50)

# Exponential backoff = wait longer between each retry
# Retry 1: wait 1s
# Retry 2: wait 2s
# Retry 3: wait 4s
# This prevents hammering a struggling API

def call_with_retry(
    system: str,
    user: str,
    temperature: float = 0.0,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> Optional[str]:
    """
    Make an LLM call with exponential backoff retry.

    Args:
        system: system prompt
        user: user message
        temperature: randomness (0=deterministic)
        max_retries: how many times to retry on failure
        base_delay: starting delay in seconds (doubles each retry)

    Returns:
        model response text or None if all retries failed
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature,
                max_tokens=300
            )
            # Success — return immediately
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{max_retries} failed: {type(e).__name__}")

            if attempt < max_retries:
                # Exponential backoff with jitter
                # Jitter = small random amount added to prevent
                # multiple agents retrying at exactly the same time
                delay = base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.5)
                total_delay = delay + jitter
                print(f"  Waiting {total_delay:.1f}s before retry...")
                time.sleep(total_delay)

    print(f"  All {max_retries} attempts failed. Last error: {last_error}")
    return None


# Test with a working call
print("\nTesting call_with_retry (should succeed first attempt):")
result = call_with_retry(
    system="You are a helpful assistant.",
    user="Say 'hello' in exactly one word."
)
print(f"Result: {result}")

# Test with broken client to see retry behaviour
print("\nTesting retry behaviour with broken endpoint:")
broken_client_backup = client
# Temporarily point to wrong port to simulate failure

def simulate_flaky_call(fail_times: int = 2) -> Optional[str]:
    """Simulates a call that fails N times then succeeds."""
    attempt_count = 0

    def flaky_llm_call() -> str:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count <= fail_times:
            raise ConnectionError(f"Simulated network failure #{attempt_count}")
        return "Success after failures"

    last_error = None
    for attempt in range(1, 4):
        try:
            result = flaky_llm_call()
            print(f"  Attempt {attempt}: SUCCESS — {result}")
            return result
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}: FAILED — {e}")
            if attempt < 3:
                time.sleep(0.5)

    return None

simulate_flaky_call(fail_times=2)

# ============================================================
# SECTION 3: Safe extraction — full error handling pipeline
# ============================================================

print("\n" + "=" * 50)
print("SECTION 3: Safe extraction pipeline")
print("=" * 50)

T = TypeVar("T", bound=BaseModel)


def clean_json(text: str) -> str:
    """Strip markdown fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def safe_extract(
    text: str,
    model_class: Type[T],
    max_retries: int = 3
) -> Optional[T]:
    """
    Extract structured data with full error handling.

    Handles three failure modes separately:
    1. LLM call failure → retry with backoff
    2. JSON parse failure → retry (model might give better output)
    3. Pydantic validation failure → fail fast (structural problem)

    Returns validated Pydantic model or None.
    """
    schema = json.dumps(model_class.model_json_schema(), indent=2)

    system = f"""Extract information and return ONLY a valid JSON object.
No markdown, no explanation. Raw JSON only.

Schema:
{schema}"""

    for attempt in range(1, max_retries + 1):
        print(f"  Attempt {attempt}/{max_retries}")

        # Step 1: Make the LLM call
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                max_tokens=500
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            # Network/API error — retry-able
            print(f"    LLM call failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            continue

        # Step 2: Parse JSON
        try:
            cleaned = clean_json(raw)
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Bad JSON — retry (model might do better next time)
            print(f"    JSON parse failed: {e.msg}")
            if attempt < max_retries:
                time.sleep(1)
            continue

        # Step 3: Validate with Pydantic
        try:
            result = model_class(**data)
            print(f"    Success on attempt {attempt}")
            return result
        except ValidationError as e:
            # Wrong shape — fail fast, retrying won't help
            # The model understood the format but gave wrong types
            print(f"    Validation failed: {[err['msg']for err in e.errors()]}")
            return None  # don't retry validation errors

    print(f"  All {max_retries} attempts exhausted")
    return None


# Test safe_extract
class ArticleSummary(BaseModel):
    title: str
    author: str
    key_points: list[str]
    word_count_estimate: int
    is_technical: bool


article = """
'Building Production AI Agents' by Sarah Chen explores the challenges
of deploying LLM-based systems at scale. The article covers error handling
strategies, monitoring approaches, and cost optimisation techniques.
Chen argues that most agent failures come from poor error handling rather
than model capability limitations. The piece is approximately 2000 words
and targets senior engineers.
"""

print("\nExtracting article summary:")
summary = safe_extract(article, ArticleSummary)
if summary:
    print(f"Title: {summary.title}")
    print(f"Author: {summary.author}")
    for topic in summary.key_points:
        print(f"{topic[0].upper}:\n{topic[1][:100]}.")
    print(f"Technical: {summary.is_technical}")
    print(f"Word count estimate: {summary.word_count_estimate}")

    print(f"{summary.is_technical}, it's {'' if summary.is_technical else 'not '}technical.")

    # ============================================================
# SECTION 4: Fallback chains
# ============================================================

print("\n" + "=" * 50)
print("SECTION 4: Fallback chains")
print("=" * 50)

# Sometimes extraction fails completely.
# A fallback chain tries progressively simpler approaches
# rather than returning None immediately.

class StrictReview(BaseModel):
    """Full review with all fields."""
    product: str
    rating: int
    pros: list[str]
    cons: list[str]
    verdict: str
    would_recommend: bool


class SimpleReview(BaseModel):
    """Minimal review — fallback if full extraction fails."""
    product: str
    rating: int
    verdict: str


def extract_review_with_fallback(text: str) -> Optional[BaseModel]:
    """
    Try full extraction first.
    Fall back to simple extraction if that fails.
    Return None only if both fail.
    """
    print("  Trying full extraction...")
    result = safe_extract(text, StrictReview, max_retries=2)
    if result:
        print("  Full extraction succeeded")
        return result

    print("  Full extraction failed. Trying simple fallback...")
    result = safe_extract(text, SimpleReview, max_retries=2)
    if result:
        print("  Simple extraction succeeded")
        return result

    print("  All extractions failed")
    return None


review_text = """
This product is amazing! Honestly one of the best purchases I've made.
Rating: 5 stars. Would definitely tell my friends about it.
"""

print("\nExtracting review with fallback chain:")
review = extract_review_with_fallback(review_text)
if review:
    print(f"Got result: {review.model_dump()}")