# week1_project.py
# Week 1 Mini-Project: Research Agent
#
# What it does:
# 1. Takes a research topic
# 2. Calls Gemini to research it (Day 6)
# 3. Structures the output with Pydantic (Day 3)
# 4. Times the API call with your timer decorator (Day 4)
# 5. Saves results to JSON (Day 7)
# 6. Loads and displays past research sessions (Day 7)
#
# Concepts used: type hints, Pydantic, decorators, async, LLM API, JSON, file I/O

import os
import json
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ============================================================
# SETUP
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

# Directory where all research results are saved
RESULTS_DIR = Path("./research_results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# PYDANTIC MODELS (Day 3)
# Define the structure we want from Gemini's response
# ============================================================

class ResearchPoint(BaseModel):
    """A single key finding from the research."""
    point: str = Field(description="The key finding or insight")
    importance: str = Field(description="Why this matters: high, medium, or low")


class ResearchResult(BaseModel):
    """Complete structured output of a research session."""
    topic: str = Field(description="The topic that was researched")
    summary: str = Field(description="2-3 sentence overview of the topic")
    key_points: list[ResearchPoint] = Field(
        description="List of key findings, maximum 4"
    )
    recommended_next_steps: list[str] = Field(
        description="What to learn or do next, maximum 3 items"
    )
    confidence: int = Field(
        ge=0, le=100,
        description="How confident the agent is in this research, 0-100"
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="When this research was conducted"
    )


# ============================================================
# DECORATORS (Day 4)
# ============================================================

def timer(func):
    """Times how long any function takes. Works on both sync and async."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"[TIMER] {func.__name__} completed in {duration:.2f}s")
        return result
    return wrapper


# ============================================================
# CORE AGENT FUNCTIONS
# ============================================================

def save_result(result: ResearchResult) -> Path:
    """Save a research result to disk as JSON."""
    # Create filename from topic — replace spaces with underscores
    safe_topic = result.topic.lower().replace(" ", "_")[:30]
    filename = f"{safe_topic}_{int(time.time())}.json"
    filepath = RESULTS_DIR / filename

    # model_dump() converts Pydantic model to plain dict (Day 3)
    # Then json.dumps() converts dict to JSON string (Day 7)
    filepath.write_text(json.dumps(result.model_dump(), indent=2))
    print(f"[SAVED] Result saved to: {filepath}")
    return filepath


def load_past_results() -> list[ResearchResult]:
    """Load all past research results from disk."""
    results = []
    # glob finds all .json files in RESULTS_DIR
    for filepath in RESULTS_DIR.glob("*.json"):
        data = json.loads(filepath.read_text())
        # Pydantic can reconstruct a model from a dict
        results.append(ResearchResult(**data))
    return results


def research_topic(topic: str) -> ResearchResult:
    """
    Core agent function: research a topic using Gemini.
    Synchronous version — avoids event loop conflicts when calling sequentially.
    """
    print(f"\n[AGENT] Researching: {topic}")

    prompt = f"""
    Research this topic for a developer learning agentic AI: "{topic}"
    
    Respond with ONLY a valid JSON object. No markdown, no backticks, just JSON.
    Use exactly this structure:
    {{
        "topic": "{topic}",
        "summary": "2-3 sentence overview",
        "key_points": [
            {{"point": "finding here", "importance": "high/medium/low"}},
            {{"point": "finding here", "importance": "high/medium/low"}}
        ],
        "recommended_next_steps": [
            "step 1",
            "step 2"
        ],
        "confidence": 85
    }}
    Maximum 4 key points, maximum 3 next steps.
    """

    # Synchronous call — no async needed for sequential single-topic research
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a technical research agent specialising in AI and software development. "
                "You always respond with valid JSON only. Never include markdown formatting."
            ),
            temperature=0.3,
        )
    )

    raw_text = response.text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    data = json.loads(raw_text)

    from datetime import datetime
    data["timestamp"] = datetime.now().isoformat()

    result = ResearchResult(**data)
    return result


@timer
def run_research_session(topic: str) -> ResearchResult:
    """Run a complete research session synchronously."""
    return research_topic(topic)
# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def display_result(result: ResearchResult) -> None:
    """Pretty-print a research result to the terminal."""
    print(f"\n{'='*50}")
    print(f"RESEARCH: {result.topic.upper()}")
    print(f"{'='*50}")
    print(f"\nSUMMARY:\n{result.summary}")
    print(f"\nKEY POINTS:")
    for i, point in enumerate(result.key_points, 1):
        print(f"  {i}. [{point.importance.upper()}] {point.point}")
    print(f"\nNEXT STEPS:")
    for i, step in enumerate(result.recommended_next_steps, 1):
        print(f"  {i}. {step}")
    print(f"\nConfidence: {result.confidence}/100")
    if result.timestamp:
        print(f"Researched at: {result.timestamp}")
    print(f"{'='*50}")


# ============================================================
# MAIN — runs the agent
# ============================================================

if __name__ == "__main__":
    # "__main__" check means: only run this block if the file is run directly.
    # If another file imports week1_project.py, this block won't run.
    # Standard Python convention for all runnable scripts.

    print("🤖 Research Agent — Week 1 Project")
    print("Using: Gemini 2.5 Flash + Pydantic + Async + Decorators\n")

    # Research two topics
    topics = [
        "MCP Model Context Protocol for AI agents",
        "CrewAI multi-agent framework"
    ]

    for topic in topics:
        # Run the research session
        result = run_research_session(topic)

        # Display it
        display_result(result)

        # Save to disk
        save_result(result)

    # Load and summarise all past sessions
    print(f"\n{'='*50}")
    print("PAST RESEARCH SESSIONS:")
    past = load_past_results()
    for r in past:
        print(f"  • {r.topic} (confidence: {r.confidence}/100)")
    print(f"Total sessions on disk: {len(past)}")