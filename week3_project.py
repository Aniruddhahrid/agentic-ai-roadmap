# week3_project.py
# Week 3 Mini-Project: Code Review Pipeline
#
# A three-stage prompt chain:
# Stage 1: Analyse  → understand what the code does
# Stage 2: Critique → identify specific issues
# Stage 3: Improve  → generate a better version
#
# Concepts used:
# - Structured outputs + Pydantic (Day 17)
# - Error handling + safe_extract (Day 18)
# - Multi-turn context passing (Day 19)
# - Prompt engineering — system prompts, XML tags (Day 16)
# - LLM API parameters (Day 15)

import json
import time
from typing import Optional, TypeVar, Type
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen2.5:7b"


# ============================================================
# PYDANTIC MODELS — one per pipeline stage
# ============================================================

class CodeAnalysis(BaseModel):
    """Stage 1 output — what does this code do?"""
    purpose: str = Field(description="One sentence: what this function does")
    inputs: list[str] = Field(description="List of input parameters and their types")
    output: str = Field(description="What the function returns")
    complexity: str = Field(description="simple, moderate, or complex")


class CodeCritique(BaseModel):
    """Stage 2 output — what's wrong with it?"""
    issues: list[str] = Field(description="Specific problems found, empty if none")
    severity: str = Field(description="none, low, medium, or high")
    missing_features: list[str] = Field(
        description="Things that should be added",
        default=[]
    )
    security_concerns: list[str] = Field(
        description="Security issues if any",
        default=[]
    )


class CodeImprovement(BaseModel):
    """Stage 3 output — improved version."""
    improved_code: str = Field(description="The complete improved function")
    changes_made: list[str] = Field(description="List of specific changes made")
    explanation: str = Field(description="One paragraph explaining the improvements")


class PipelineResult(BaseModel):
    """Complete result from all three stages."""
    original_code: str
    analysis: CodeAnalysis
    critique: CodeCritique
    improvement: CodeImprovement
    stages_completed: int
    total_time_seconds: float


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_json(text: str) -> str:
    """Strip markdown fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


T = TypeVar("T", bound=BaseModel)


def safe_extract(
    system: str,
    user: str,
    model_class: Type[T],
    max_retries: int = 3
) -> Optional[T]:
    """
    Make an LLM call and extract structured output safely.
    Retries on JSON parse errors, fails fast on validation errors.
    Returns None if all attempts fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.0,
                max_tokens=800
            )
            raw = response.choices[0].message.content.strip()
            cleaned = clean_json(raw)
            data = json.loads(cleaned)
            return model_class(**data)

        except json.JSONDecodeError as e:
            print(f"    [Attempt {attempt}] JSON parse error: {e.msg}")
            if attempt < max_retries:
                time.sleep(1)

        except ValidationError as e:
            print(f"    [Attempt {attempt}] Validation error: {[err['msg'] for err in e.errors()]}")
            return None  # fail fast on validation errors

        except Exception as e:
            print(f"    [Attempt {attempt}] API error: {type(e).__name__}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    return None


# ============================================================
# PIPELINE STAGES
# ============================================================

def stage_1_analyse(code: str) -> Optional[CodeAnalysis]:
    """
    Stage 1: Understand what the code does.
    No prior context needed — fresh look at the code.
    """
    print("  Running Stage 1: Analysis...")

    system = """You are a senior Python developer performing code analysis.
Analyse the provided code and return ONLY a JSON object. No markdown.

Required JSON format:
{
    "purpose": "one sentence describing what this function does",
    "inputs": ["param1: type", "param2: type"],
    "output": "what the function returns",
    "complexity": "simple/moderate/complex"
}"""

    user = f"""<code>
{code}
</code>

Analyse this code and return the JSON object."""

    return safe_extract(system, user, CodeAnalysis)


def stage_2_critique(code: str, analysis: CodeAnalysis) -> Optional[CodeCritique]:
    """
    Stage 2: Identify issues.
    Uses Stage 1's analysis as context — this is the chain in action.
    The model knows what the code is supposed to do before critiquing it.
    """
    print("  Running Stage 2: Critique...")

    system = """You are a strict code reviewer at a production AI company.
You have already analysed what the code does. Now critique it for issues.
Return ONLY a JSON object. No markdown.

Required JSON format:
{
    "issues": ["specific issue 1", "specific issue 2"],
    "severity": "none/low/medium/high",
    "missing_features": ["thing that should be added"],
    "security_concerns": ["security issue if any"]
}

If no issues found, use empty lists and severity "none"."""

    # Pass the analysis context forward — this is what makes it a chain
    # Stage 2 knows what Stage 1 found
    user = f"""<analysis>
Purpose: {analysis.purpose}
Complexity: {analysis.complexity}
</analysis>

<code>
{code}
</code>

Critique this code and return the JSON object."""

    return safe_extract(system, user, CodeCritique)


def stage_3_improve(
    code: str,
    analysis: CodeAnalysis,
    critique: CodeCritique
) -> Optional[CodeImprovement]:
    """
    Stage 3: Generate improved version.
    Uses BOTH Stage 1 and Stage 2 outputs as context.
    The chain carries full context forward to the final stage.
    """
    print("  Running Stage 3: Improvement...")

    # Only ask for improvements if there are actual issues
    if critique.severity == "none" and not critique.issues:
        print("    No issues found — skipping improvement")
        return CodeImprovement(
            improved_code=code,
            changes_made=["No changes needed — code is already clean"],
            explanation="The original code has no significant issues."
        )

    system = """You are a senior Python developer improving code.
You have the analysis and critique. Now write an improved version.
Return ONLY a JSON object. No markdown.

Required JSON format:
{
    "improved_code": "the complete improved function as a string",
    "changes_made": ["specific change 1", "specific change 2"],
    "explanation": "one paragraph explaining the improvements"
}"""

    # Full context chain — both previous stages feed into this one
    issues_str = "\n".join(f"- {i}" for i in critique.issues) or "None"
    security_str = "\n".join(f"- {s}" for s in critique.security_concerns) or "None"

    user = f"""<analysis>
Purpose: {analysis.purpose}
Complexity: {analysis.complexity}
</analysis>

<critique>
Issues:
{issues_str}
Security concerns:
{security_str}
Severity: {critique.severity}
</critique>

<original_code>
{code}
</original_code>

Write an improved version and return the JSON object."""

    return safe_extract(system, user, CodeImprovement)


# ============================================================
# MAIN PIPELINE RUNNER
# ============================================================

def run_pipeline(code: str) -> Optional[PipelineResult]:
    """
    Run the full three-stage code review pipeline.

    Each stage feeds its output into the next.
    If any stage fails, the pipeline stops gracefully.
    Returns a complete PipelineResult or None if pipeline failed.
    """
    print(f"\nStarting Code Review Pipeline")
    print(f"{'=' * 40}")
    start_time = time.time()
    stages_completed = 0

    # Stage 1
    analysis = stage_1_analyse(code)
    if not analysis:
        print("Pipeline failed at Stage 1")
        return None
    stages_completed += 1
    print(f"  ✓ Stage 1 complete: {analysis.purpose}.")

    # Stage 2 — receives Stage 1 output
    critique = stage_2_critique(code, analysis)
    if not critique:
        print("Pipeline failed at Stage 2")
        return None
    stages_completed += 1
    print(f"  ✓ Stage 2 complete: {critique.severity} severity, {len(critique.issues)} issues")

    # Stage 3 — receives Stage 1 and Stage 2 outputs
    improvement = stage_3_improve(code, analysis, critique)
    if not improvement:
        print("Pipeline failed at Stage 3")
        return None
    stages_completed += 1
    print(f"  ✓ Stage 3 complete: {len(improvement.changes_made)} changes made")

    total_time = time.time() - start_time

    return PipelineResult(
        original_code=code,
        analysis=analysis,
        critique=critique,
        improvement=improvement,
        stages_completed=stages_completed,
        total_time_seconds=round(total_time, 2)
    )


def display_result(result: PipelineResult) -> None:
    """Pretty print the pipeline result."""
    print(f"\n{'=' * 50}")
    print("CODE REVIEW PIPELINE RESULT")
    print(f"{'=' * 50}")

    print(f"\n📋 ANALYSIS")
    print(f"  Purpose: {result.analysis.purpose}")
    print(f"  Complexity: {result.analysis.complexity}")
    print(f"  Inputs: {', '.join(result.analysis.inputs)}")
    print(f"  Output: {result.analysis.output}")

    print(f"\n🔍 CRITIQUE")
    print(f"  Severity: {result.critique.severity.upper()}")
    if result.critique.issues:
        print(f"  Issues:")
        for issue in result.critique.issues:
            print(f"    - {issue}")
    if result.critique.security_concerns:
        print(f"  Security:")
        for concern in result.critique.security_concerns:
            print(f"    ⚠️  {concern}")

    print(f"\n✨ IMPROVEMENTS")
    print(f"  Changes made:")
    for change in result.improvement.changes_made:
        print(f"    - {change}")
    print(f"\n  Explanation: {result.improvement.explanation}")

    print(f"\n  Improved code:")
    print(f"  {'-' * 40}")
    for line in result.improvement.improved_code.split('\n'):
        print(f"  {line}")

    print(f"\n⏱️  Pipeline completed in {result.total_time_seconds}s")
    print(f"   Stages completed: {result.stages_completed}/3")


# ============================================================
# TEST CASES
# ============================================================

if __name__ == "__main__":

    # Test case 1 — code with multiple issues
    bad_code = '''
def get_user_data(id):
    import requests
    url = "http://api.example.com/users/" + id
    r = requests.get(url)
    data = r.json()
    return data["user"]
'''

    result1 = run_pipeline(bad_code)
    if result1:
        display_result(result1)

    print("\n" + "=" * 50)

    # Test case 2 — cleaner code
    better_code = '''
def calculate_average(numbers: list[float]) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
'''

    result2 = run_pipeline(better_code)
    if result2:
        display_result(result2)