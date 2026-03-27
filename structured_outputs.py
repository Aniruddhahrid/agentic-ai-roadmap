# structured_outputs.py
# Structured outputs — forcing reliable, validated data from LLMs.
# Combines prompt engineering (Day 16) + Pydantic (Day 3) + error handling.

import json
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen2.5:7b"


def call(system: str, user: str, temperature: float = 0.0) -> str:
    """Core helper — returns raw text response."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


def clean_json(text: str) -> str:
    """
    Strip markdown fences from LLM output.
    Models often wrap JSON in ```json ... ``` despite being told not to.
    This function handles that case reliably.
    """
    text = text.strip()
    if text.startswith("```"):
        # Split on ``` and take the middle part
        parts = text.split("```")
        # parts[1] contains the code block content
        text = parts[1]
        # Remove language identifier if present (json, python, etc.)
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


# ============================================================
# SECTION 1: Basic structured extraction
# ============================================================

print("=" * 50)
print("SECTION 1: Basic structured extraction")
print("=" * 50)

# Define what we want the output to look like using Pydantic
class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str
    city: str


# The prompt tells the model exactly what JSON to return
system = """Extract information from the text and return ONLY a JSON object.
No markdown, no explanation, just the JSON.

Required format:
{
    "name": "string",
    "age": integer,
    "occupation": "string",
    "city": "string"
}"""

text = "Sarah Chen, 34, is a machine learning engineer based in San Francisco who specialises in LLM applications."

raw_response = call(system, text)
print(f"Raw response:\n{raw_response}")

# Clean and parse
cleaned = clean_json(raw_response)
data = json.loads(cleaned)

# Validate with Pydantic
person = PersonInfo(**data)
print(f"\nValidated output:")
print(f"  Name: {person.name}")
print(f"  Age: {person.age}")
print(f"  Occupation: {person.occupation}")
print(f"  City: {person.city}")
print(f"  Type check — age is int: {type(person.age) == int}")

# ============================================================
# SECTION 2: Nested structured output
# ============================================================

print("\n" + "=" * 50)
print("SECTION 2: Nested structured output")
print("=" * 50)

# Nested Pydantic models — same as Day 3 but now driven by LLM output
class Skill(BaseModel):
    name: str
    level: str = Field(description="beginner, intermediate, or expert")


class DeveloperProfile(BaseModel):
    name: str
    years_experience: int
    primary_language: str
    skills: list[Skill]
    is_available: bool


system2 = """Extract developer information and return ONLY a JSON object.
No markdown, no explanation.

Required format:
{
    "name": "string",
    "years_experience": integer,
    "primary_language": "string",
    "skills": [
        {"name": "string", "level": "beginner/intermediate/expert"}
    ],
    "is_available": boolean
}"""

bio = """
Alex Rivera has 6 years of experience as a backend developer.
Python is their main language. They're an expert in FastAPI and Docker,
intermediate with Kubernetes, and a beginner with Rust.
Currently open to new opportunities.
"""

raw = call(system2, bio)
cleaned = clean_json(raw)
data = json.loads(cleaned)
profile = DeveloperProfile(**data)

print(f"Name: {profile.name}")
print(f"Experience: {profile.years_experience} years")
print(f"Available: {profile.is_available}")
print(f"Skills:")
for skill in profile.skills:
    print(f"  - {skill.name}: {skill.level}")

    # ============================================================
# SECTION 3: The extract() pattern — reusable structured extraction
# ============================================================

print("\n" + "=" * 50)
print("SECTION 3: The extract() pattern")
print("=" * 50)

from pydantic import BaseModel
from typing import TypeVar, Type

# TypeVar lets us make extract() work with ANY Pydantic model
# T = "some Pydantic model type" — generic placeholder
T = TypeVar("T", bound=BaseModel)


def extract(text: str, model_class: Type[T], context: str = "") -> Optional[T]:
    """
    Extract structured data from any text into any Pydantic model.
    
    This is the core pattern for LLM structured output:
    1. Build a prompt from the model's schema
    2. Call the LLM
    3. Clean the response
    4. Parse JSON
    5. Validate with Pydantic
    6. Return the validated object or None on failure
    
    Args:
        text: the input text to extract from
        model_class: any Pydantic BaseModel class
        context: optional extra instructions for the extraction
    
    Returns:
        validated Pydantic model instance or None if extraction failed
    """
    # Build schema description from Pydantic model
    # model_json_schema() returns the JSON schema of the model
    schema = json.dumps(model_class.model_json_schema(), indent=2)

    system = f"""Extract information from the provided text and return ONLY a JSON object.
No markdown formatting, no explanation, just the raw JSON.

{'Additional context: ' + context if context else ''}

JSON schema to follow:
{schema}"""

    try:
        raw = call(system, text)
        cleaned = clean_json(raw)
        data = json.loads(cleaned)
        return model_class(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        # Return None on failure — caller decides what to do
        print(f"  [EXTRACT FAILED] {e}")
        return None


# Test extract() with different models on different texts

class CompanyInfo(BaseModel):
    name: str
    industry: str
    founded_year: int
    employee_count: str
    headquarters: str


class ProductReview(BaseModel):
    product_name: str
    rating: int = Field(ge=1, le=5)
    sentiment: str
    key_issues: list[str]
    would_recommend: bool


# Test 1 — company extraction
company_text = """
Anthropic is an AI safety company founded in 2021 by former OpenAI researchers.
Headquartered in San Francisco, the company has around 500 employees and focuses
on building reliable, interpretable AI systems.
"""

company = extract(company_text, CompanyInfo)
if company:
    print(f"Company: {company.name} ({company.industry})")
    print(f"Founded: {company.founded_year} | HQ: {company.headquarters}")

# Test 2 — review extraction
review_text = """
I've been using this AI writing tool for 3 months. Rating: 2/5.
It constantly loses context, the suggestions are often irrelevant,
and it crashed twice last week. I would not recommend it to anyone
doing serious work. The only good thing is the interface looks nice.
"""

review = extract(review_text, ProductReview)
if review:
    print(f"\nProduct: {review.product_name}")
    print(f"Rating: {review.rating}/5 | Recommend: {review.would_recommend}")
    print("Issues:\n")
    for ele in review.key_issues:
        print(f"{ele}")

# ============================================================
# SECTION 4: Optional fields — handling incomplete data
# ============================================================

print("\n" + "=" * 50)
print("SECTION 4: Optional fields")
print("=" * 50)


class JobPosting(BaseModel):
    title: str
    company: str
    location: str
    salary_range: Optional[str] = None    # might not be mentioned
    remote: Optional[bool] = None         # might not be clear
    required_skills: list[str] = []       # default empty list
    years_experience: Optional[int] = None


# Incomplete posting — missing salary and remote info
incomplete_posting = """
Software Engineer at TechFlow.
Based in Bangalore. Must know Python and PostgreSQL.
We're looking for someone with 3+ years of experience.
"""

job = extract(incomplete_posting, JobPosting)
if job:
    print(f"Title: {job.title} at {job.company}")
    print(f"Location: {job.location}")
    print(f"Salary: {job.salary_range or 'Not specified'}")
    print(f"Remote: {job.remote if job.remote is not None else 'Not specified'}")
    print(f"Skills: {job.required_skills}")
    print(f"Experience: {job.years_experience} years")

# ============================================================
# SECTION 5: Batch extraction
# ============================================================

print("\n" + "=" * 50)
print("SECTION 5: Batch extraction")
print("=" * 50)


class NewsHeadline(BaseModel):
    topic: str
    sentiment: str    # positive, negative, neutral
    is_ai_related: bool


headlines = [
    "OpenAI announces GPT-5 with unprecedented reasoning capabilities",
    "Stock markets tumble amid rising inflation concerns",
    "New study shows AI can detect cancer earlier than doctors",
    "Local football team wins championship after 20-year drought",
    "Anthropic raises $2B to accelerate AI safety research",
]

results: list[NewsHeadline] = []
failed = 0

for headline in headlines:
    result = extract(headline, NewsHeadline)
    if result:
        results.append(result)
    else:
        failed += 1

print(f"Extracted: {len(results)}/{len(headlines)} | Failed: {failed}")
print("\nAI-related headlines:")
for r in results:
    if r.is_ai_related:
        print(f"  [{r.sentiment.upper()}] {r.topic}")