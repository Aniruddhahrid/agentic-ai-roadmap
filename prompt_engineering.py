# prompt_engineering.py
# Prompt engineering using OpenAI SDK pattern via Ollama (local).
# This is the INDUSTRY STANDARD pattern — LangChain, CrewAI, every
# agent framework uses this exact interface. Learning it on Ollama
# means zero changes when switching to GPT-4o or Claude later.

import os
from openai import OpenAI

# OpenAI SDK pointing at Ollama's local server
# This same client works with:
#   OpenAI:    base_url="https://api.openai.com/v1", api_key=os.getenv("OPENAI_API_KEY")
#   Anthropic: different SDK but same concept
#   Any OpenAI-compatible API: just change base_url
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"   # Ollama doesn't check this, but the SDK requires it
)

MODEL = "qwen2.5:7b"


def call(system: str, user: str, temperature: float = 0.0) -> str:
    """
    Core helper function. Makes a chat completion call.
    
    This is the OpenAI messages format — a list of dicts with role and content.
    Every LLM API uses this exact format: system sets behaviour, user is the input.
    This pattern appears in every agent framework you'll use from week 5 onwards.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature,
        max_tokens=500
    )
    # response.choices[0].message.content = the model's reply as a plain string
    return response.choices[0].message.content.strip()


# ============================================================
# SECTION 1: System prompts — role, constraints, format
# ============================================================

print("=" * 50)
print("SECTION 1: System prompt anatomy")
print("=" * 50)

# BAD system prompt — vague, no constraints, no format
bad_system = "You are a helpful assistant."

# GOOD system prompt — role + constraints + output format
good_system = """You are a technical interviewer at a senior AI agent startup.
Your job is to evaluate whether a candidate understands a concept correctly.

Rules:
- Be direct and concise — maximum 3 sentences
- Give a verdict: CORRECT, PARTIALLY CORRECT, or INCORRECT
- If incorrect, explain exactly what's wrong
- Never be encouraging or use filler phrases like "Great question!"

Output format:
VERDICT: [CORRECT/PARTIALLY CORRECT/INCORRECT]
EXPLANATION: [your explanation]"""

user_message = "Can you evaluate this answer: 'An AI agent is just a chatbot that uses GPT.'"

print("\nBad system prompt response:")
print(call(bad_system, user_message))

print("\nGood system prompt response:")
print(call(good_system, user_message))


# ============================================================
# SECTION 2: Few-shot prompting — examples in the prompt
# ============================================================

print("\n" + "=" * 50)
print("SECTION 2: Few-shot vs zero-shot")
print("=" * 50)

zero_shot_system = """Classify the sentiment of customer feedback about an AI product.
Respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL."""

few_shot_system = """Classify the sentiment of customer feedback about an AI product.
Respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.

Examples:
Input: "This saved me 3 hours today, absolutely love it"
Output: POSITIVE

Input: "It works sometimes but crashes randomly, very frustrating"
Output: NEGATIVE

Input: "Does what it says, nothing more nothing less"
Output: NEUTRAL

Input: "Tried it for 5 minutes and gave up, complete waste of time"
Output: NEGATIVE"""

test_inputs = [
    "The agent keeps forgetting context between sessions",
    "Exactly what I needed for my workflow",
    "It's okay I guess, gets the job done",
]

print("\nZero-shot vs Few-shot comparison:")
for text in test_inputs:
    zero = call(zero_shot_system, text)
    few = call(few_shot_system, text)
    print(f"\nInput: '{text}'")
    print(f"  Zero-shot: {zero}")
    print(f"  Few-shot:  {few}")


# ============================================================
# SECTION 3: Chain-of-thought — reasoning before answering
# ============================================================

print("\n" + "=" * 50)
print("SECTION 3: Chain-of-thought prompting")
print("=" * 50)

without_cot = """You are a technical advisor.
Answer technical questions directly and concisely."""

with_cot = """You are a technical advisor.
Before answering any question, you must:
1. ANALYSE: Break down what's actually being asked
2. CONSIDER: List relevant factors and trade-offs
3. CONCLUDE: Give your final recommendation

Always use this exact structure in your response."""

question = """A solo developer is building an AI agent that needs to
process 1000 customer support emails per day, extract structured data,
and store results. Should they use async or sync API calls?"""

print("\nWithout chain-of-thought:")
print(call(without_cot, question))

print("\nWith chain-of-thought:")
print(call(with_cot, question))


# ============================================================
# SECTION 4: XML tags and delimiters
# ============================================================

print("\n" + "=" * 50)
print("SECTION 4: XML tags for structure")
print("=" * 50)

bad_prompt = """Summarise this code and identify any bugs:
def get_user(id):
    return requests.get("https://api.example.com/users/" + id).json()
Keep the summary under 2 sentences."""

good_prompt = """Analyse the code provided and identify issues.

<code>
def get_user(id):
    return requests.get("https://api.example.com/users/" + id).json()
</code>

<instructions>
- Summarise what the code does in one sentence
- List any bugs or issues found
- Keep total response under 4 sentences
</instructions>"""

print("\nBad prompt (no delimiters):")
print(call("You are a code reviewer.", bad_prompt))

print("\nGood prompt (XML tags):")
print(call("You are a code reviewer.", good_prompt))


# ============================================================
# SECTION 5: Production-grade combined prompt
# ============================================================

print("\n" + "=" * 50)
print("SECTION 5: Production-grade combined prompt")
print("=" * 50)

production_system = """You are a code review agent for a Python AI startup.
You review pull requests and provide structured feedback.

Your review must follow this exact format:
SUMMARY: [one sentence describing what the PR does]
ISSUES: [numbered list, or "None found"]
SEVERITY: [BLOCKING/NEEDS_WORK/APPROVED]
RECOMMENDATION: [one sentence action item]

Rules:
- BLOCKING = security issues, data loss risk, or broken functionality
- NEEDS_WORK = code quality, missing tests, unclear naming
- APPROVED = production ready
- Never use encouraging language
- Be specific — reference exact line issues if relevant

Examples:
PR: "adds retry logic to API calls"
SUMMARY: Adds exponential backoff retry decorator to all external API calls.
ISSUES:
1. Retry limit hardcoded to 3 — should be configurable
2. No logging on retry attempts
SEVERITY: NEEDS_WORK
RECOMMENDATION: Extract retry limit to config and add structured logging.

PR: "fixes typo in README"
SUMMARY: Corrects spelling error in installation instructions.
ISSUES: None found
SEVERITY: APPROVED
RECOMMENDATION: Merge immediately."""

pr_description = """
<pr_title>Add user authentication to the agent API</pr_title>

<changes>
def authenticate(token):
    if token == "secret123":
        return True
    return False

@app.route('/agent', methods=['POST'])
def run_agent():
    token = request.headers.get('Authorization')
    if not authenticate(token):
        return {'error': 'unauthorized'}, 401
    # run agent logic
</changes>
"""

print(call(production_system, pr_description, temperature=0.0))

# ============================================================
# SYSTEM PROMPT TEMPLATE — use this structure for every agent
# ============================================================
#
# ROLE:        Who the model is. Be specific.
#              Bad:  "You are a helpful assistant."
#              Good: "You are a senior Python developer specialising
#                     in LLM application architecture."
#
# CONTEXT:     What situation the model is operating in.
#              "You are reviewing code submitted by junior developers
#               at a Series A AI startup."
#
# CONSTRAINTS: What the model must and must not do.
#              "Always point out security vulnerabilities first.
#               Never suggest rewrites longer than 5 lines.
#               Do not use filler phrases."
#
# OUTPUT FORMAT: Exactly what structure the response should follow.
#              "Respond in this format:
#               ISSUE: [one line]
#               SEVERITY: [high/medium/low]
#               FIX: [code snippet]"
#
# The more specific each section, the more reliable the output.