# # pydantic_practice.py
# # Pydantic: type hints that actually enforce themselves at runtime.
# # Used in LangChain, FastAPI, CrewAI, and every major agent framework.

# from pydantic import BaseModel
from typing import Optional

# # ============================================================
# # SECTION 1: Your first Pydantic model
# # ============================================================

# # A Pydantic model is a class that inherits from BaseModel.
# # Each class attribute is a field with a type hint.
# # Pydantic reads those type hints and validates all data against them.

# class User(BaseModel):
#     name: str
#     age: int
#     email: str
#     is_active: bool = True  # = True means this field has a default value
#                              # if you don't provide it, it defaults to True


# # Creating an instance — Pydantic validates automatically on creation
# user = User(name="Anirudh", age=22, email="anirudh@example.com")
# print(user)
# # name='Anirudh' age=22 email='anirudh@example.com' is_active=True

# # Access fields like normal class attributes
# print(user.name)      # Anirudh
# print(user.age)       # 22
# print(user.is_active) # True — the default kicked in

# # Convert to a plain Python dictionary — you'll use this constantly
# # when sending data to APIs or saving to JSON
# print(f" Plain Python dict -- {user.model_dump()}")
# # {'name': 'Anirudh', 'age': 22, 'email': 'anirudh@example.com', 'is_active': True}

# # Convert to JSON string — useful for API responses
# print(f" JSON string -- {user.model_dump_json()}")
# # {"name":"Anirudh","age":22,"email":"anirudh@example.com","is_active":true}

# # ============================================================
# # SECTION 2: What happens when data doesn't match
# # ============================================================

# # Pydantic will try to coerce (convert) data to the right type if it can.
# # "22" as a string can become 22 as an int — Pydantic handles this.
# user2 = User(name="Bob", age="25", email="bob@example.com")
# #                         ^^^^
# #                         This is a string, but age is int.
# #                         Pydantic silently converts it. No error.
# print(user2.age)        # 25 (integer, not string)
# print(f"type converted from string to {type(user2.age)}")  # <class 'int'>

# # But Pydantic won't convert things that make no sense.
# # Wrapping in try/except catches the validation error gracefully.
# from pydantic import ValidationError

# try:
#     bad_user = User(name="Charlie", age="not_a_number", email="charlie@example.com")
#     #                               ^^^^^^^^^^^^^^^^
#     #                               Cannot convert "not_a_number" to int.
#     #                               Pydantic raises ValidationError.
# except ValidationError as e:
#     print(f"\nValidation failed:")
#     print(e)
#     # You'll see a clear, structured error telling you exactly which field
#     # failed and why. Compare this to a generic Python crash 50 lines later.

#     # ============================================================
# # SECTION 3: Field — adding validation rules and descriptions
# # ============================================================

from pydantic import BaseModel, Field

# # Field() replaces the simple = default syntax when you need more control.
# # It's imported from pydantic and adds constraints on top of the type.

# class AgentTask(BaseModel):
#     # description= documents what this field is for.
#     # LangChain reads this description and passes it to the LLM so the
#     # LLM knows what to put in each field. Critical for tool definitions.
#     task: str = Field(description="The instruction to give the agent")

#     # ge= means "greater than or equal to" (minimum value)
#     # le= means "less than or equal to" (maximum value)
#     max_tokens: int = Field(default=1000, ge=1, le=4000,
#                             description="Max response length in tokens")

#     # gt= means "greater than" (strictly, not equal)
#     temperature: float = Field(default=0.7, gt=0.0, le=1.0,
#                                description="Randomness. 0=deterministic, 1=creative")

#     # min_length= and max_length= for strings
#     model: str = Field(default="gemini-2.0-flash", min_length=3,
#                        description="Which LLM model to use")


# # Valid task
# task = AgentTask(task="Summarise AI news", max_tokens=500, temperature=0.3)
# print(task.model_dump())

# # Invalid — temperature above 1.0
# try:
#     bad_task = AgentTask(task="Do something", temperature=1.5)
# except ValidationError as e:
#     print(f"\nTemperature validation failed:")
#     print(e)

# # Invalid — max_tokens below 1
# try:
#     bad_task2 = AgentTask(task="Do something", max_tokens=0)
# except ValidationError as e:
#     print(f"\nmax_tokens validation failed:")
#     print(e)

    # ============================================================
# SECTION 4: Nested models — the real-world shape of agent data
# ============================================================


class Message(BaseModel):
    role: str = Field(description="Either 'user' or 'assistant'")
    content: str = Field(description="The text content of the message")


class Conversation(BaseModel):
    # A list of Message objects — not plain dicts, actual validated Message instances
    messages: list[Message]
    total_tokens_used: int = 0
    model: str = "gemini-2.0-flash"


# Build a conversation — each dict in the list automatically
# becomes a validated Message object. Pydantic handles the conversion.
convo = Conversation(
    messages=[
        {"role": "user", "content": "What is an agent?"},
        {"role": "assistant", "content": "An agent is an AI that takes actions."},
    ]
)

print(convo.model_dump())
# Clean nested dictionary

# Access nested data naturally
print(convo.messages[0].role)     # user
print(convo.messages[1].content)  # An agent is an AI that takes actions.
print(len(convo.messages))        # 2

# Invalid role still passes — Pydantic validates type, not value here.
# Next section shows how to validate values too.

# ============================================================
# SECTION 5: Structured LLM output — the pattern you'll use constantly
# ============================================================

# Imagine you ask an LLM: "Analyse this email and extract key information."
# Instead of getting back a blob of text, you want structured data.
# This is the Pydantic model you'd define for the output:

class EmailAnalysis(BaseModel):
    subject: str = Field(description="The email's subject line")
    tone: str = Field(description="Overall tone: professional, casual, aggressive, friendly")
    score: int = Field(ge=0, le=100, description="Quality score from 0 to 100")
    key_issues: list[str] = Field(default=[], description="List of specific problems found")
    suggested_rewrite: Optional[str] = Field(
        default=None,
        description="Rewritten version if score is below 70, otherwise None"
    )
    is_ready_to_send: bool = Field(description="True if email is ready, False if needs work")


# In real usage, an LLM would populate this.
# For now, we're building it manually to understand the shape.
analysis = EmailAnalysis(
    subject="Follow up on our meeting",
    tone="professional",
    score=65,
    key_issues=[
        "No clear call to action",
        "Too long — recipient won't read past line 3",
        "Missing specific next steps"
    ],
    suggested_rewrite="Hi Sarah, following up on Tuesday's meeting. Can we confirm the budget by Friday? Reply yes/no. Thanks, Anirudh.",
    is_ready_to_send=False
)

print(analysis.model_dump())
print(f"\nScore: {analysis.score}/100")
print(f"Ready to send: {analysis.is_ready_to_send}")
print(f"Issues found: {len(analysis.key_issues)}")
for issue in analysis.key_issues:
    print(f"  - {issue}")