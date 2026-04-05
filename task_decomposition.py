"""
Day 24: Task Decomposition — Planning Before Acting

A reactive agent (Day 23) responds to situations as they arise.
A planning agent decomposes the task first, then executes methodically.

This is the difference between "winging it" and "having a plan."
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import List, Dict, Any

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen2.5:7b"

# === TOOLS (same from Day 23) ===

def get_weather(location: str) -> str:
    """Fake weather API — simulates different temperatures for different cities."""
    # In real world, this would call an actual API
    weather_db = {
        "Tokyo": {"temperature": 22, "conditions": "Partly cloudy"},
        "London": {"temperature": 15, "conditions": "Rainy"},
        "Paris": {"temperature": 18, "conditions": "Sunny"},
        "Mumbai": {"temperature": 32, "conditions": "Hot and humid"},
        "Sydney": {"temperature": 25, "conditions": "Clear"}
    }
    
    data = weather_db.get(location, {"temperature": 20, "conditions": "Unknown"})
    return json.dumps({
        "location": location,
        "temperature": f"{data['temperature']}°C",
        "conditions": data['conditions']
    })

def calculate(expression: str) -> str:
    """Safe calculator for demo purposes."""
    try:
        result = eval(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})

def search_web(query: str) -> str:
    """
    Fake web search — in real world this would call Google/Bing API.
    Simulates finding information on the web.
    """
    # Simulate search results based on query keywords
    if "population" in query.lower():
        if "tokyo" in query.lower():
            return json.dumps({"result": "Tokyo has a population of approximately 14 million"})
        elif "london" in query.lower():
            return json.dumps({"result": "London has a population of approximately 9 million"})
    
    return json.dumps({"result": f"Search results for: {query}"})

# Tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_web": search_web
}

def create_plan(user_query: str) -> List[str]:
    """
    Ask the LLM to break down a complex task into subtasks.
    
    This is a separate LLM call BEFORE we start executing.
    The LLM acts as a "project manager" here — it just plans, doesn't execute.
    
    Why separate? Because planning requires different thinking than execution.
    When you plan, you zoom out. When you execute, you zoom in.
    """
    # This system prompt tells the LLM to act as a planner
    # Notice we DON'T give it tools here — planning is pure reasoning
    planning_messages = [
        {
            "role": "system",
            "content": (
                "You are a task planning assistant. "
                "Break down complex tasks into clear, sequential subtasks. "
                "Return ONLY a numbered list of subtasks, nothing else. "
                "Each subtask should be atomic — one clear action. "
                "Format: 1. [subtask]\\n2. [subtask]\\n..."
            )
        },
        {
            "role": "user",
            "content": f"Break down this task into subtasks:\n\n{user_query}"
        }
    ]
    
    print(f"\n{'='*60}")
    print(f"📋 PLANNING PHASE")
    print(f"{'='*60}")
    print(f"User query: {user_query}\n")
    
    # Call the LLM in "planning mode" — no tools, just thinking
    response = client.chat.completions.create(
        model=MODEL,
        messages=planning_messages,
        temperature=0.3  # Lower temperature = more focused planning
    )
    
    plan_text = response.choices[0].message.content
    print(f"Generated plan:\n{plan_text}\n")
    
    # Parse the numbered list into a Python list
    # This is brittle — real systems use structured outputs (Day 17 Pydantic)
    # But for learning, simple string parsing is fine
    subtasks = []
    for line in plan_text.strip().split('\n'):
        # Remove numbering like "1. " or "1) " or "- "
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Find where the actual task starts (after number and punctuation)
            task = line.split('.', 1)[-1].split(')', 1)[-1].strip()
            if task:
                subtasks.append(task)
    
    return subtasks

def execute_subtask(subtask: str, context: List[Dict[str, Any]]) -> str:
    """
    Execute a single subtask using the ReAct loop from Day 23.
    
    Args:
        subtask: The subtask to execute (e.g., "Get weather for Tokyo")
        context: Results from previous subtasks — the agent can reference these
    
    Returns:
        The result of executing this subtask
    
    Why context? Because subtask 3 might need info from subtask 1.
    Example: "Calculate average" needs the numbers from "Get weather for 3 cities"
    """
    # Build the message with context from previous subtasks
    # This is how the agent "remembers" what it learned earlier
    context_str = ""
    if context:
        context_str = "\n\nContext from previous steps:\n"
        for i, result in enumerate(context, 1):
            context_str += f"Step {i}: {result}\n"
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant executing a subtask. "
                "Use the available tools to complete the task. "
                "Be concise and focus only on this specific subtask."
            )
        },
        {
            "role": "user",
            "content": f"Task: {subtask}{context_str}"
        }
    ]
    
    # ReAct loop — same pattern from Day 23
    # We limit to 3 iterations per subtask to keep it focused
    for iteration in range(3):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS
        )
        
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump())
        
        if assistant_message.tool_calls:
            # Execute tools
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                function_to_call = AVAILABLE_FUNCTIONS[function_name]
                function_response = function_to_call(**function_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
        else:
            # LLM gave a text response — subtask complete
            return assistant_message.content
    
    return "Subtask incomplete after 3 iterations"

def run_planning_agent(user_query: str) -> str:
    """
    The full planning agent:
    1. Create a plan (decompose task into subtasks)
    2. Execute each subtask in order
    3. Synthesize the results into a final answer
    
    This is a "plan-then-execute" architecture.
    Compare this to Day 23's reactive approach.
    """
    # Step 1: Create the plan
    subtasks = create_plan(user_query)
    
    if not subtasks:
        return "Failed to create a plan."
    
    print(f"{'='*60}")
    print(f"🔧 EXECUTION PHASE ({len(subtasks)} subtasks)")
    print(f"{'='*60}\n")
    
    # Step 2: Execute each subtask, accumulating results
    results = []
    for i, subtask in enumerate(subtasks, 1):
        print(f"\n--- Subtask {i}/{len(subtasks)}: {subtask} ---")
        result = execute_subtask(subtask, results)
        print(f"✅ Result: {result}\n")
        results.append(result)
    
    # Step 3: Synthesize the results
    # We ask the LLM to combine all subtask results into a final answer
    print(f"{'='*60}")
    print(f"🎯 SYNTHESIS PHASE")
    print(f"{'='*60}\n")
    
    synthesis_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Synthesize the results below into a final answer."
        },
        {
            "role": "user",
            "content": (
                f"Original question: {user_query}\n\n"
                f"Subtask results:\n" +
                "\n".join(f"{i+1}. {r}" for i, r in enumerate(results)) +
                "\n\nProvide a clear, complete answer to the original question."
            )
        }
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=synthesis_messages
    )
    
    final_answer = response.choices[0].message.content
    print(f"Final answer:\n{final_answer}\n")
    
    return final_answer

if __name__ == "__main__":
    # Test 1: Multi-city comparison (requires planning)
    print("\n" + "="*80)
    print("TEST 1: Compare weather across multiple cities")
    print("="*80)
    result = run_planning_agent(
        "Compare the weather in Tokyo, London, and Paris. "
        "Tell me which city has the highest temperature and by how much."
    )
    
    print("\n" + "="*80)
    print("TEST 2: Multi-step calculation with data gathering")
    print("="*80)
    result = run_planning_agent(
        "Get the temperature in Tokyo and Mumbai. "
        "Calculate the average of these two temperatures. "
        "Then tell me if the average is above or below 25 degrees."
    )

    def run_without_planning(user_query: str) -> str:
        """
        Same task, but without planning — just the Day 23 reactive approach.
        This shows why planning matters.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to tools."
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
        
        for iteration in range(10):  # More iterations because it might take longer
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS
            )
            
            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump())
            
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    function_to_call = AVAILABLE_FUNCTIONS[function_name]
                    function_response = function_to_call(**function_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_response
                    })
            else:
                return assistant_message.content
        
        return "Failed to complete in 10 iterations"

# Add this test to __main__
if __name__ == "__main__":
    # ... previous tests ...
    
    print("\n" + "="*80)
    print("COMPARISON: Same task with and without planning")
    print("="*80)
    
    task = "Get weather for Tokyo and London, then tell me which is warmer."
    
    print("\n--- WITHOUT PLANNING (Day 23 reactive approach) ---")
    result1 = run_without_planning(task)
    print(f"Result: {result1}")
    
    print("\n--- WITH PLANNING (Day 24 decomposition approach) ---")
    result2 = run_planning_agent(task)