"""
Day 30: LangGraph Nodes and Edges Deep Dive

Today we build a REAL agent with:
- LLM calls in nodes
- Tool execution in nodes
- Conditional routing based on LLM decisions
- Proper state management

This is production-ready agent architecture.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen2.5:7b"

# === STATE SCHEMA ===

class ResearchState(TypedDict):
    """
    State for a research agent workflow.
    
    Fields:
    - messages: Conversation history
    - query: Original user query
    - search_results: Results from web search
    - analysis: LLM's analysis of results
    - needs_more_info: Whether agent needs to search again
    - iteration: Current iteration count
    """
    messages: Annotated[list, operator.add]
    query: str
    search_results: list[str]
    analysis: str
    needs_more_info: bool
    iteration: int

# === TOOLS ===

def mock_web_search(query: str) -> list[str]:
    """
    Mock search tool — in production, use Tavily or similar.
    Returns fake search results based on query.
    """
    # Simulate search results
    results = [
        f"Article 1: {query} - comprehensive overview",
        f"Research paper: Recent findings on {query}",
        f"News: Latest developments in {query}"
    ]
    return results

# === NODES ===

def entry_node(state: ResearchState) -> ResearchState:
    """
    Node 1: Initialize the research workflow.
    """
    print(f"\n{'='*60}")
    print(f"🚀 ENTRY NODE")
    print(f"{'='*60}")
    print(f"Query: {state['query']}")
    
    return {
        "messages": [HumanMessage(content=state['query'])],
        "iteration": 0,
        "search_results": [],
        "needs_more_info": True
    }

def search_node(state: ResearchState) -> ResearchState:
    """
    Node 2: Execute web search.
    
    This node performs the actual tool execution.
    In production, this would call Tavily, Google, etc.
    """
    print(f"\n{'='*60}")
    print(f"🔍 SEARCH NODE (Iteration {state['iteration'] + 1})")
    print(f"{'='*60}")
    
    # Execute search
    results = mock_web_search(state['query'])
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result}")
    
    return {
        "search_results": results,
        "iteration": state['iteration'] + 1
    }

def analyze_node(state: ResearchState) -> ResearchState:
    """
    Node 3: LLM analyzes search results.
    
    This is where the LLM reasoning happens.
    The LLM decides: Do I have enough info, or need more searches?
    """
    print(f"\n{'='*60}")
    print(f"🤔 ANALYZE NODE")
    print(f"{'='*60}")
    
    # Build context from search results
    context = "\n".join(state['search_results'])
    
    # Ask LLM to analyze
    system_prompt = """You are a research analyst. Analyze the search results and:
1. Summarize key findings
2. Determine if you have enough information to answer the user's query
3. Respond in JSON format:
{
  "summary": "...",
  "has_enough_info": true/false,
  "reasoning": "..."
}"""
    
    user_prompt = f"""Query: {state['query']}

Search Results:
{context}

Analyze these results."""
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    # Parse LLM response
    try:
        # Try to extract JSON from response
        content = response.choices[0].message.content
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        analysis = json.loads(content.strip())
        
        print(f"\nAnalysis summary: {analysis['summary'][:100]}...")
        print(f"Has enough info: {analysis['has_enough_info']}")
        print(f"Reasoning: {analysis['reasoning']}")
        
        return {
            "analysis": analysis['summary'],
            "needs_more_info": not analysis['has_enough_info'],
            "messages": [AIMessage(content=f"Analysis: {analysis['summary']}")]
        }
    except:
        # Fallback if JSON parsing fails
        return {
            "analysis": response.choices[0].message.content,
            "needs_more_info": False,
            "messages": [AIMessage(content=response.choices[0].message.content)]
        }

def synthesize_node(state: ResearchState) -> ResearchState:
    """
    Node 4: Create final answer.
    
    This is the output node — synthesize everything into a final response.
    """
    print(f"\n{'='*60}")
    print(f"✅ SYNTHESIZE NODE")
    print(f"{'='*60}")
    
    # Build final answer using all gathered info
    system_prompt = "You are a helpful assistant. Provide a clear, concise answer based on the analysis."
    
    user_prompt = f"""Original query: {state['query']}

Analysis: {state['analysis']}

Provide a final answer to the user's query."""
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5
    )
    
    final_answer = response.choices[0].message.content
    
    print(f"\nFinal Answer:\n{final_answer}")
    
    return {
        "messages": [AIMessage(content=final_answer)]
    }

# === CONDITIONAL ROUTING ===

def should_continue_research(state: ResearchState) -> Literal["search", "synthesize"]:
    """
    Conditional edge: Decide if we need more research.
    
    This demonstrates conditional routing based on:
    1. LLM's decision (needs_more_info)
    2. Iteration limits (safety stop)
    
    Returns:
    - "search": Go back to search_node for more info
    - "synthesize": Move to final answer
    """
    # Safety: max 3 iterations
    if state['iteration'] >= 3:
        print(f"\n🛑 ROUTER: Max iterations reached, moving to synthesis")
        return "synthesize"
    
    # LLM decides
    if state['needs_more_info']:
        print(f"\n🔄 ROUTER: Need more info, searching again")
        return "search"
    else:
        print(f"\n✅ ROUTER: Enough info, moving to synthesis")
        return "synthesize"

# === BUILD GRAPH ===

def create_research_agent():
    """
    Build the complete research agent graph.
    
    Flow:
    entry → search → analyze → (conditional) → search OR synthesize
    
    This creates a loop: search → analyze → search → analyze → ...
    until the LLM decides it has enough info.
    """
    workflow = StateGraph(ResearchState)
    
    # Add all nodes
    workflow.add_node("entry", entry_node)
    workflow.add_node("search", search_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("synthesize", synthesize_node)
    
    # Entry always goes to search
    workflow.add_edge("entry", "search")
    
    # Search always goes to analyze
    workflow.add_edge("search", "analyze")
    
    # Analyze has conditional routing
    workflow.add_conditional_edges(
        "analyze",
        should_continue_research,
        {
            "search": "search",  # Loop back
            "synthesize": "synthesize"  # Move to end
        }
    )
    
    # Synthesize ends the workflow
    workflow.add_edge("synthesize", END)
    
    # Set entry point
    workflow.set_entry_point("entry")
    
    return workflow.compile()

# === DEMOS ===

def demo_research_agent():
    """
    Demo: Research agent with conditional looping.
    """
    print("\n" + "="*80)
    print("DEMO: Research Agent with Conditional Routing")
    print("="*80)
    
    agent = create_research_agent()
    
    # Test query
    initial_state = {
        "query": "What are the latest developments in quantum computing?",
        "messages": [],
        "search_results": [],
        "analysis": "",
        "needs_more_info": True,
        "iteration": 0
    }
    
    final_state = agent.invoke(initial_state)
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)
    print(f"Total iterations: {final_state['iteration']}")
    print(f"Messages generated: {len(final_state['messages'])}")

# === MAIN ===

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DAY 30: LangGraph Nodes and Edges")
    print("="*80)
    
    demo_research_agent()
    
    print("\n" + "="*80)
    print("KEY CONCEPTS")
    print("="*80)
    print("""
1. Nodes = where work happens (tool calls, LLM calls, logic)
2. Edges = transitions between nodes
3. Conditional edges = route based on state values
4. Loops = conditional edge pointing back to earlier node
5. This pattern handles: search → analyze → maybe search again → final answer

Tomorrow (Day 31): State management patterns and parallel execution
    """)