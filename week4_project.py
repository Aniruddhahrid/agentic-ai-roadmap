"""
Day 28: Week 4 Mini-Project — Complete Agent System

Combines everything from Week 4:
- Agent loop (Days 22-24)
- Tools (Day 22)
- Task decomposition (Day 24)
- Embeddings & RAG (Days 25-26)
- ChromaDB for vector storage (Day 27)

This is a production-ready foundation for agentic systems.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

CHAT_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text"

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="./week4_agent_db")

# === KNOWLEDGE BASE ===

# Domain: Customer support agent for a SaaS company
COMPANY_KNOWLEDGE = [
    "CloudSync is a cloud storage and file synchronization service founded in 2021. We offer secure file storage, sharing, and collaboration tools.",
    "Our pricing: Free (5GB), Personal ($9.99/month for 100GB), Team ($19.99/user/month for 1TB + collaboration features), Enterprise (custom pricing, unlimited storage).",
    "File sync works across Windows, Mac, Linux, iOS, and Android. Desktop app auto-syncs folders, mobile apps support offline access.",
    "We offer 256-bit AES encryption at rest and TLS 1.3 in transit. Enterprise plans include zero-knowledge encryption where we can't access your files.",
    "Customer support: Free users get email support (48hr response), paid users get priority email + live chat 9am-9pm EST Mon-Fri.",
    "File versioning: Keep up to 30 days of file history on Free/Personal, 1 year on Team, unlimited on Enterprise. Restore any version anytime.",
    "Sharing: Generate password-protected share links with expiry dates. Team/Enterprise get advanced permissions (view-only, edit, download control).",
    "Desktop app troubleshooting: If sync stops, try 'Pause and Resume' from system tray icon. Check firewall isn't blocking port 443.",
    "Mobile app works offline — changes sync automatically when connection restored. Starred files always available offline.",
    "Team plan includes: shared folders with permissions, activity logs, admin console for user management, and SSO integration.",
]


# === TOOLS ===

def get_current_time() -> str:
    """Returns current date and time."""
    return json.dumps({
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "UTC"
    })


def calculate(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        result = eval(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    Search the company knowledge base using RAG.
    This is the key integration — combining tools with RAG.
    """
    # Get or create knowledge collection
    try:
        collection = chroma_client.get_collection("company_knowledge")
    except:
        # First time — create and populate
        collection = chroma_client.create_collection("company_knowledge")
        
        # Embed and add all knowledge documents
        embeddings = []
        for doc in COMPANY_KNOWLEDGE:
            emb = get_embedding(doc)
            embeddings.append(emb)
        
        collection.add(
            documents=COMPANY_KNOWLEDGE,
            embeddings=embeddings,
            ids=[f"kb_{i}" for i in range(len(COMPANY_KNOWLEDGE))]
        )
    
    # Search
    query_emb = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    
    # Format results
    docs = results['documents'][0]
    return json.dumps({
        "found": len(docs),
        "documents": docs
    })


def get_embedding(text: str) -> List[float]:
    """Get embedding vector for text."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


# Tool definitions for LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}}
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
                    "expression": {"type": "string", "description": "Math expression like '2 + 2' or '15 * 8'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search company knowledge base for information about products, policies, troubleshooting, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results to return", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]

AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base
}


# === AGENT WITH MEMORY ===

class Agent:
    """
    Production-ready agent with:
    - Tool execution (Day 22)
    - Conversation memory (Day 19)
    - RAG integration (Days 25-27)
    - Task planning (Day 24)
    """
    
    def __init__(self):
        self.conversation_history = []
        self.system_prompt = """You are a helpful customer support agent for CloudSync, a cloud storage service.

You have access to:
1. search_knowledge_base: Look up company information, policies, troubleshooting steps
2. calculate: Perform calculations (e.g., storage costs)
3. get_current_time: Get current date/time

Always:
- Search knowledge base when you need factual info about the company
- Be helpful and concise
- If you don't know something, search the knowledge base before saying you don't know
"""
    
    def run(self, user_message: str, max_iterations: int = 5) -> str:
        """
        Run the agent loop.
        
        This combines:
        - Day 23's ReAct loop
        - Day 19's conversation memory
        - Day 26's RAG (via search_knowledge_base tool)
        """
        print(f"\n{'='*80}")
        print(f"USER: {user_message}")
        print(f"{'='*80}")
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages for this turn
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history
        
        # Agent loop
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=TOOLS
            )
            
            assistant_message = response.choices[0].message
            
            # Add to conversation history
            self.conversation_history.append(assistant_message.model_dump())
            messages.append(assistant_message.model_dump())
            
            # Check for tool calls
            if assistant_message.tool_calls:
                print(f"🔧 Agent wants to use tools:")
                
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    print(f"   - {func_name}({func_args})")
                    
                    # Execute tool
                    func_to_call = AVAILABLE_FUNCTIONS[func_name]
                    func_result = func_to_call(**func_args)
                    
                    print(f"   → Result: {func_result[:100]}...")
                    
                    # Add tool result to conversation
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": func_result
                    }
                    
                    self.conversation_history.append(tool_message)
                    messages.append(tool_message)
            
            else:
                # Agent gave final answer
                print(f"\n✅ AGENT: {assistant_message.content}")
                return assistant_message.content
        
        return "Agent exceeded maximum iterations."
    
    def reset(self):
        """Clear conversation history for new session."""
        self.conversation_history = []


# === DEMOS ===

def demo_simple_query():
    """Demo 1: Simple question requiring knowledge base search."""
    print("\n" + "="*80)
    print("DEMO 1: Simple Knowledge Base Query")
    print("="*80)
    
    agent = Agent()
    agent.run("What pricing plans do you offer?")


def demo_multi_turn():
    """Demo 2: Multi-turn conversation with memory."""
    print("\n" + "="*80)
    print("DEMO 2: Multi-Turn Conversation with Memory")
    print("="*80)
    
    agent = Agent()
    
    # Turn 1
    agent.run("I'm on the Free plan. How much storage do I have?")
    
    # Turn 2 — agent remembers previous context
    agent.run("If I upgrade to Personal, how much more storage would I get?")
    
    # Turn 3 — requires calculation
    agent.run("Calculate the cost for a year of Personal plan")


def demo_tool_chaining():
    """Demo 3: Agent using multiple tools in sequence."""
    print("\n" + "="*80)
    print("DEMO 3: Tool Chaining")
    print("="*80)
    
    agent = Agent()
    agent.run("Search the knowledge base for Team plan features, then calculate the annual cost for 5 users")


def demo_troubleshooting():
    """Demo 4: Customer support troubleshooting scenario."""
    print("\n" + "="*80)
    print("DEMO 4: Troubleshooting Support")
    print("="*80)
    
    agent = Agent()
    agent.run("My desktop app stopped syncing files. What should I do?")


def demo_comparison():
    """Demo 5: Compare multiple options from knowledge base."""
    print("\n" + "="*80)
    print("DEMO 5: Comparison Query")
    print("="*80)
    
    agent = Agent()
    agent.run("Compare the file versioning features between Personal and Enterprise plans")


# === MAIN ===

if __name__ == "__main__":
    print("\n" + "="*80)
    print("WEEK 4 MINI-PROJECT: Complete Agent System")
    print("="*80)
    print("\nThis agent combines:")
    print("  - Tools (calculator, time, knowledge search)")
    print("  - RAG (ChromaDB for knowledge retrieval)")
    print("  - Memory (multi-turn conversations)")
    print("  - ReAct loop (autonomous tool use)")
    
    # Run all demos
    demo_simple_query()
    demo_multi_turn()
    demo_tool_chaining()
    demo_troubleshooting()
    demo_comparison()
    
    print("\n" + "="*80)
    print("WEEK 4 COMPLETE!")
    print("="*80)
    print("""
You've built a production-ready agent foundation:

✅ Day 22: Function calling & tool dispatch
✅ Day 23: ReAct loop (think→act→observe)
✅ Day 24: Task decomposition & planning
✅ Day 25: Embeddings & semantic similarity
✅ Day 26: RAG pipeline
✅ Day 27: ChromaDB vector database
✅ Day 28: Complete integrated system

What you can build with this:
- Customer support bots with knowledge bases
- Research assistants that cite sources
- Code documentation Q&A systems
- Internal company chatbots

Next week (Week 5): LangGraph for complex agent workflows!
    """)