"""
Day 26: Simple RAG Pipeline

RAG = Retrieval-Augmented Generation
1. User asks a question
2. Find relevant documents using embeddings (Day 25)
3. Add those documents to the LLM's context
4. LLM answers based on actual information, not guessing

This prevents hallucinations and keeps answers grounded in truth.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from typing import List, Tuple

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

CHAT_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text"

# === EMBEDDING FUNCTIONS (from Day 25) ===

def get_embedding(text: str) -> List[float]:
    """Convert text to embedding vector."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate similarity between two vectors (0 to 1)."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# === KNOWLEDGE BASE ===

# This is our "private data" — information the LLM doesn't know
# In real world, this could be:
# - Company docs, Notion pages, Google Docs
# - Customer support tickets
# - Internal wikis, Slack history
# - Product documentation

KNOWLEDGE_BASE = [
    # Company info
    "TechStart Inc. was founded in January 2020 by Dr. Sarah Chen and Michael Rodriguez in Austin, Texas. The company specializes in AI-powered productivity tools for remote teams.",
    
    # Product info
    "Our flagship product, TaskFlow, uses machine learning to automatically prioritize tasks based on deadlines, team capacity, and project dependencies. It integrates with Slack, Jira, and GitHub.",
    
    # Pricing
    "TaskFlow offers three pricing tiers: Starter ($15/user/month, up to 10 users), Professional ($30/user/month, unlimited users, priority support), and Enterprise (custom pricing, includes dedicated success manager and custom integrations).",
    
    # Support
    "Customer support is available via email (support@techstart.io) Monday-Friday 8am-6pm CST. Professional plan users also get 24/7 live chat support. Enterprise customers have a dedicated Slack channel with our engineering team.",
    
    # Policies
    "We offer a 60-day money-back guarantee on annual plans and 14-day guarantee on monthly plans. Refunds are processed within 5-7 business days to the original payment method.",
    
    # Features - Starter
    "Starter plan includes: basic task management, calendar integration, mobile apps (iOS/Android), and 5GB file storage per user. Limited to 3 projects maximum.",
    
    # Features - Pro
    "Professional plan includes everything in Starter plus: unlimited projects, advanced analytics, custom workflows, API access, 50GB storage per user, and priority email/chat support.",
    
    # Features - Enterprise
    "Enterprise plan includes everything in Professional plus: SSO/SAML authentication, custom integrations, dedicated account manager, SLA guarantees, audit logs, and unlimited storage.",
    
    # Technical
    "TaskFlow is built on a modern tech stack: React frontend, Python FastAPI backend, PostgreSQL database, Redis for caching, and deployed on AWS with 99.9% uptime SLA.",
    
    # Security
    "We are SOC 2 Type II certified, GDPR compliant, and encrypt all data in transit (TLS 1.3) and at rest (AES-256). We conduct annual third-party security audits and penetration testing.",
]


# === RAG PIPELINE ===

def retrieve_relevant_docs(
    query: str,
    knowledge_base: List[str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Step 1 of RAG: RETRIEVAL
    
    Find the most relevant documents from the knowledge base.
    This is exactly what we did in Day 25's semantic search.
    
    Args:
        query: User's question
        knowledge_base: List of documents to search through
        top_k: How many relevant docs to retrieve
    
    Returns:
        List of (document, similarity_score) tuples
    """
    print(f"\n🔍 RETRIEVAL: Finding relevant documents for query...")
    print(f"   Query: '{query}'")
    
    # Embed the query
    query_embedding = get_embedding(query)
    
    # Calculate similarity with each document in knowledge base
    similarities = []
    for doc in knowledge_base:
        doc_embedding = get_embedding(doc)
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc, similarity))
    
    # Sort by similarity and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_docs = similarities[:top_k]
    
    print(f"\n📄 Retrieved {len(top_docs)} documents:")
    for i, (doc, score) in enumerate(top_docs, 1):
        print(f"   {i}. (similarity: {score:.4f}) {doc[:80]}...")
    
    return top_docs


def generate_answer(query: str, context_docs: List[Tuple[str, float]]) -> str:
    """
    Steps 2-3 of RAG: AUGMENTATION + GENERATION
    
    Take the retrieved documents and add them to the LLM's context.
    Then ask the LLM to answer based on that context.
    
    This is where RAG prevents hallucinations — the LLM can only
    answer using the provided documents.
    
    Args:
        query: User's question
        context_docs: Retrieved documents with similarity scores
    
    Returns:
        LLM's answer grounded in the provided context
    """
    print(f"\n🤖 GENERATION: Asking LLM to answer using retrieved context...")
    
    # Build the context string from retrieved documents
    # We join all the docs together, separated by newlines
    context = "\n\n".join([doc for doc, _ in context_docs])
    
    # The RAG prompt template — this is critical
    # We tell the LLM: "Answer ONLY using this information"
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "If the context doesn't contain the answer, say 'I don't have that information in the provided context.' "
        "Do not make up information. Only use what's explicitly stated in the context."
    )
    
    user_prompt = f"""Context:
{context}

Question: {query}

Answer the question based only on the context above. Be concise and accurate."""
    
    # Call the LLM with the augmented context
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3  # Lower temperature = more factual, less creative
    )
    
    answer = response.choices[0].message.content
    
    print(f"\n✅ ANSWER: {answer}")
    
    return answer


def rag_pipeline(query: str, knowledge_base: List[str], top_k: int = 3) -> str:
    """
    The complete RAG pipeline:
    1. Retrieval: Find relevant docs using embeddings
    2. Augmentation: Add docs to LLM context
    3. Generation: LLM answers using that context
    
    This is the function you'd call in production.
    """
    print(f"\n{'='*80}")
    print(f"RAG PIPELINE START")
    print(f"{'='*80}")
    
    # Step 1: Retrieve
    relevant_docs = retrieve_relevant_docs(query, knowledge_base, top_k)
    
    # Steps 2-3: Augment + Generate
    answer = generate_answer(query, relevant_docs)
    
    print(f"\n{'='*80}")
    print(f"RAG PIPELINE COMPLETE")
    print(f"{'='*80}\n")
    
    return answer


# === COMPARISON: WITH vs WITHOUT RAG ===

def answer_without_rag(query: str) -> str:
    """
    Call the LLM WITHOUT RAG — no context provided.
    This shows what happens when the LLM doesn't have grounding info.
    """
    print(f"\n🤖 WITHOUT RAG: Asking LLM without any context...")
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.3
    )
    
    answer = response.choices[0].message.content
    print(f"   Answer: {answer}")
    
    return answer


# === DEMOS ===

def demo_basic_rag():
    """
    Demo 1: Basic RAG pipeline
    Ask a question that's answered in our knowledge base.
    """
    print("\n" + "="*80)
    print("DEMO 1: Basic RAG — Answer a factual question")
    print("="*80)
    
    query = "What pricing plans does TaskFlow offer and what are the costs?"
    
    # WITH RAG
    print("\n--- WITH RAG (grounded in knowledge base) ---")
    rag_answer = rag_pipeline(query, KNOWLEDGE_BASE, top_k=3)
    
    # WITHOUT RAG
    print("\n--- WITHOUT RAG (LLM guessing) ---")
    no_rag_answer = answer_without_rag(query)
    
    print("\n💡 COMPARISON:")
    print("   WITH RAG: Specific, accurate info from knowledge base")
    print("   WITHOUT RAG: Generic or hallucinated response")


def demo_rag_refuses_hallucination():
    """
    Demo 2: RAG refuses to answer when info isn't in knowledge base.
    This shows how RAG prevents hallucinations.
    """
    print("\n" + "="*80)
    print("DEMO 2: RAG Refusing to Hallucinate")
    print("="*80)
    
    # Ask something NOT in our knowledge base
    query = "What is the CEO's favorite programming language?"
    
    print("\n--- WITH RAG (should refuse to answer) ---")
    rag_answer = rag_pipeline(query, KNOWLEDGE_BASE, top_k=3)
    
    print("\n--- WITHOUT RAG (will hallucinate) ---")
    no_rag_answer = answer_without_rag(query)
    
    print("\n💡 NOTICE:")
    print("   WITH RAG: Says 'I don't have that information'")
    print("   WITHOUT RAG: Makes up an answer")


def demo_multiple_sources():
    """
    Demo 3: RAG synthesizing info from multiple documents.
    The answer requires combining 2-3 different docs.
    """
    print("\n" + "="*80)
    print("DEMO 3: Synthesizing Multiple Sources")
    print("="*80)
    
    query = "Compare the features between Starter and Professional plans"
    
    # This requires info from multiple documents in the knowledge base
    answer = rag_pipeline(query, KNOWLEDGE_BASE, top_k=5)
    
    print("\n💡 NOTICE: Answer combines info from multiple retrieved documents")


def demo_specific_detail():
    """
    Demo 4: RAG finding very specific details.
    Tests if retrieval picks up exact details (like numbers, dates).
    """
    print("\n" + "="*80)
    print("DEMO 4: Finding Specific Details")
    print("="*80)
    
    query = "How many days do I have to request a refund on a monthly plan?"
    
    answer = rag_pipeline(query, KNOWLEDGE_BASE, top_k=2)
    
    print("\n💡 NOTICE: RAG finds exact detail (14 days) from knowledge base")


# === MAIN ===

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DAY 26: RAG Pipeline (Retrieval-Augmented Generation)")
    print("="*80)
    
    # Run all demos
    demo_basic_rag()
    demo_rag_refuses_hallucination()
    demo_multiple_sources()
    demo_specific_detail()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. RAG = Retrieval + Augmentation + Generation
2. Prevents hallucinations by grounding LLM in real documents
3. Enables LLMs to answer questions about private/recent data
4. Critical components:
   - Embedding-based retrieval (Day 25)
   - Context injection (the prompt template)
   - Instruction to only use provided context
5. Tomorrow (Day 27): ChromaDB for efficient vector storage at scale
    """)