"""
Day 27: ChromaDB — Local Vector Database

ChromaDB = vector database for embeddings
- Stores embeddings persistently (survives restarts)
- Fast similarity search with indexing
- No need to re-embed documents every time

This is what you use when you have 1000+ documents in production RAG.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import time

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

CHAT_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text"

# === CHROMADB SETUP ===

# Initialize ChromaDB client
# persist_directory = where the database is saved on disk
# This means your embeddings survive program restarts
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"  # Creates a folder called chroma_db in current directory
)

# ChromaDB uses "collections" — like tables in SQL
# Each collection holds documents + their embeddings
COLLECTION_NAME = "company_docs"


def get_embedding(text: str) -> List[float]:
    """Get embedding for text — same as Days 25-26."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


# === KNOWLEDGE BASE (same from Day 26) ===

KNOWLEDGE_BASE = [
    "TechStart Inc. was founded in January 2020 by Dr. Sarah Chen and Michael Rodriguez in Austin, Texas. The company specializes in AI-powered productivity tools for remote teams.",
    "Our flagship product, TaskFlow, uses machine learning to automatically prioritize tasks based on deadlines, team capacity, and project dependencies. It integrates with Slack, Jira, and GitHub.",
    "TaskFlow offers three pricing tiers: Starter ($15/user/month, up to 10 users), Professional ($30/user/month, unlimited users, priority support), and Enterprise (custom pricing, includes dedicated success manager and custom integrations).",
    "Customer support is available via email (support@techstart.io) Monday-Friday 8am-6pm CST. Professional plan users also get 24/7 live chat support. Enterprise customers have a dedicated Slack channel with our engineering team.",
    "We offer a 60-day money-back guarantee on annual plans and 14-day guarantee on monthly plans. Refunds are processed within 5-7 business days to the original payment method.",
    "Starter plan includes: basic task management, calendar integration, mobile apps (iOS/Android), and 5GB file storage per user. Limited to 3 projects maximum.",
    "Professional plan includes everything in Starter plus: unlimited projects, advanced analytics, custom workflows, API access, 50GB storage per user, and priority email/chat support.",
    "Enterprise plan includes everything in Professional plus: SSO/SAML authentication, custom integrations, dedicated account manager, SLA guarantees, audit logs, and unlimited storage.",
    "TaskFlow is built on a modern tech stack: React frontend, Python FastAPI backend, PostgreSQL database, Redis for caching, and deployed on AWS with 99.9% uptime SLA.",
    "We are SOC 2 Type II certified, GDPR compliant, and encrypt all data in transit (TLS 1.3) and at rest (AES-256). We conduct annual third-party security audits and penetration testing.",
]


# === CHROMADB FUNCTIONS ===

def create_collection(force_recreate: bool = False):
    """
    Create or get the ChromaDB collection.
    
    Args:
        force_recreate: If True, deletes existing collection and creates new one.
                       Useful for testing or when knowledge base changes.
    
    A collection stores:
    - documents: the actual text
    - embeddings: the vector representations
    - metadata: optional extra info (tags, dates, etc.)
    - ids: unique identifiers for each document
    """
    if force_recreate:
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            print(f"🗑️  Deleted existing collection: {COLLECTION_NAME}")
        except:
            pass  # Collection doesn't exist yet
    
    # get_or_create_collection = creates if doesn't exist, returns if exists
    # This makes the code idempotent — safe to run multiple times
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Company knowledge base for RAG"}
    )
    
    print(f"📦 Collection ready: {COLLECTION_NAME}")
    print(f"   Documents in collection: {collection.count()}")
    
    return collection


def populate_collection(collection, documents: List[str]):
    """
    Add documents to ChromaDB collection.
    
    This is the "indexing" phase — we embed all documents once
    and store them. Future searches are fast because we don't
    need to re-embed.
    
    ChromaDB automatically:
    - Generates embeddings (if we provide an embedding function)
    - OR we can provide pre-computed embeddings
    
    We'll provide pre-computed embeddings to use our Ollama model.
    """
    if collection.count() > 0:
        print(f"⏭️  Collection already has {collection.count()} documents, skipping population.")
        return
    
    print(f"\n📝 Populating collection with {len(documents)} documents...")
    print("   This might take a moment — embedding all documents...")
    
    start_time = time.time()
    
    # Generate embeddings for all documents
    embeddings = []
    for i, doc in enumerate(documents, 1):
        print(f"   Embedding document {i}/{len(documents)}...", end='\r')
        emb = get_embedding(doc)
        embeddings.append(emb)
    
    print(f"\n   ✅ Generated {len(embeddings)} embeddings")
    
    # Add to ChromaDB
    # ids must be unique strings — we use doc_0, doc_1, etc.
    collection.add(
        documents=documents,  # The actual text
        embeddings=embeddings,  # The vector representations
        ids=[f"doc_{i}" for i in range(len(documents))],  # Unique IDs
        metadatas=[{"source": "knowledge_base", "index": i} for i in range(len(documents))]
    )
    
    elapsed = time.time() - start_time
    print(f"   ✅ Populated collection in {elapsed:.2f} seconds")
    print(f"   Collection now has {collection.count()} documents\n")


def search_collection(collection, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Search the collection using semantic similarity.
    
    This is MUCH faster than Day 26's manual approach:
    - Day 26: Embed query, embed all docs, calculate all similarities
    - Day 27: Embed query, use ChromaDB's index for fast lookup
    
    ChromaDB uses ANN (Approximate Nearest Neighbors) algorithms
    for sub-linear search time even with millions of vectors.
    
    Returns:
        List of dicts with 'document', 'distance', 'metadata', 'id'
    """
    print(f"\n🔍 Searching collection...")
    print(f"   Query: '{query}'")
    
    start_time = time.time()
    
    # Embed the query
    query_embedding = get_embedding(query)
    
    # Search ChromaDB
    # n_results = how many results to return (our top_k)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    elapsed = time.time() - start_time
    
    # ChromaDB returns results in a specific format — we'll convert to simpler format
    # results structure:
    # {
    #   'documents': [[doc1, doc2, doc3]],
    #   'distances': [[dist1, dist2, dist3]],
    #   'metadatas': [[meta1, meta2, meta3]],
    #   'ids': [[id1, id2, id3]]
    # }
    
    formatted_results = []
    for i in range(len(results['documents'][0])):
        formatted_results.append({
            'document': results['documents'][0][i],
            'distance': results['distances'][0][i],  # Lower = more similar (opposite of similarity score)
            'metadata': results['metadatas'][0][i],
            'id': results['ids'][0][i]
        })
    
    print(f"   ✅ Found {len(formatted_results)} results in {elapsed:.4f} seconds")
    
    for i, result in enumerate(formatted_results, 1):
        print(f"\n   {i}. (distance: {result['distance']:.4f}) {result['document'][:80]}...")
    
    return formatted_results


def rag_with_chromadb(collection, query: str, top_k: int = 3) -> str:
    """
    Full RAG pipeline using ChromaDB.
    
    This is the production-ready version of Day 26's RAG.
    """
    print(f"\n{'='*80}")
    print(f"RAG WITH CHROMADB")
    print(f"{'='*80}")
    
    # Step 1: Retrieve from ChromaDB
    results = search_collection(collection, query, top_k)
    
    # Step 2: Build context
    context = "\n\n".join([r['document'] for r in results])
    
    # Step 3: Generate answer
    print(f"\n🤖 Generating answer...")
    
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "If the context doesn't contain the answer, say 'I don't have that information.' "
        "Do not make up information."
    )
    
    user_prompt = f"""Context:
{context}

Question: {query}

Answer based only on the context above."""
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    answer = response.choices[0].message.content
    
    print(f"\n✅ ANSWER: {answer}\n")
    
    return answer


# === DEMOS ===

def demo_basic_chromadb():
    """
    Demo 1: Create collection, populate it, search it.
    Shows the basic ChromaDB workflow.
    """
    print("\n" + "="*80)
    print("DEMO 1: Basic ChromaDB Workflow")
    print("="*80)
    
    # Create collection (or get existing)
    collection = create_collection(force_recreate=True)
    
    # Populate with knowledge base
    populate_collection(collection, KNOWLEDGE_BASE)
    
    # Search
    query = "What are the pricing plans?"
    results = search_collection(collection, query, top_k=3)


def demo_persistence():
    """
    Demo 2: Show that ChromaDB persists to disk.
    
    Run this demo twice — second time it won't re-embed documents
    because they're already stored in ./chroma_db folder.
    """
    print("\n" + "="*80)
    print("DEMO 2: Persistence Across Runs")
    print("="*80)
    
    collection = create_collection(force_recreate=False)  # Don't recreate
    populate_collection(collection, KNOWLEDGE_BASE)
    
    print("\n💡 NOTICE:")
    if collection.count() == len(KNOWLEDGE_BASE):
        print("   Collection already existed! Data persisted from previous run.")
        print("   Check ./chroma_db folder — that's where data is stored.")
    else:
        print("   First run — created new collection.")
        print("   Run this script again to see persistence in action.")


def demo_full_rag():
    """
    Demo 3: Full RAG pipeline with ChromaDB.
    """
    print("\n" + "="*80)
    print("DEMO 3: Full RAG Pipeline")
    print("="*80)
    
    collection = create_collection(force_recreate=False)
    populate_collection(collection, KNOWLEDGE_BASE)
    
    # Test multiple queries
    queries = [
        "What support options are available?",
        "Compare Starter and Professional plans",
        "What security certifications does the company have?"
    ]
    
    for query in queries:
        answer = rag_with_chromadb(collection, query, top_k=3)
        print("\n" + "-"*80 + "\n")


def demo_metadata_filtering():
    """
    Demo 4: Using metadata to filter search results.
    
    ChromaDB supports metadata filtering — useful for:
    - Multi-tenant apps (filter by user_id)
    - Date ranges (only recent docs)
    - Document types (only PDFs, only emails)
    """
    print("\n" + "="*80)
    print("DEMO 4: Metadata Filtering")
    print("="*80)
    
    # Create collection with richer metadata
    collection = create_collection(force_recreate=True)
    
    # Add documents with category metadata
    docs_with_categories = [
        ("TaskFlow offers three pricing tiers: Starter, Professional, and Enterprise.", "pricing"),
        ("Customer support is available via email Monday-Friday 8am-6pm CST.", "support"),
        ("We are SOC 2 Type II certified and GDPR compliant.", "security"),
        ("Starter plan includes basic task management and mobile apps.", "pricing"),
        ("Professional plan includes priority support and analytics.", "pricing"),
    ]
    
    documents = [doc for doc, _ in docs_with_categories]
    categories = [cat for _, cat in docs_with_categories]
    
    embeddings = [get_embedding(doc) for doc in documents]
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"category": cat} for cat in categories]
    )
    
    print(f"\n📦 Added {len(documents)} documents with categories")
    
    # Search with metadata filter — only "pricing" documents
    query = "What plans are available?"
    query_embedding = get_embedding(query)
    
    print(f"\n🔍 Searching only 'pricing' category...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"category": "pricing"}  # Metadata filter
    )
    
    print(f"\n   Results (filtered to pricing only):")
    for i, doc in enumerate(results['documents'][0], 1):
        print(f"   {i}. {doc}")
    
    print("\n💡 NOTICE: Only pricing-related docs returned, even though")
    print("   the query could match support or security docs too.")


# === MAIN ===

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DAY 27: ChromaDB — Local Vector Database")
    print("="*80)
    
    demo_basic_chromadb()
    demo_persistence()
    demo_full_rag()
    demo_metadata_filtering()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. ChromaDB stores embeddings persistently (./chroma_db folder)
2. No need to re-embed documents every time — just query the collection
3. Fast similarity search using ANN (Approximate Nearest Neighbors)
4. Supports metadata filtering for multi-tenant or filtered search
5. Production RAG uses vector DBs, not manual similarity calculations
6. Tomorrow (Day 28): Combine everything — agent + tools + RAG + memory
    """)