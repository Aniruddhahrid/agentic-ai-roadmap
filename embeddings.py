"""
Day 25: Embeddings and Semantic Similarity

Embeddings = converting text into vectors (lists of numbers)
Semantic similarity = finding text with similar meaning (even if different words)

This is the foundation of:
- RAG (Retrieval-Augmented Generation) — Day 26
- Vector databases — Day 27
- Long-term memory for agents
- Semantic search

Why embeddings?
- LLMs can't hold infinite context
- We need to find relevant info from large datasets
- Keyword search misses semantically similar content
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from typing import List, Tuple
import json

load_dotenv()

# We're still using Ollama, but now for embeddings instead of chat
# Ollama can run embedding models (like nomic-embed-text) locally
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Embedding model — smaller and faster than chat models
# This model converts text → 768-dimensional vectors
EMBEDDING_MODEL = "nomic-embed-text"

# === CORE EMBEDDING FUNCTIONS ===

def get_embedding(text: str) -> List[float]:
    """
    Convert text into a vector (embedding).
    
    What happens inside:
    1. Text is tokenized (broken into pieces)
    2. Each token is converted to numbers
    3. A neural network processes these numbers
    4. Output: a 768-dimensional vector that captures the "meaning"
    
    Why 768 dimensions? That's what this model produces. Different models
    produce different sizes (OpenAI's ada-002 is 1536 dimensions).
    
    The vector's values encode semantic meaning — similar texts have
    similar vectors, measured by cosine similarity.
    """
    # Call the embedding endpoint — different from chat endpoint
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    
    # Extract the embedding vector from the response
    # It's a list of 768 floating-point numbers
    embedding = response.data[0].embedding
    
    return embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Measure how similar two vectors are (0 to 1).
    
    Math explanation:
    - Two identical vectors = similarity of 1.0
    - Two completely different vectors = similarity of 0.0
    - Vectors pointing in opposite directions = similarity of -1.0
      (rare with embeddings, usually between 0-1)
    
    Why cosine? It measures the angle between vectors, not their length.
    This means "dog" and "puppy" have high similarity even if one word
    appears more frequently in the training data.
    
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    Where:
    - A · B = dot product (multiply corresponding elements, sum them)
    - ||A|| = magnitude of vector A (length in n-dimensional space)
    """
    # Convert to numpy arrays for vector operations
    # numpy is the standard library for numerical computing in Python
    a = np.array(vec1)
    b = np.array(vec2)
    
    # Dot product: multiply each pair of elements and sum
    # Example: [1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 = 32
    dot_product = np.dot(a, b)
    
    # Magnitude (length) of each vector
    # np.linalg.norm calculates: sqrt(x1² + x2² + ... + xn²)
    # This is the Euclidean distance from origin to the point
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Cosine similarity = dot product divided by product of magnitudes
    # This normalizes the result to be between -1 and 1
    similarity = dot_product / (norm_a * norm_b)
    
    return float(similarity)


def find_most_similar(
    query: str,
    documents: List[str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Find the top_k most similar documents to the query.
    
    This is the core of semantic search:
    1. Convert query to embedding
    2. Convert all documents to embeddings
    3. Calculate similarity between query and each document
    4. Return top_k most similar
    
    Args:
        query: The search query (e.g., "how to train a dog")
        documents: List of documents to search through
        top_k: How many results to return
    
    Returns:
        List of (document, similarity_score) tuples, sorted by similarity
    
    Real-world scale: This is a naive implementation. For 1M+ documents,
    you'd use a vector database like ChromaDB (Day 27) or Pinecone.
    """
    print(f"\n🔍 Embedding query: '{query}'...")
    query_embedding = get_embedding(query)
    
    print(f"📄 Embedding {len(documents)} documents...")
    # Calculate similarity with each document
    similarities = []
    for doc in documents:
        doc_embedding = get_embedding(doc)
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return similarities[:top_k]


# === DEMONSTRATION FUNCTIONS ===

def demo_basic_similarity():
    """
    Demo 1: Show that embeddings capture semantic meaning.
    
    "Dog" and "puppy" should be more similar than "dog" and "car",
    even though they're all different words.
    """
    print("\n" + "="*80)
    print("DEMO 1: Basic Semantic Similarity")
    print("="*80)
    
    # These are semantically similar
    text1 = "I love dogs"
    text2 = "Puppies are adorable"
    text3 = "I enjoy driving cars"
    
    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Text 3: '{text3}'")
    
    print("\nGetting embeddings...")
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    emb3 = get_embedding(text3)
    
    print(f"Embedding dimensions: {len(emb1)} (each text becomes a {len(emb1)}-number vector)")
    print(f"First 5 values of Text 1 embedding: {emb1[:5]}")
    
    # Calculate similarities
    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)
    
    print(f"\n📊 Similarity Results:")
    print(f"  '{text1}' vs '{text2}': {sim_1_2:.4f}")
    print(f"  '{text1}' vs '{text3}': {sim_1_3:.4f}")
    
    if sim_1_2 > sim_1_3:
        print(f"\n✅ Correct! Dog/puppy similarity ({sim_1_2:.4f}) > Dog/car similarity ({sim_1_3:.4f})")
    else:
        print(f"\n⚠️  Unexpected: Car seems more related than puppy")


def demo_semantic_search():
    """
    Demo 2: Search through product reviews using semantic similarity.
    
    This shows how embeddings enable "fuzzy" search — finding relevant
    content even when exact keywords don't match.
    """
    print("\n" + "="*80)
    print("DEMO 2: Semantic Search Through Product Reviews")
    print("="*80)
    
    # Fake product reviews dataset
    # Notice: different ways of saying similar things
    reviews = [
        "The battery life is terrible, only lasts 2 hours",
        "Great screen quality and vibrant colors",
        "Setup was confusing, took me an hour to figure out",
        "Charges very slowly, takes forever to reach 100%",
        "The display is absolutely stunning, best I've ever seen",
        "Instructions were unclear and hard to follow",
        "Camera quality is amazing in low light",
        "Battery drains too quickly during video calls",
        "User manual was poorly written",
        "The screen is too dim even at max brightness"
    ]
    
    # User question (no exact keyword matches for some reviews)
    query = "problems with display performance"
    
    print(f"\nSearching through {len(reviews)} reviews...")
    print(f"Query: '{query}'")
    
    # Find top 3 most relevant reviews
    results = find_most_similar(query, reviews, top_k=3)
    
    print(f"\n🎯 Top 3 Most Relevant Reviews:")
    for i, (review, score) in enumerate(results, 1):
        print(f"\n{i}. Similarity: {score:.4f}")
        print(f"   Review: '{review}'")
    
    print("\n💡 Notice:")
    print("   - Found 'battery drains quickly' even though query said 'battery performance'")
    print("   - Found 'charges slowly' because it's battery-related")
    print("   - Semantic similarity found meaning, not just keywords")


def demo_clustering():
    """
    Demo 3: Group similar documents together (clustering).
    
    This shows how embeddings can categorize documents without
    explicit labels — just based on semantic similarity.
    """
    print("\n" + "="*80)
    print("DEMO 3: Document Clustering by Similarity")
    print("="*80)
    
    # Mix of topics: animals, technology, food
    documents = [
        "Lions are powerful predators in the savanna",
        "The new smartphone has excellent camera features",
        "Elephants have incredible memory and intelligence",
        "This laptop is perfect for video editing work",
        "Pizza is one of the most popular Italian dishes",
        "Wolves hunt in coordinated packs",
        "Pasta carbonara is a classic Roman recipe",
        "The tablet has a long-lasting battery",
        "Sushi is a traditional Japanese cuisine"
    ]
    
    print(f"\nDocuments to cluster:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc}")
    
    # Pick one document from each category
    anchors = {
        "Animals": "Lions are powerful predators in the savanna",
        "Technology": "The new smartphone has excellent camera features",
        "Food": "Pizza is one of the most popular Italian dishes"
    }
    
    print(f"\n🏷️  Clustering around 3 anchor documents...")
    
    for category, anchor in anchors.items():
        print(f"\n--- {category} cluster (anchor: '{anchor[:50]}...') ---")
        results = find_most_similar(anchor, documents, top_k=3)
        
        for doc, score in results:
            if doc == anchor:
                continue  # Skip the anchor itself
            print(f"  • {doc} (similarity: {score:.4f})")


def demo_multilingual():
    """
    Demo 4: Cross-language similarity (if model supports it).
    
    Some embedding models can understand that "Hello" in English
    is similar to "Hola" in Spanish because they mean the same thing.
    
    Note: nomic-embed-text has some multilingual capability but isn't
    specifically trained for it. For production multilingual search,
    use models like "multilingual-e5" or OpenAI's multilingual embeddings.
    """
    print("\n" + "="*80)
    print("DEMO 4: Cross-Language Similarity (Limited)")
    print("="*80)
    
    # Same concept in different languages
    texts = {
        "English": "Good morning, how are you?",
        "Spanish": "Buenos días, ¿cómo estás?",
        "French": "Bonjour, comment allez-vous?",
        "Random": "The capital of France is Paris"
    }
    
    print("\nTexts to compare:")
    for lang, text in texts.items():
        print(f"  {lang}: '{text}'")
    
    english_emb = get_embedding(texts["English"])
    
    print(f"\nComparing all texts to English greeting:")
    for lang, text in texts.items():
        if lang == "English":
            continue
        emb = get_embedding(text)
        sim = cosine_similarity(english_emb, emb)
        print(f"  {lang}: {sim:.4f}")
    
    print("\n💡 Note: Spanish/French greetings should score higher than unrelated text")
    print("   But multilingual similarity won't be perfect with this model")


# === PRACTICAL APPLICATION ===

def demo_practical_rag_preview():
    """
    Demo 5: Preview of RAG (Retrieval-Augmented Generation).
    
    This is what you'll build in Day 26 — using embeddings to find
    relevant context, then feeding that context to an LLM.
    
    Flow:
    1. User asks a question
    2. Find relevant documents using embeddings
    3. Send question + relevant docs to LLM
    4. LLM answers using that context
    """
    print("\n" + "="*80)
    print("DEMO 5: RAG Preview (What's Coming in Day 26)")
    print("="*80)
    
    # Knowledge base about a fictional company
    knowledge_base = [
        "Our company was founded in 2020 by Jane Smith and John Doe in San Francisco.",
        "We offer three subscription tiers: Basic ($10/month), Pro ($25/month), and Enterprise (custom pricing).",
        "Customer support is available Monday-Friday 9am-5pm PST via email and chat.",
        "Our refund policy allows cancellations within 30 days for a full refund.",
        "We have offices in San Francisco, New York, London, and Tokyo.",
        "The Pro plan includes priority support and advanced analytics features.",
        "Enterprise customers get dedicated account managers and custom integrations.",
        "We process all payments securely through Stripe and support major credit cards.",
        "Our platform is SOC 2 Type II certified and GDPR compliant.",
        "Beta features are released first to Pro and Enterprise users."
    ]
    
    # User question
    question = "What support options are available and when?"
    
    print(f"Knowledge base: {len(knowledge_base)} documents")
    print(f"User question: '{question}'")
    
    # Step 1: Find relevant documents
    print(f"\n🔍 Finding relevant context...")
    relevant_docs = find_most_similar(question, knowledge_base, top_k=2)
    
    print(f"\n📄 Top 2 relevant documents:")
    for doc, score in relevant_docs:
        print(f"  • (score: {score:.4f}) {doc}")
    
    # Step 2: Create context for LLM
    context = "\n".join([doc for doc, _ in relevant_docs])
    
    print(f"\n🤖 This context would now be sent to the LLM along with the question.")
    print(f"    The LLM would answer based on this specific information.")
    print(f"\n    This is RAG (Retrieval-Augmented Generation) — coming tomorrow!")


# === MAIN ===

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DAY 25: Embeddings and Semantic Similarity")
    print("="*80)
    
    print("\n⚠️  SETUP CHECK:")
    print("Before running, make sure you have the nomic-embed-text model.")
    print("Run this in your terminal if you haven't already:")
    print("  ollama pull nomic-embed-text")
    print("\nThis model is ~274MB and optimized for semantic search.")
    input("\nPress Enter when ready to continue...")
    
    # Run all demos
    demo_basic_similarity()
    demo_semantic_search()
    demo_clustering()
    demo_multilingual()
    demo_practical_rag_preview()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Embeddings convert text → vectors that capture semantic meaning
2. Cosine similarity measures how close two vectors are (0 = different, 1 = same)
3. This enables "fuzzy search" — finding relevant content without exact keywords
4. Applications: search, clustering, deduplication, recommendation, RAG
5. Tomorrow (Day 26): Build a full RAG pipeline using these embeddings
    """)