"""RAG System for Real Estate Market Knowledge Base

This module implements a Retrieval-Augmented Generation (RAG) system using
Chroma vector database for semantic search over market knowledge. The RAG
system enables the agent to retrieve relevant market context for property
analysis.

Key Classes:
    - RealEstateKnowledgeBase: Core RAG with semantic search
    - RealEstateRAG: Backward-compatible wrapper

The system stores:
    1. Market trends (neighborhood conditions, appreciation trends)
    2. Comparable properties (recent sales with prices and features)
    3. Regulations (local real estate rules)

All documents are embedded and indexed for semantic similarity search,
enabling the agent to find relevant context without exact keyword matches.

Example:
    >>> rag = RealEstateRAG()
    >>> initialize_sample_market_data(rag)
    >>> insights = rag.retrieve_market_insights(\"Northridge property market\")
\"\"\"

import chromadb
from typing import List, Dict, Any, Optional


class RealEstateKnowledgeBase:
    """Stores and retrieves real estate market information using Chroma.
    
    Core RAG implementation for semantic search over market knowledge.
    Supports storing and retrieving:
    - Market trends (neighborhood conditions, appreciation)
    - Comparable properties (recent sales)
    - Regulations (local real estate rules)
    
    Uses cosine similarity for vector comparisons.
    """
    
    def __init__(self, db_path: str = "./chroma_db"):
        """Initialize knowledge base with persistent storage.
        
        Args:
            db_path: Directory path for Chroma database persistence.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create/get collection with cosine similarity
        self.market_data = self.client.get_or_create_collection(
            name="market_data",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.is_initialized = False
    
    def add_data(self, items: List[Dict[str, Any]]):
        """Add items to knowledge base for semantic search indexing.
        
        Adds items to the Chroma collection. Documents are automatically
        embedded and indexed for semantic similarity search.
        
        Args:
            items: List of dicts, each with:
                - id: Unique identifier for the item
                - text: The content to index
                - metadata: Optional dict with tags/filtering info
                
        Example:
            >>> items = [{
            ...     "id": "market_1",
            ...     "text": "Northridge shows 8% yearly appreciation",
            ...     "metadata": {"type": "trend", "neighborhood": "Northridge"}
            ... }]
            >>> kb.add_data(items)
        """
        if not items:
            return
        
        ids = [str(item["id"]) for item in items]
        texts = [item["text"] for item in items]
        metadatas = [item.get("metadata", {}) for item in items]
        
        self.market_data.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(items)} items to knowledge base")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant information using semantic similarity.
        
        Performs semantic search on indexed documents. The query is embedded
        and compared against document embeddings using cosine similarity.
        
        Args:
            query: Natural language search query. Should describe what you're
                   looking for (e.g., \"property appreciation in Northridge\")
            top_k: Number of results to return. Default 3.
            
        Returns:
            List of dicts with:
                - id: Document identifier
                - text: The document content
                - metadata: Associated metadata
                - relevance: Similarity score (0-1, higher = more similar)
                
        Example:
            >>> results = kb.search(\"market conditions\", top_k=3)
            >>> for result in results:
            ...     print(f\"{result['text']} (relevance: {result['relevance']})\")
        """
        try:
            results = self.market_data.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Format results nicely
            formatted = []
            if results["ids"] and len(results["ids"]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    formatted.append({
                        "id": doc_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "relevance": round(1 - results["distances"][0][i], 2)
                    })
            
            return formatted
        
        except Exception as e:
            print(f"Search error: {e}")
            return []


# For backward compatibility
class RealEstateRAG:
    """Wrapper for backward compatibility with previous code"""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.kb = RealEstateKnowledgeBase(persist_dir)
    
    def add_market_insights(self, insights: List[Dict]):
        self.kb.add_data(insights)
    
    def add_comparable_properties(self, properties: List[Dict]):
        self.kb.add_data(properties)
    
    def retrieve_market_insights(self, query: str, top_k: int = 3):
        return self.kb.search(query, top_k)
    
    def retrieve_comparable_properties(self, query: str, top_k: int = 3):
        return self.kb.search(query, top_k)
    
    def retrieve_regulations(self, query: str, top_k: int = 3):
        return self.kb.search(query, top_k)


def initialize_sample_market_data(rag: RealEstateRAG):
    """
    Load sample market data into the knowledge base.
    In production, this would connect to real MLS databases.
    """
    
    # Sample market data about neighborhoods
    market_data = [
        {
            "id": "market_1",
            "text": "Northridge is a stable neighborhood. Average home price: $330,000-$360,000. Properties typically 1500-2200 sqft. Good schools and parks nearby.",
            "metadata": {"type": "neighborhood", "name": "Northridge"}
        },
        {
            "id": "market_2",
            "text": "Westside market shows growth with 8% yearly appreciation. Popular with professionals. Average property price per sqft: $185.",
            "metadata": {"type": "trend", "name": "Westside"}
        },
        {
            "id": "market_3",
            "text": "Downtown area experiencing revitalization. New development attracting younger buyers. Condo market strong with 15% recent appreciation.",
            "metadata": {"type": "trend", "name": "Downtown"}
        },
        {
            "id": "comp_1",
            "text": "123 Oak St, Northridge: 2100 sqft, 3BR/2BA, Sold $345,000. Built 2005. Modern updates. Good condition.",
            "metadata": {"type": "comparable", "price": 345000, "sqft": 2100}
        },
        {
            "id": "comp_2",
            "text": "456 Maple Ave, Northridge: 1950 sqft, 3BR/2BA, Sold $335,000. Built 2008. Recently updated kitchen.",
            "metadata": {"type": "comparable", "price": 335000, "sqft": 1950}
        },
        {
            "id": "comp_3",
            "text": "789 Pine Rd, Northridge: 2250 sqft, 4BR/2.5BA, Sold $365,000. Built 2000. Premium corner lot.",
            "metadata": {"type": "comparable", "price": 365000, "sqft": 2250}
        },
        {
            "id": "rule_1",
            "text": "Property taxes in this area: 0.75% of assessed value. Homestead exemption available for primary residences.",
            "metadata": {"type": "regulation", "category": "taxes"}
        },
        {
            "id": "rule_2",
            "text": "HOA fees typically $200-300/month. Includes landscaping, security, and pool access.",
            "metadata": {"type": "regulation", "category": "hoa"}
        }
    ]
    
    # Add all data
    rag.add_market_insights(market_data[:3])
    rag.add_comparable_properties(market_data[3:6])
    rag.retrieve_regulations(market_data[6:])  # This populates regulations
    
    print("✓ Sample market data loaded")


if __name__ == "__main__":
    # Test the system
    rag = RealEstateRAG()
    initialize_sample_market_data(rag)
    
    # Test search
    print("\nSearching for Northridge properties...")
    results = rag.retrieve_market_insights("Northridge prices")
    
    for result in results:
        print(f"\n📄 {result['text'][:80]}...")
        print(f"   Relevance: {result['relevance']}")

