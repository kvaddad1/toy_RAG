from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient
import cohere

# Your API keys
COHERE_API_KEY = "apikey" #copiable later as well  
QDRANT_URL = "database_cluster_url"   
QDRANT_API_KEY = "apikey"  #one-time cannot see again - so needs to copy

class RAGSystem:
    def __init__(self, collection_name="test_collection"):
        # Initialize Cohere embedding
        self.embed_model = CohereEmbedding(
            api_key=COHERE_API_KEY,
            model_name="embed-english-v3.0"
        )
        
        # Initialize LLM with corrected parameters
        self.llm = Cohere(
            api_key=COHERE_API_KEY,
            model="command",
            temperature=0.7,
            max_tokens=512
        )
        
        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=20
        )
        
        # Initialize Qdrant
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        
        # Create vector store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name
        )
        
        # Configure settings
        Settings.chunk_size = 512
        Settings.chunk_overlap = 20
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = self.node_parser
        
        # Initialize index
        self.index = VectorStoreIndex.from_documents(
            [],
            vector_store=self.vector_store
        )
        
        print("RAG system initialized successfully!")

    def add_documents(self, texts):
        """Add documents to the knowledge base."""
        documents = [Document(text=text) for text in texts]
        self.index = VectorStoreIndex.from_documents(
            documents,
            vector_store=self.vector_store
        )
        print(f"Added {len(texts)} documents to the knowledge base")

    def query(self, question: str) -> str:
        """Query the knowledge base."""
        query_engine = self.index.as_query_engine(
            similarity_top_k=2,
            streaming=False
        )
        response = query_engine.query(question)
        return str(response)

def main():
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Example documents
    documents = [
        """RAG (Retrieval-Augmented Generation) is an AI framework that enhances 
        large language models with external knowledge. It works by first retrieving 
        relevant information from a knowledge base, then using this information to 
        generate more accurate and contextual responses.""",
        
        """Vector databases are essential components of RAG systems. They store 
        document embeddings and enable efficient similarity search. When a query 
        comes in, the system finds the most relevant documents by comparing the 
        query's embedding with stored document embeddings.""",
        
        """Cohere provides powerful language models that can be used in RAG systems. 
        Their embedding models convert text into vector representations, while their 
        generation models create human-like responses based on the retrieved context."""
    ]
    
    # Add documents
    rag.add_documents(documents)
    
    # Example queries
    questions = [
        "What is RAG and how does it work?",
        "What role do vector databases play in RAG systems?",
        "How does Cohere fit into RAG systems?"
    ]
    
    # Test queries
    for question in questions:
        print(f"\nQuestion: {question}")
        print(f"Answer: {rag.query(question)}")
        print("-" * 80)

if __name__ == "__main__":
    main()
