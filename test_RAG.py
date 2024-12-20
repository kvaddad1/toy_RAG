from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from qdrant_client import QdrantClient
import cohere

# Your API keys
COHERE_API_KEY = "apikey" #copiable later as well  
QDRANT_URL = "database_cluster_url"   
QDRANT_API_KEY = "apikey"  #one-time cannot see again - so needs to copy

def test_connections():
    print("Starting connection tests...")
    
    try:
        # Test Cohere
        print("Testing Cohere connection...")
        co = cohere.Client(COHERE_API_KEY)
        # Fixed the embed parameters
        response = co.embed(
            texts=["Test connection"],
            model="embed-english-v3.0",
            input_type="search_document"  # Added this required parameter
        )
        print("✓ Cohere connection successful")
        
        # Test Qdrant
        print("Testing Qdrant connection...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections()
        print("✓ Qdrant connection successful")
        
        # Test LlamaIndex components
        print("Testing LlamaIndex components...")
        embed_model = CohereEmbedding(api_key=COHERE_API_KEY)
        llm = Cohere(api_key=COHERE_API_KEY)
        print("✓ LlamaIndex components initialized successfully")
        
        print("\nAll connections tested successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    test_connections()
