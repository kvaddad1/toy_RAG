# toy_RAG
### This is to play and generate TOY RAG system using Cohere and Qdrant API 


# Create project directory
```bash
mkdir rag_project
cd rag_project
```

# Create virtual environment using Python 3
```bash
python3 -m venv venv
```

# Activate the virtual environment
```bash
source venv/bin/activate
pip3 install llama-index-core llama-index-vector-stores-qdrant llama-index-embeddings-cohere llama-index-llms-cohere cohere qdrant-client
```

#if version issues - upgrade pip
```bash
python3 -m pip install --upgrade pip
```

#try re-installing 
```bash
pip3 install llama-index-core llama-index-vector-stores-qdrant llama-index-embeddings-cohere llama-index-llms-cohere cohere qdrant-client
```

#Create account on Cohere, Qdrant and generate api and urls 
- cohere: https://docs.cohere.com/
- qdrant: https://qdrant.tech/

# Your API keys
```python
COHERE_API_KEY = "apikey" #copiable later as well  
QDRANT_URL = "database_cluster_url"   
QDRANT_API_KEY = "apikey"  #one-time cannot see again - so needs to copy
```

### Scripts to play with toy RAG system 

* test_RAG.py: verifies connectivity to Cohere (for embeddings and LLM) and Qdrant (for vector storage) before setting up the main RAG system.

```bash
python test_RAG.py
```
Will see something like: 
```
Starting connection tests...
Testing Cohere connection...
✓ Cohere connection successful
Testing Qdrant connection...
✓ Qdrant connection successful
Testing LlamaIndex components...
✓ LlamaIndex components initialized successfully

All connections tested successfully!
```

* demo_rag_system.py: basic demo file to understand how to use the RAG system. (using Cohere and Qdrant API on a private data -- provided in the code itself as document) - automatic questions and answers 
```bash
python demo_rag_system.py
```

* rag_system.py: basic demo file to understand how to use the RAG system. (using Cohere and Qdrant API on a private data -- provided in the code itself as document) - chatbot style 
```bash
python rag_system.py
```

* personal_rag_system.py: Just replaced with custom (personal recipe of "undrallu" - a south indian dish) data to see how the RAG system works with personal data.
```bash
python personal_rag_system.py
```

* custom_rag_system.py: Which takes file path as input and uses the data in the file to create a RAG system. Yet to test, couple of issues with the code functionality -- mostly need to fine tune parameters to make use of folder-based files (like adjusting temperature, etc.) code still works. 
```bash
python custom_rag_system.py
```

* Course name on linkedin Learning platform: 
- Hands-On AI: RAG using LlamaIndex, Instructor: Harpreet Sahota
