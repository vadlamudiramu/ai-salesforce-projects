
# AI Copilot - RAG Workflow
import openai
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

# Step 1: Initialize embedding model
embeddings = OpenAIEmbeddings()

# Step 2: Connect to Pinecone
pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")
index = pinecone.Index("fitness-copilot")

# Step 3: Query embedding and search
query = "Best exercise routine for weight loss"
query_vector = embeddings.embed_query(query)
results = index.query(vector=query_vector, top_k=5, include_metadata=True)

# Step 4: Print matched results
for match in results["matches"]:
    print(match["metadata"])
