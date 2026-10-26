from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the document

DOCUMENT_PATH = Path("data/documents/company_policy.txt")

loader = TextLoader(
str(DOCUMENT_PATH),
encoding="utf-8"
)

documents = loader.load()

# 2. Split the document into chunks

text_splitter = RecursiveCharacterTextSplitter(
chunk_size=500,
chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

# 3. Create the embedding model

embedding_model = TextEmbedding()

# 4. Generate embeddings for all chunks

chunk_texts = [chunk.page_content for chunk in chunks]

chunk_vectors = np.array(
list(embedding_model.embed(chunk_texts))
)

# 5. Enter the user's question

query = "Explain the company's refund and customer support policies."

# 6. Generate an embedding for the question

query_vector = np.array(
list(embedding_model.embed([query]))[0]
)

# 7. Calculate similarity scores

similarities = np.dot(
chunk_vectors,
query_vector
)

# 8. Find the top 3 most relevant chunks

top_k = min(3, len(chunks))

top_indices = np.argsort(similarities)[::-1][:top_k]

# 9. Display the retrieved results

print("User question:")
print(query)

print("\nTop", top_k, "relevant chunks:")
for rank, index in enumerate(top_indices, start=1):
    print(f"\n--- Result {rank} ---")
    print("Similarity score:", round(float(similarities[index]), 4))
    print("Source:", chunks[index].metadata["source"])
    print("Content:")
    print(chunks[index].page_content)
    print("-" * 50)
