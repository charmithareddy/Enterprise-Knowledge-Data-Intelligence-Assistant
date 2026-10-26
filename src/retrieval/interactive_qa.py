from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load the company policy document
document_path = Path("data/documents/company_policy.txt")

loader = TextLoader(
    str(document_path),
    encoding="utf-8"
)

documents = loader.load()


# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)


# 3. Load the embedding model
print("Loading the embedding model...")

embedding_model = TextEmbedding()


# 4. Generate embeddings for all document chunks
chunk_texts = [chunk.page_content for chunk in chunks]

chunk_vectors = np.array(
    list(embedding_model.embed(chunk_texts))
)

print("Document loaded successfully!")
print("Number of chunks:", len(chunks))


# 5. Start interactive question answering
print("\n" + "=" * 55)
print(" NEXORA TECHNOLOGIES - POLICY ASSISTANT")
print("=" * 55)

print("Ask questions about the company policy document.")
print("Type 'exit' to stop.\n")


while True:

    # Get a question from the user
    query = input("Your question: ").strip()

    # Stop when the user types exit
    if query.lower() == "exit":
        print("\nThank you for using the Policy Assistant!")
        break

    # Handle empty questions
    if not query:
        print("Please enter a question.\n")
        continue

    # 6. Convert the question into an embedding
    query_vector = np.array(
        list(embedding_model.embed([query]))[0]
    )

    # 7. Calculate similarity scores
    similarities = np.dot(
        chunk_vectors,
        query_vector
    )

    # 8. Retrieve the two most relevant chunks
    top_k = min(2, len(chunks))

    top_indices = np.argsort(similarities)[::-1][:top_k]

    # 9. Display the retrieved information
    print("\nRelevant information:\n")

    for rank, index in enumerate(top_indices, start=1):

        source = chunks[index].metadata.get(
            "source",
            "Unknown source"
        )

        print(f"Result {rank}")
        print("Source:", source)
        print("Similarity score:", round(
            float(similarities[index]), 4
        ))
        print("Information:")
        print(chunks[index].page_content)
        print("-" * 55)

    print()