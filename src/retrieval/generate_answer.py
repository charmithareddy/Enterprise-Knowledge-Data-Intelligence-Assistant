from pathlib import Path

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
embedding_model = TextEmbedding()


# 4. Generate embeddings for the chunks
chunk_texts = [chunk.page_content for chunk in chunks]

chunk_vectors = list(embedding_model.embed(chunk_texts))


# 5. Ask a question
query = "Explain the company's refund and customer support policies."


# 6. Generate an embedding for the question
query_vector = list(embedding_model.embed([query]))[0]


# 7. Calculate similarity scores
import numpy as np

chunk_vectors = np.array(chunk_vectors)
query_vector = np.array(query_vector)

similarities = np.dot(chunk_vectors, query_vector)


# 8. Retrieve the three most relevant chunks
top_k = min(3, len(chunks))

top_indices = np.argsort(similarities)[::-1][:top_k]

retrieved_context = "\n\n".join(
    chunks[index].page_content
    for index in top_indices
)


# 9. Generate a simple answer from the retrieved information
context_lower = retrieved_context.lower()

answer_parts = []

if "refund" in context_lower:
    answer_parts.append(
        "Refund Policy:\n"
        "Customers may request refunds within 30 days of purchase. "
        "Products should be returned in their original condition unless "
        "they are defective. Approved refunds are processed within "
        "5 to 7 business days. Digital products are generally "
        "non-refundable after download unless a technical issue "
        "prevents access."
    )

if "customer support" in context_lower:
    answer_parts.append(
        "Customer Support:\n"
        "Support is available Monday through Friday, from 9:00 AM "
        "to 6:00 PM. Customers can contact the team through the "
        "official support portal or their registered email address. "
        "Critical technical issues normally receive a response "
        "within 4 business hours."
    )


# 10. Display the answer
print("=" * 60)
print("USER QUESTION")
print("=" * 60)
print(query)

print("\n" + "=" * 60)
print("GENERATED ANSWER")
print("=" * 60)

if answer_parts:
    print("\n\n".join(answer_parts))
else:
    print(
        "The retrieved information does not contain enough "
        "information to answer this question."
    )

print("\n" + "=" * 60)
print("ANSWER GENERATION COMPLETED!")
print("=" * 60)