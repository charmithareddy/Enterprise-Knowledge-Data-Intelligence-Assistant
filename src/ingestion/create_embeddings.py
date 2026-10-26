from pathlib import Path

from fastembed import TextEmbedding
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Location of our document
DOCUMENT_PATH = Path("data/documents/company_policy.txt")


# Load the document
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()


# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)


# Create the embedding model
embedding_model = TextEmbedding()


# Create embeddings for all chunks
texts = [chunk.page_content for chunk in chunks]

vectors = list(embedding_model.embed(texts))


# Display information
print("Number of chunks:", len(chunks))

print("Number of embeddings:", len(vectors))

print("Vector dimensions:", len(vectors[0]))


# Display first chunk and its embedding
print("\nFirst chunk:")
print(chunks[0].page_content)

print("\nFirst 10 values of first embedding:")
print(vectors[0][:10])