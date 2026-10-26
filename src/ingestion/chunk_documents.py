from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Location of our document
DOCUMENT_PATH = Path("data/documents/company_policy.txt")


# Load the document
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")
documents = loader.load()


# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# Split the document into chunks
chunks = text_splitter.split_documents(documents)


# Display information
print("Number of chunks created:", len(chunks))

print("\nFirst chunk:\n")
print(chunks[0].page_content)

print("\nFirst chunk metadata:\n")
print(chunks[0].metadata)