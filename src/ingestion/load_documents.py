from pathlib import Path

from langchain_community.document_loaders import TextLoader


# Location of our document
DOCUMENT_PATH = Path("data/documents/company_policy.txt")


# Load the document
loader = TextLoader(str(DOCUMENT_PATH), encoding="utf-8")

documents = loader.load()


# Display basic information
print("Number of documents loaded:", len(documents))

print("\nDocument content:\n")
print(documents[0].page_content)

print("\nDocument metadata:\n")
print(documents[0].metadata)