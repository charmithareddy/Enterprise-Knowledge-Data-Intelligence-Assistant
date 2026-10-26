from pathlib import Path
import json
import re
import urllib.request
import urllib.error

import numpy as np
from fastembed import TextEmbedding


# ============================================================
# 1. CONFIGURATION
# ============================================================

DOCUMENT_PATH = Path("data/documents/company_policy.txt")

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

SIMILARITY_THRESHOLD = 0.40
SCORE_MARGIN = 0.15

TOP_K = 3
MAX_SECTION_LENGTH = 900


# ============================================================
# 2. LOAD DOCUMENT
# ============================================================

def load_document():
    print("\nLoading company document...")

    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Document not found: {DOCUMENT_PATH}\n"
            "Run the program from the project root folder."
        )

    text = DOCUMENT_PATH.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("The document is empty.")

    print(f"Loaded: {DOCUMENT_PATH.name}")

    return text


# ============================================================
# 3. POLICY-AWARE CHUNKING
# ============================================================

def build_chunks(text):
    print("\nIdentifying individual policy sections...")

    # Matches headings such as:
    # 1. REFUND POLICY
    # 2. CUSTOMER SUPPORT
    # 3. EMPLOYEE REMOTE WORK POLICY
    # 4. DATA SECURITY
    # 5. ANNUAL LEAVE

    heading_pattern = re.compile(
        r"(?m)^\s*(\d+\.\s+[A-Z][A-Z0-9 &()/-]*)\s*$"
    )

    matches = list(heading_pattern.finditer(text))

    sections = []

    if matches:
        for index, match in enumerate(matches):
            start = match.start()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(text)

            section_text = text[start:end].strip()
            heading = match.group(1).strip()

            if section_text:
                sections.append({
                    "section": heading,
                    "content": section_text
                })

    else:
        # Fallback for documents without numbered headings.
        sections.append({
            "section": "General Document",
            "content": text
        })

    chunks = []

    for section in sections:
        section_name = section["section"]
        content = section["content"]

        # Keep a policy together whenever possible.
        if len(content) <= MAX_SECTION_LENGTH:
            chunks.append({
                "section": section_name,
                "content": content,
                "source": DOCUMENT_PATH.name
            })
            continue

        # Split unusually long sections at paragraph boundaries.
        paragraphs = [
            paragraph.strip()
            for paragraph in content.split("\n\n")
            if paragraph.strip()
        ]

        current_text = ""

        for paragraph in paragraphs:
            candidate = (
                current_text + "\n\n" + paragraph
            ).strip()

            if (
                current_text
                and len(candidate) > MAX_SECTION_LENGTH
            ):
                chunks.append({
                    "section": section_name,
                    "content": current_text,
                    "source": DOCUMENT_PATH.name
                })

                current_text = paragraph

            else:
                current_text = candidate

        if current_text:
            chunks.append({
                "section": section_name,
                "content": current_text,
                "source": DOCUMENT_PATH.name
            })

    if not chunks:
        raise ValueError("No document sections were created.")

    print(f"Policy sections identified: {len(sections)}")
    print(f"Retrieval chunks created: {len(chunks)}")

    for chunk in chunks:
        print(f"  - {chunk['section']}")

    return chunks


# ============================================================
# 4. CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):
    print("\nLoading FastEmbed model...")

    embedding_model = TextEmbedding()

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    vectors = np.asarray(
        list(embedding_model.embed(texts)),
        dtype=np.float32
    )

    # Normalize vectors so dot products represent cosine similarity.
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    vectors = vectors / np.maximum(norms, 1e-12)

    print("Document embeddings created successfully!")

    return embedding_model, vectors


# ============================================================
# 5. RETRIEVE RELEVANT POLICY SECTIONS
# ============================================================

def retrieve_context(
    question,
    chunks,
    embedding_model,
    document_vectors
):
    query_vector = np.asarray(
        list(embedding_model.embed([question]))[0],
        dtype=np.float32
    )

    query_norm = np.linalg.norm(query_vector)

    query_vector = query_vector / max(
        float(query_norm),
        1e-12
    )

    scores = np.dot(
        document_vectors,
        query_vector
    )

    ranked_indices = np.argsort(scores)[::-1]

    best_score = float(scores[ranked_indices[0]])

    # Keep chunks that are sufficiently relevant and close
    # to the best matching score.
    effective_threshold = max(
        SIMILARITY_THRESHOLD,
        best_score - SCORE_MARGIN
    )

    selected_chunks = []

    print("\nSearching policy sections...")

    print(f"Best similarity score: {best_score:.4f}")
    print(
        "Effective relevance threshold: "
        f"{effective_threshold:.4f}"
    )

    for index in ranked_indices:
        score = float(scores[index])

        if score < effective_threshold:
            continue

        selected_chunks.append({
            "section": chunks[index]["section"],
            "source": chunks[index]["source"],
            "content": chunks[index]["content"],
            "score": score
        })

        if len(selected_chunks) >= TOP_K:
            break

    if not selected_chunks:
        return (
            "No sufficiently relevant policy information "
            "was found in the document.",
            []
        )

    context_parts = []

    for chunk in selected_chunks:
        context_parts.append(
            f"POLICY SECTION: {chunk['section']}\n"
            f"SOURCE DOCUMENT: {chunk['source']}\n"
            f"POLICY TEXT:\n{chunk['content']}"
        )

    context = "\n\n" + ("-" * 50) + "\n\n"
    context = context.join(context_parts)

    print(f"Relevant sections selected: {len(selected_chunks)}")

    return context, selected_chunks


# ============================================================
# 6. GENERATE AN EVIDENCE-BASED ANSWER
# ============================================================

def generate_answer(question, context):
    prompt = f"""
You are an enterprise knowledge assistant for
Nexora Technologies.

Answer the user's question using only the supplied
policy sections.

IMPORTANT RULES:

1. Answer the exact question asked.
2. Read all supplied policy sections before answering.
3. Include every relevant rule, condition, limit,
   exception, and requirement found in the context.
4. For questions asking about a policy or its rules,
   provide a complete numbered list or bullet list.
5. Do not omit an important condition just to make
   the answer shorter.
6. Do not include unrelated policies.
7. Do not invent facts or use outside knowledge.
8. If the document does not explicitly provide the
   requested policy information, say that the available
   document does not specify it.
9. Distinguish between general rules and exceptions.
10. Avoid repeating the same information.
11. Use clear, professional language.

Before finalizing your answer, check that you have
included all relevant details from the supplied context.

SUPPLIED POLICY SECTIONS:

{context}

USER QUESTION:

{question}

COMPLETE ANSWER:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    request = urllib.request.Request(
        OLLAMA_API,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=180
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        answer = result.get("response", "").strip()

        if not answer:
            return "The model returned an empty answer."

        return answer

    except urllib.error.HTTPError as error:
        return (
            f"Ollama HTTP error: {error.code}\n"
            "Check whether Ollama and the selected model "
            "are working correctly."
        )

    except urllib.error.URLError as error:
        return (
            "Could not connect to Ollama.\n"
            "Ensure Ollama is running and the model "
            f"'{MODEL_NAME}' is installed.\n"
            f"Details: {error}"
        )

    except TimeoutError:
        return (
            "The model took too long to respond. "
            "Try again after closing unnecessary applications."
        )

    except (json.JSONDecodeError, ValueError) as error:
        return (
            "Could not process the response from Ollama.\n"
            f"Details: {error}"
        )


# ============================================================
# 7. DISPLAY SOURCES
# ============================================================

def display_sources(sources):
    print("\n" + "=" * 60)
    print("RETRIEVED POLICY SECTIONS")
    print("=" * 60)

    if not sources:
        print("No relevant sections were retrieved.")
        return

    for number, source in enumerate(sources, start=1):
        print(f"\n[{number}] {source['section']}")
        print(f"Document: {source['source']}")
        print(f"Similarity score: {source['score']:.4f}")
        print("\nSupporting text:")
        print(source["content"])
        print("-" * 60)


# ============================================================
# 8. MAIN APPLICATION
# ============================================================

def main():
    text = load_document()

    chunks = build_chunks(text)

    embedding_model, document_vectors = create_embeddings(
        chunks
    )

    print("\n" + "=" * 60)
    print("       ENTERPRISE KNOWLEDGE ASSISTANT")
    print("=" * 60)
    print("Retrieval: Policy-aware semantic search")
    print("Embeddings: FastEmbed")
    print("Language model: Qwen2.5 3B")
    print("Runtime: Ollama")
    print("Source: Nexora Technologies company policy")
    print("\nAsk a question about the company policy.")
    print("Type 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYour question: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\n\nAssistant stopped.")
            break

        if question.lower() in {"exit", "quit"}:
            print("\nThank you for using the assistant!")
            break

        if not question:
            print("Please enter a question.")
            continue

        context, sources = retrieve_context(
            question,
            chunks,
            embedding_model,
            document_vectors
        )

        if not sources:
            print("\nNo relevant policy sections were found.")
            continue

        print("\nGenerating answer using the local AI model...")

        answer = generate_answer(
            question,
            context
        )

        print("\n" + "=" * 60)
        print("AI GENERATED ANSWER")
        print("=" * 60)
        print(answer)

        display_sources(sources)


# ============================================================
# 9. START APPLICATION
# ============================================================

if __name__ == "__main__":
    main()