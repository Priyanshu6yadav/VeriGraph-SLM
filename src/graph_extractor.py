import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# 1. Initialize local LLM
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# 2. Define extraction prompt
prompt = ChatPromptTemplate.from_template("""
You are an information extraction system.

Extract factual entities and relationships from the
provided document text.

Rules:
1. Extract only information explicitly supported by the text.
2. Do not invent entities or relationships.
3. Keep entity names concise and meaningful.
4. Use short uppercase relationship names.
5. Return only valid JSON.
6. If no relationships are found, return an empty list.

Required JSON format:

{{
    "triples": [
        {{
            "subject": "Transformer",
            "predicate": "USES",
            "object": "Attention Mechanism"
        }}
    ]
}}

Document text:
{text}
""")


# 3. Create extraction chain
extraction_chain = prompt | llm


# 4. Extract triples from one chunk
def extract_triples(text):

    response = extraction_chain.invoke({
        "text": text
    })

    response_text = response.content.strip()

    # Remove Markdown code fences if the model adds them
    if response_text.startswith("```"):
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        response_text = response_text.strip()

    try:
        result = json.loads(response_text)
        triples = result.get("triples", [])

        if not isinstance(triples, list):
            return []

        # Validate each triple
        valid_triples = []

        for triple in triples:
            if not isinstance(triple, dict):
                continue

            subject = triple.get("subject")
            predicate = triple.get("predicate")
            object_ = triple.get("object")

            if not all(
                isinstance(value, str) and value.strip()
                for value in [subject, predicate, object_]
            ):
                continue

            valid_triples.append({
                "subject": subject.strip(),
                "predicate": predicate.strip().upper(),
                "object": object_.strip()
            })

        return valid_triples

    except (json.JSONDecodeError, AttributeError, TypeError):
        print("Could not parse LLM response:")
        print(response_text)
        return []


def extract_triples_from_chunk(chunk):

    # Extract relationships from chunk text
    triples = extract_triples(chunk["text"])

    # Attach document evidence to each relationship
    evidence_triples = []

    for triple in triples:

        triple_with_evidence = {
            **triple,
            "source": chunk["source"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "evidence_text": chunk["text"]
        }

        evidence_triples.append(triple_with_evidence)

    return evidence_triples
