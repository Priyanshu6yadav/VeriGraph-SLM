import json
import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# 1. Initialize local LLM
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# 2. Define extraction prompt
prompt = ChatPromptTemplate.from_template("""You are an expert knowledge graph extraction system.
Extract factual entities and meaningful relationships from the provided document text.

Predicates can be any clear active relationship in UPPERCASE, for example:
USES, HAS, CONTAINS, STORES, DEFINED_AS, CONSISTS_OF, PERFORMS, OUTPUTS, SUPPORTS, REQUIRES, ELIMINATES, REPLACES.

Rules:
1. Extract relationships explicitly stated or directly described in the text.
2. Subject and Object must be concise entities, concepts, or technical terms.
3. Predicate must be in UPPERCASE with underscores instead of spaces (e.g. DEFINED_AS, CONTAINS, USES).
4. Extract 1 to 5 high-quality relational triples per chunk if the text describes, defines, or connects concepts.
5. Return ONLY a valid JSON object with a "triples" array. Do not include markdown or explanations.

Required JSON format:
{{
    "triples": [
        {{
            "subject": "ConceptA",
            "predicate": "RELATION",
            "object": "ConceptB"
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

    # Robust JSON extraction
    if "```" in response_text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        if match:
            response_text = match.group(1).strip()
    elif "{" in response_text and "}" in response_text:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        response_text = response_text[start:end].strip()

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
