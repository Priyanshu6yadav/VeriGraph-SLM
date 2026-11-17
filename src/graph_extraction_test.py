
from ingestion_test import process_pdf
from graph_extractor import extract_triples


pdf_path = "data/raw_docs/attention is you need.pdf"
chunks = process_pdf(pdf_path)

selected_chunk = next(
    (
        chunk for chunk in chunks
        if chunk["chunk_id"] == "page_2_chunk_3"
    ),
    None
)

if selected_chunk is None:
    raise ValueError("Selected chunk not found")


# Extract relationships
triples = extract_triples(selected_chunk["text"])


# Attach original document evidence
for triple in triples:
    triple["source"] = selected_chunk.get(
        "source", "Unknown"
    )
    triple["page_number"] = selected_chunk["page_number"]
    triple["chunk_id"] = selected_chunk["chunk_id"]


# Display evidence-backed triples
print("\nEvidence-backed triples:")

for triple in triples:
    print("\nTriple:")
    print(
        f"{triple['subject']} "
        f"--[{triple['predicate']}]--> "
        f"{triple['object']}"
    )

    print("Source:", triple["source"])
    print("Page:", triple["page_number"])
    print("Chunk:", triple["chunk_id"])
