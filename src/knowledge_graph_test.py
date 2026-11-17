
from ingestion_test import process_pdf
from graph_extractor import extract_triples_from_chunk
from graph_storage import save_graph
from graph_visualizer import visualize_graph

from graph_builder import (
    create_graph,
    add_triples,
    get_graph_statistics
)


pdf_path = "data/raw_docs/attention is you need.pdf"

# 1. Process PDF
chunks = process_pdf(pdf_path)

print("Total chunks:", len(chunks))


# 2. Create graph
graph = create_graph()

all_triples = []


# 3. Extract triples from every chunk
for index, chunk in enumerate(chunks):

    print(
        f"Processing chunk {index + 1}/{len(chunks)}: "
        f"{chunk['chunk_id']}"
    )

    triples = extract_triples_from_chunk(chunk)

    all_triples.extend(triples)


# 4. Normalize and insert triples
graph = add_triples(graph, all_triples)
save_graph(graph)
visualize_graph(graph)

# 5. Display statistics
print("\nKnowledge Graph Statistics:")
print(get_graph_statistics(graph))

print("\nRaw extracted triples:", len(all_triples))


# 6. Display relationships with evidence
print("\nNormalized Relationships:")

for subject, object_, key, data in graph.edges(
    keys=True,
    data=True
):

    print(
        f"\n{subject} --[{data['relation']}]--> {object_}"
    )

    print("Evidence count:", len(data["evidence"]))

    for evidence in data["evidence"][:3]:

        print(
            f"  Source: {evidence['source']} | "
            f"Page: {evidence['page_number']} | "
            f"Chunk: {evidence['chunk_id']}"
        )


# 7. Show a relationship with multiple evidence records
print("\nExample consolidated relationship:")

for subject, object_, key, data in graph.edges(
    keys=True,
    data=True
):

    if len(data["evidence"]) > 1:

        print(
            f"{subject} --[{data['relation']}]--> {object_}"
        )

        print(
            "Number of supporting evidence records:",
            len(data["evidence"])
        )

        break
