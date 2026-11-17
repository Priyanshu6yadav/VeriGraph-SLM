from graph_builder import create_graph, add_triples, get_graph_statistics


# Sample triples for testing graph construction.
# These are illustrative test facts, not verified extractions.
triples = [
    {
        "subject": "Transformer",
        "predicate": "USES",
        "object": "Self-Attention",
        "source": "attention is you need.pdf",
        "page_number": 2,
        "chunk_id": "page_2_chunk_3"
    },
    {
        "subject": "Transformer",
        "predicate": "ELIMINATES",
        "object": "Recurrence",
        "source": "attention is you need.pdf",
        "page_number": 2,
        "chunk_id": "page_2_chunk_3"
    },
    {
        "subject": "Transformer",
        "predicate": "CONTAINS",
        "object": "Encoder",
        "source": "attention is you need.pdf",
        "page_number": 1,
        "chunk_id": "page_1_chunk_1"
    }
]


# Create graph
graph = create_graph()

# Insert triples
graph = add_triples(graph, triples)

# Print statistics
print("Graph Statistics:")
print(get_graph_statistics(graph))


# Display entities
print("\nEntities:")
for node in graph.nodes:
    print(node)


# Display relationships and evidence
print("\nRelationships:")
for source, target, data in graph.edges(data=True):
    print(f"{source} --[{data['relation']}]--> {target}")
    print(
        f"Evidence: {data['source']}, "
        f"Page {data['page_number']}, "
        f"Chunk {data['chunk_id']}"
    )
