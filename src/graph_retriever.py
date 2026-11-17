
def find_matching_entities(graph, query: str):
    """
    Find matching entities while removing duplicate
    capitalization variants and shorter overlapping names.
    """

    query_lower = query.casefold()

    matches = []

    for entity in graph.nodes():

        entity_name = str(entity).strip()
        normalized_name = entity_name.casefold()

        if normalized_name and normalized_name in query_lower:

            matches.append((entity, normalized_name))

    # Remove duplicate names that differ only by capitalization
    unique_matches = {}

    for entity, normalized_name in matches:

        if normalized_name not in unique_matches:
            unique_matches[normalized_name] = entity

    normalized_names = list(unique_matches.keys())

    # Remove shorter names contained in longer matched names
    filtered_matches = []

    for name in normalized_names:

        is_overlapping = any(
            name != other and name in other
            for other in normalized_names
        )

        if not is_overlapping:
            filtered_matches.append(unique_matches[name])

    # Prefer longer, more specific entity names
    filtered_matches.sort(
        key=lambda entity: len(str(entity)),
        reverse=True
    )

    return filtered_matches



def retrieve_from_graph(
    graph,
    query: str,
    max_entities: int = 5
):
    """
    Retrieve relationships and supporting evidence
    connected to entities found in the question.
    """

    matched_entities = find_matching_entities(
        graph,
        query
    )

    matched_entities = matched_entities[:max_entities]

    results = []

    visited_edges = set()

    for entity in matched_entities:

        # Retrieve outgoing relationships
        outgoing_edges = graph.out_edges(
            entity,
            keys=True,
            data=True
        )

        # Retrieve incoming relationships
        incoming_edges = graph.in_edges(
            entity,
            keys=True,
            data=True
        )

        all_edges = list(outgoing_edges) + list(incoming_edges)

        for subject, object_, key, data in all_edges:

            edge_id = (subject, object_, key)

            # Avoid processing the same edge twice
            if edge_id in visited_edges:
                continue

            visited_edges.add(edge_id)

            evidence_records = data.get("evidence", [])

            relationship_result = {
                "subject": subject,
                "relation": data.get("relation", "UNKNOWN"),
                "object": object_,
                "evidence": evidence_records
            }

            results.append(relationship_result)

    return {
        "query": query,
        "matched_entities": matched_entities,
        "relationships": results
    }


def print_graph_results(results):

    print("\nGRAPH RETRIEVAL RESULTS")
    print("=" * 60)

    print("\nQuestion:", results["query"])

    print("\nMatched entities:")

    for entity in results["matched_entities"]:
        print("-", entity)

    print("\nRetrieved relationships:")

    for relationship in results["relationships"]:

        print(
            f"\n{relationship['subject']} "
            f"--[{relationship['relation']}]--> "
            f"{relationship['object']}"
        )

        evidence = relationship["evidence"]

        print("Evidence records:", len(evidence))

        for item in evidence[:3]:

            print("  Source:", item.get("source", "Unknown"))
            print("  Page:", item.get("page_number", "Unknown"))
            print("  Chunk:", item.get("chunk_id", "Unknown"))

            print(
                "  Passage:",
                item.get("evidence_text", "Not available")
            )


if __name__ == "__main__":

    from graph_storage import load_graph

    graph = load_graph()

    question = "What is connected to Attention Mechanism?"

    results = retrieve_from_graph(
        graph=graph,
        query=question
    )

    print_graph_results(results)
