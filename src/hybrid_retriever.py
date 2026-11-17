
from vector_retriever import (
    load_vector_store,
    retrieve_chunks
)

from graph_retriever import retrieve_from_graph
from graph_storage import load_graph


def normalize_vector_results(vector_results): # normalize_vector_results() — converts FAISS results into a common format.
    """
    Convert vector retrieval results into a common
    evidence structure.
    """

    normalized = []

    for item in vector_results:

        normalized.append({
            "text": item.get("text", ""),
            "source": item.get("source", "Unknown"),
            "page_number": item.get("page_number"),
            "chunk_id": item.get("chunk_id"),
            "retrieval_method": ["vector"],
            "distance": item.get("distance"),
            "relationships": []
        })

    return normalized


def normalize_graph_results(graph_results): # normalize_graph_results() — converts graph evidence into the same format.
    """
    Convert graph relationship evidence into the
    same evidence structure used by vector retrieval.
    """

    normalized = []

    for relationship in graph_results["relationships"]:

        subject = relationship["subject"]
        relation = relationship["relation"]
        object_ = relationship["object"]

        for item in relationship.get("evidence", []):

            normalized.append({
                "text": item.get("evidence_text", ""),
                "source": item.get("source", "Unknown"),
                "page_number": item.get("page_number"),
                "chunk_id": item.get("chunk_id"),
                "retrieval_method": ["graph"],
                "distance": None,
                "relationships": [{
                    "subject": subject,
                    "relation": relation,
                    "object": object_
                }]
            })

    return normalized


def merge_evidence(vector_evidence, graph_evidence):
    """
    Combine vector and graph evidence.
    Remove duplicate passages while preserving
    their retrieval methods and relationships.
    """

    merged = {}

    all_evidence = vector_evidence + graph_evidence

    for item in all_evidence:

        # Source + page + chunk identify the evidence record
        key = (
            item["source"],
            item["page_number"],
            item["chunk_id"]
        )

        if key not in merged:

            merged[key] = item.copy()

            merged[key]["retrieval_method"] = list(
                item["retrieval_method"]
            )

            merged[key]["relationships"] = list(
                item["relationships"]
            )

        else:

            existing = merged[key]

            # Record all methods that found this passage
            for method in item["retrieval_method"]:

                if method not in existing["retrieval_method"]:
                    existing["retrieval_method"].append(method)

            # Preserve graph relationships
            for relationship in item["relationships"]:

                if relationship not in existing["relationships"]:
                    existing["relationships"].append(relationship)

            # Keep the passage text if the first result was empty
            if not existing["text"] and item["text"]:
                existing["text"] = item["text"]

    return list(merged.values())


def hybrid_retrieve( # hybrid_retrieve() — combines and deduplicates the evidence.
    query: str,
    vector_store,
    graph,
    top_k: int = 5
):
    """
    Run vector retrieval and graph retrieval,
    then combine their evidence.
    """

    print("\n========== HYBRID RETRIEVAL ==========")

    # 1. Vector retrieval
    print("\nRunning vector retrieval...")

    vector_results = retrieve_chunks(
        vector_store=vector_store,
        query=query,
        top_k=top_k
    )

    # 2. Graph retrieval
    print("\nRunning graph retrieval...")

    graph_results = retrieve_from_graph(
        graph=graph,
        query=query
    )

    # 3. Normalize both result types
    vector_evidence = normalize_vector_results(
        vector_results
    )

    graph_evidence = normalize_graph_results(
        graph_results
    )

    # 4. Merge and deduplicate
    combined_evidence = merge_evidence(
        vector_evidence,
        graph_evidence
    )

    return {
        "query": query,
        "matched_entities": graph_results["matched_entities"],
        "relationships": graph_results["relationships"],
        "vector_result_count": len(vector_results),
        "graph_evidence_count": len(graph_evidence),
        "combined_evidence": combined_evidence
    }


def print_hybrid_results(results):

    print("\n========== HYBRID RESULTS ==========")

    print("\nQuestion:", results["query"])

    print("\nMatched entities:")

    for entity in results["matched_entities"]:
        print("-", entity)

    print("\nVector results:", results["vector_result_count"])

    print("Graph evidence:", results["graph_evidence_count"])

    print("Unique combined evidence:", len(results["combined_evidence"]))

    for index, item in enumerate(
        results["combined_evidence"],
        start=1
    ):

        print("\n-----------------------------------")

        print("Evidence:", index)

        print("Source:", item["source"])
        print("Page:", item["page_number"])
        print("Chunk:", item["chunk_id"])

        print(
            "Found by:",
            ", ".join(item["retrieval_method"])
        )

        if item["relationships"]:

            print("Supporting relationships:")

            for relationship in item["relationships"]:

                print(
                    f"{relationship['subject']} "
                    f"--[{relationship['relation']}]--> "
                    f"{relationship['object']}"
                )

        print("Passage:")
        print(item["text"])


if __name__ == "__main__":

    question = "What does the Transformer use?"

    vector_store = load_vector_store()

    graph = load_graph()

    results = hybrid_retrieve(
        query=question,
        vector_store=vector_store,
        graph=graph,
        top_k=5
    )

    print_hybrid_results(results)
