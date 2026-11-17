
import re
import networkx as nx


# Explicit aliases only. Add new aliases after reviewing them.
ENTITY_ALIASES = {
    "transformer": "Transformer",
    "multi-head attention": "Multi-Head Attention",
    "multi head attention": "Multi-Head Attention",
    "self-attention": "Self-Attention",
    "self attention": "Self-Attention",
    "positional encoding": "Positional Encoding",
    "positional encodings": "Positional Encoding",
}


def normalize_entity(entity):
    """
    Normalize entity formatting and apply reviewed aliases.
    """

    # Remove leading/trailing spaces and collapse repeated spaces
    entity = re.sub(r"\s+", " ", entity.strip())

    # Case-insensitive lookup
    normalized_key = entity.casefold()

    # Use a reviewed canonical name when available
    if normalized_key in ENTITY_ALIASES:
        return ENTITY_ALIASES[normalized_key]

    # Preserve original wording for unknown entities
    return entity


def normalize_predicate(predicate):
    """
    Normalize relationship names.
    """

    return re.sub(
        r"\s+",
        "_",
        predicate.strip().upper()
    )


def create_graph():
    """
    Create a directed multigraph.
    """

    return nx.MultiDiGraph()


def add_triples(graph, triples):
    """
    Add normalized relationships while preserving
    all unique evidence records.
    """

    for triple in triples:

        subject = normalize_entity(triple["subject"])
        predicate = normalize_predicate(triple["predicate"])
        object_ = normalize_entity(triple["object"])

        # Add canonical entities as nodes
        graph.add_node(subject)
        graph.add_node(object_)

        # Use predicate as edge key so repeated
        # subject-predicate-object triples share an edge.
        edge_key = predicate

        if graph.has_edge(subject, object_, key=edge_key):

            edge_data = graph[subject][object_][edge_key]

        else:

            graph.add_edge(
                subject,
                object_,
                key=edge_key,
                relation=predicate,
                evidence=[]
            )

            edge_data = graph[subject][object_][edge_key]

        # Build evidence record
        evidence_record = {
            "source": triple["source"],
            "page_number": triple["page_number"],
            "chunk_id": triple["chunk_id"],
            "evidence_text": triple["evidence_text"]
        }

        # Avoid adding the same chunk evidence twice
        evidence_exists = any(
            item["source"] == evidence_record["source"]
            and item["chunk_id"] == evidence_record["chunk_id"]
            for item in edge_data["evidence"]
        )

        if not evidence_exists:
            edge_data["evidence"].append(evidence_record)

    return graph


def get_graph_statistics(graph):

    total_evidence = sum(
        len(data["evidence"])
        for _, _, data in graph.edges(data=True)
    )

    return {
        "total_entities": graph.number_of_nodes(),
        "total_relationships": graph.number_of_edges(),
        "total_evidence_records": total_evidence
    }
