import networkx as nx


def create_graph():
    """
    Create an empty directed multigraph.
    """
    return nx.MultiDiGraph()


def add_triples(graph, triples):
    """
    Add entities and their relationships to the graph.
    Preserve document evidence on every relationship.
    """

    for triple in triples:

        subject = triple["subject"]
        predicate = triple["predicate"]
        object_ = triple["object"]

        # Add entities as graph nodes
        graph.add_node(subject)
        graph.add_node(object_)

        # Add the relationship as a directed edge
        graph.add_edge(
            subject,
            object_,
            relation=predicate,
            source=triple["source"],
            page_number=triple["page_number"],
            chunk_id=triple["chunk_id"]
        )

    return graph


def get_graph_statistics(graph):
    """
    Return basic graph statistics.
    """

    return {
        "total_entities": graph.number_of_nodes(),
        "total_relationships": graph.number_of_edges()
    }
