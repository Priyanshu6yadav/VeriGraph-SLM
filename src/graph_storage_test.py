from graph_storage import load_graph
from graph_builder import get_graph_statistics


# Load saved graph
graph = load_graph()


# Display statistics
print("\nLoaded Graph Statistics:")

print(get_graph_statistics(graph))


# Inspect one relationship
subject = "Transformer"
object_ = "Attention Mechanism"
predicate = "USES"

if graph.has_edge(subject, object_, key=predicate):

    edge_data = graph[subject][object_][predicate]

    print("\nRelationship found:")
    print(f"{subject} --[{predicate}]--> {object_}")

    print(
        "Evidence records:",
        len(edge_data["evidence"])
    )

else:
    print("\nRelationship not found.")
