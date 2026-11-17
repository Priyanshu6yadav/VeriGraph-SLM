import pickle
from pathlib import Path


# Project storage location
GRAPH_DIR = Path("data/processed/knowledge_graph")

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRAPH_PATH = GRAPH_DIR / "knowledge_graph.pkl"


def save_graph(graph):

    with open(GRAPH_PATH, "wb") as file:
        pickle.dump(graph, file)

    print("Knowledge graph saved:", GRAPH_PATH)


def load_graph():

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge graph not found: {GRAPH_PATH}"
        )

    with open(GRAPH_PATH, "rb") as file:
        graph = pickle.load(file)

    print("Knowledge graph loaded successfully!")

    return graph
