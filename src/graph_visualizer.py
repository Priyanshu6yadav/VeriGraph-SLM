
import json
from pathlib import Path

from pyvis.network import Network
from graph_storage import load_graph


def visualize_graph(graph):

    net = Network(
        height="780px",
        width="100%",
        directed=True,
        bgcolor="#101827",
        font_color="#F8FAFC"
    )

    net.barnes_hut(
        gravity=-5000,
        central_gravity=0.25,
        spring_length=220,
        spring_strength=0.04,
        damping=0.12
    )

    # 1. Add entity nodes
    for node in graph.nodes():

        degree = graph.degree(node)

        net.add_node(
            node,
            label=node,
            title=f"Entity: {node}\nConnections: {degree}",
            size=18 + degree * 3,
            color={
                "background": "#3B82F6",
                "border": "#BFDBFE",
                "highlight": {
                    "background": "#F59E0B",
                    "border": "#FDE68A"
                }
            },
            font={
                "size": 18,
                "color": "#F8FAFC",
                "face": "Arial"
            }
        )

    # 2. Add relationships and preserve complete evidence
    for subject, object_, key, data in graph.edges(
        keys=True,
        data=True
    ):

        evidence = data.get("evidence", [])

        evidence_json = json.dumps(
            evidence,
            ensure_ascii=False
        )

        net.add_edge(
            subject,
            object_,
            label=data["relation"],
            title=(
                f"{data['relation']}\n"
                f"Evidence records: {len(evidence)}"
            ),
            evidence_json=evidence_json,
            arrows="to",
            color={
                "color": "#94A3B8",
                "highlight": "#F59E0B"
            },
            font={
                "size": 13,
                "color": "#E2E8F0",
                "strokeWidth": 3,
                "strokeColor": "#101827"
            },
            width=2
        )

    # 3. Generate HTML
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "knowledge_graph.html"

    net.write_html(
        str(output_path),
        open_browser=False
    )

    # 4. Add interactive explorer and evidence inspector
    html = output_path.read_text(encoding="utf-8")

    explorer_ui = r"""
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #101827;
            color: #F8FAFC;
        }

        #toolbar {
            padding: 12px;
            display: flex;
            gap: 10px;
            background: #172338;
            align-items: center;
            flex-wrap: wrap;
        }

        #searchBox {
            width: 260px;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #64748B;
            background: #0F172A;
            color: white;
        }

        button {
            padding: 10px 14px;
            border: 0;
            border-radius: 6px;
            background: #2563EB;
            color: white;
            cursor: pointer;
        }

        button:hover {
            background: #1D4ED8;
        }

        #status {
            color: #CBD5E1;
            font-size: 13px;
        }

        #details {
            padding: 18px;
            background: #172338;
            border-top: 1px solid #334155;
            min-height: 150px;
        }

        #details h2 {
            color: #93C5FD;
            margin-top: 0;
        }

        .evidence-card {
            background: #0F172A;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 14px;
            margin: 12px 0;
        }

        .evidence-card h3 {
            color: #FBBF24;
            margin-top: 0;
            font-size: 16px;
        }

        .evidence-meta {
            color: #93C5FD;
            font-size: 13px;
            margin-bottom: 10px;
        }

        .evidence-text {
            color: #E2E8F0;
            line-height: 1.6;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }

        .empty-message {
            color: #CBD5E1;
        }

        #mynetwork {
            background: #101827;
        }
    </style>

    <div id="toolbar">
        <input
            id="searchBox"
            type="text"
            placeholder="Search entity, e.g. Transformer"
        />

        <button onclick="searchEntity()">Search</button>
        <button onclick="resetGraph()">Reset Graph</button>

        <span id="status">
            Select a node or relationship.
        </span>
    </div>

    <div id="details">
        <h2>Evidence Inspector</h2>
        <p class="empty-message">
            Click a relationship edge to inspect its supporting
            document passages.
        </p>
    </div>

    <script>

        // Safely display text in the evidence panel
        function addText(parent, tag, text, className) {

            const element = document.createElement(tag);

            element.textContent = text;

            if (className) {
                element.className = className;
            }

            parent.appendChild(element);

            return element;
        }


        // Display an entity
        function showEntity(nodeId) {

            const node = nodes.get(nodeId);

            const connected = network.getConnectedNodes(nodeId);

            const panel = document.getElementById("details");

            panel.replaceChildren();

            addText(
                panel,
                "h2",
                "Entity: " + node.label
            );

            addText(
                panel,
                "p",
                "Number of connected entities: " + connected.length
            );

            addText(
                panel,
                "h3",
                "Connected entities"
            );

            if (connected.length === 0) {

                addText(
                    panel,
                    "p",
                    "No connected entities found.",
                    "empty-message"
                );

            } else {

                connected.forEach(id => {

                    const connectedNode = nodes.get(id);

                    addText(
                        panel,
                        "p",
                        connectedNode.label
                    );

                });

            }
        }


        // Display all evidence for a selected relationship
        function showRelationship(edge) {

            const panel = document.getElementById("details");

            panel.replaceChildren();

            const subject = nodes.get(edge.from);
            const object = nodes.get(edge.to);

            addText(
                panel,
                "h2",
                "Relationship: " + edge.label
            );

            addText(
                panel,
                "p",
                "From: " + subject.label
            );

            addText(
                panel,
                "p",
                "To: " + object.label
            );

            let evidence = [];

            try {

                evidence = JSON.parse(
                    edge.evidence_json || "[]"
                );

            } catch (error) {

                addText(
                    panel,
                    "p",
                    "Could not read evidence data.",
                    "empty-message"
                );

                return;
            }

            addText(
                panel,
                "h3",
                "Supporting evidence: " + evidence.length
            );

            if (evidence.length === 0) {

                addText(
                    panel,
                    "p",
                    "No evidence records are attached to this relationship.",
                    "empty-message"
                );

                return;
            }

            evidence.forEach((item, index) => {

                const card = document.createElement("div");

                card.className = "evidence-card";

                addText(
                    card,
                    "h3",
                    "Evidence " + (index + 1)
                );

                addText(
                    card,
                    "div",
                    "Source: " + (item.source || "Unknown") +
                    " | Page: " + (item.page_number ?? "Unknown") +
                    " | Chunk: " + (item.chunk_id || "Unknown"),
                    "evidence-meta"
                );

                addText(
                    card,
                    "div",
                    item.evidence_text || "No passage text available.",
                    "evidence-text"
                );

                panel.appendChild(card);

            });
        }


        // Search for an entity
        function searchEntity() {

            const query = document
                .getElementById("searchBox")
                .value.trim()
                .toLowerCase();

            if (!query) {
                alert("Enter an entity name first.");
                return;
            }

            const matches = nodes.get().filter(node =>
                String(node.label || node.id)
                    .toLowerCase()
                    .includes(query)
            );

            if (matches.length === 0) {
                alert("No matching entity found.");
                return;
            }

            const match = matches[0];

            network.selectNodes([match.id]);

            network.focus(match.id, {
                scale: 1.2,
                animation: true
            });

            highlightNeighborhood(match.id);

            showEntity(match.id);

            document.getElementById("status").innerText =
                "Found: " + match.label;
        }


        // Highlight selected entity and its direct neighbors
        function highlightNeighborhood(nodeId) {

            const connected = network.getConnectedNodes(nodeId);

            const visibleIds = new Set([
                nodeId,
                ...connected
            ]);

            nodes.get().forEach(node => {

                const active = visibleIds.has(node.id);

                nodes.update({
                    id: node.id,
                    color: {
                        background: active ? "#F59E0B" : "#334155",
                        border: active ? "#FDE68A" : "#475569"
                    }
                });

            });

            edges.get().forEach(edge => {

                const active =
                    visibleIds.has(edge.from) &&
                    visibleIds.has(edge.to) &&
                    (edge.from === nodeId || edge.to === nodeId);

                edges.update({
                    id: edge.id,
                    color: {
                        color: active ? "#F59E0B" : "#334155",
                        opacity: active ? 1 : 0.2
                    },
                    width: active ? 3 : 1
                });

            });
        }


        // Restore original graph colors
        function resetGraph() {

            nodes.get().forEach(node => {

                nodes.update({
                    id: node.id,
                    color: {
                        background: "#3B82F6",
                        border: "#BFDBFE"
                    }
                });

            });

            edges.get().forEach(edge => {

                edges.update({
                    id: edge.id,
                    color: {
                        color: "#94A3B8",
                        opacity: 1
                    },
                    width: 2
                });

            });

            network.unselectAll();

            document.getElementById("status").innerText =
                "Showing complete graph.";

            const panel = document.getElementById("details");

            panel.replaceChildren();

            addText(
                panel,
                "h2",
                "Evidence Inspector"
            );

            addText(
                panel,
                "p",
                "Select a relationship to inspect its supporting passages."
            );
        }


        // Node click
        network.on("selectNode", function(params) {

            if (!params.nodes.length) return;

            const nodeId = params.nodes[0];

            highlightNeighborhood(nodeId);

            showEntity(nodeId);

            document.getElementById("status").innerText =
                "Selected entity: " + nodes.get(nodeId).label;

        });


        // Relationship click
        network.on("selectEdge", function(params) {

            if (!params.edges.length) return;

            const edge = edges.get(params.edges[0]);

            showRelationship(edge);

            document.getElementById("status").innerText =
                "Selected relationship: " + edge.label;

        });


        // Search using Enter key
        document.getElementById("searchBox")
            .addEventListener("keydown", function(event) {

                if (event.key === "Enter") {
                    searchEntity();
                }

            });

    </script>
    """

    html = html.replace("</body>", explorer_ui + "\n</body>")

    output_path.write_text(
        html,
        encoding="utf-8"
    )

    print("\nInteractive Graph Explorer created!")
    print("Location:", output_path.resolve())


if __name__ == "__main__":

    graph = load_graph()

    visualize_graph(graph)
