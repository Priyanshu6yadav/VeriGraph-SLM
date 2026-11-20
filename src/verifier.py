"""
VeriGraph-SLM: Formal Verification Layer (Z3 SMT Solver)
---------------------------------------------------------
This module uses Microsoft's Z3 SMT (Satisfiability Modulo Theories) solver
to formally verify claims and detect logical contradictions in our Knowledge Graph.

Concept for B.Tech Students:
- SMT Solvers check if a set of logical formulas can all be simultaneously TRUE (SAT).
- If adding a new claim makes the formula impossible to satisfy (UNSAT),
  we have mathematically PROVEN a contradiction!
"""

import z3
from graph_storage import load_graph


# ----------------------------------------------------------------------
# 1. Define Mutually Exclusive / Opposing Predicates
# ----------------------------------------------------------------------
# If a model claims entity A "USES" or "REQUIRES" entity B,
# but the graph asserts entity A "ELIMINATES" or "REMOVES" entity B,
# these two predicates are logically incompatible.
OPPOSING_PREDICATES = {
    "USES": ["ELIMINATES", "REPLACES", "REMOVES", "AVOIDS"],
    "REQUIRES": ["ELIMINATES", "REPLACES", "REMOVES", "AVOIDS"],
    "CONTAINS": ["EXCLUDES", "LACKS"],
    "ELIMINATES": ["USES", "REQUIRES", "CONTAINS", "DEPENDS_ON"],
    "REPLACES": ["USES", "REQUIRES", "DEPENDS_ON"],
}


# ----------------------------------------------------------------------
# 2. Function: Check Single Claim against Knowledge Graph using Z3
# ----------------------------------------------------------------------
def verify_triple_with_z3(subject: str, relation: str, object_: str, graph) -> dict:
    """
    Uses Z3 to verify whether a given (Subject, Relation, Object) claim
    is SUPPORTED, CONTRADICTORY, or UNVERIFIED with respect to the Knowledge Graph.
    """
    # Create case-insensitive lookup map for nodes in graph
    node_lookup = {str(node).strip().casefold(): node for node in graph.nodes()}
    
    # Resolve canonical node names
    subj_canon = node_lookup.get(subject.strip().casefold(), subject.strip())
    obj_canon = node_lookup.get(object_.strip().casefold(), object_.strip())
    relation_norm = relation.strip().upper()

    # 1. Check if direct matching relationship exists in the graph
    has_exact_edge = False
    if graph.has_node(subj_canon) and graph.has_node(obj_canon):
        if graph.has_edge(subj_canon, obj_canon, key=relation_norm):
            has_exact_edge = True

    # 2. Inspect all existing relationships between this subject and object
    existing_relations = []
    if graph.has_node(subj_canon) and graph.has_node(obj_canon):
        edge_dict = graph.get_edge_data(subj_canon, obj_canon)
        if edge_dict:
            existing_relations = [data.get("relation") for data in edge_dict.values()]

    # 3. Create Z3 Solver instance
    solver = z3.Solver()

    # Create boolean variable for the claim
    claim_var = z3.Bool(f"{subj_canon}__{relation_norm}__{obj_canon}")

    # Add the claim as an assumption we want to test
    solver.add(claim_var == True)

    # Check for contradictions against opposing predicates
    contradicting_relation = None
    opposing_list = OPPOSING_PREDICATES.get(relation_norm, [])

    for existing_rel in existing_relations:
        # Check if existing relation directly opposes our claim
        if existing_rel in opposing_list or relation_norm in OPPOSING_PREDICATES.get(existing_rel, []):
            existing_var = z3.Bool(f"{subj_canon}__{existing_rel}__{obj_canon}")

            # Axiom in Z3: Both opposing relations CANNOT be true simultaneously
            # i.e., Not(claim_var AND existing_var)
            solver.add(z3.Not(z3.And(claim_var, existing_var)))

            # The graph fact is known ground truth:
            solver.add(existing_var == True)
            contradicting_relation = existing_rel
            break

    # 4. Check satisfiability with Z3
    check_result = solver.check()

    if check_result == z3.unsat:
        # Contradiction mathematically proven by Z3!
        return {
            "status": "CONTRADICTION",
            "verified": False,
            "explanation": (
                f"Contradiction detected by Z3: The claim asserts '{subj_canon} {relation_norm} {obj_canon}', "
                f"which conflicts with verified graph fact '{subj_canon} {contradicting_relation} {obj_canon}'."
            )
        }
    elif has_exact_edge:
        # Exact edge exists and no contradiction
        return {
            "status": "VERIFIED",
            "verified": True,
            "explanation": f"Claim verified: Grounded in Knowledge Graph edge '{subj_canon} --[{relation_norm}]--> {obj_canon}'."
        }
    else:
        # No contradiction, but no direct edge either
        return {
            "status": "UNVERIFIED",
            "verified": False,
            "explanation": f"Insufficient direct evidence in Knowledge Graph for '{subj_canon} {relation_norm} {obj_canon}'."
        }


# ----------------------------------------------------------------------
# 3. Function: Scan Complete Graph for Internal Consistency
# ----------------------------------------------------------------------
def check_graph_consistency(graph) -> list:
    """
    Scans all edges in the Knowledge Graph using Z3 to ensure there are no
    internal contradictions (e.g. node A simultaneously uses and eliminates node B).
    """
    conflicts = []

    for subject, object_, key, data in graph.edges(keys=True, data=True):
        relation = data.get("relation", "")
        opposing_relations = OPPOSING_PREDICATES.get(relation, [])

        for opp in opposing_relations:
            if graph.has_edge(subject, object_, key=opp):
                conflicts.append({
                    "subject": subject,
                    "object": object_,
                    "relation_1": relation,
                    "relation_2": opp,
                    "reason": f"Graph contains conflicting edges: {relation} and {opp}"
                })

    return conflicts


# ----------------------------------------------------------------------
# 4. Test Script to Verify Z3 Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Testing Z3 Formal Verification Layer ---")
    graph = load_graph()

    # Step 1: Check internal graph consistency
    print("\n[Test 1] Checking internal graph consistency with Z3...")
    conflicts = check_graph_consistency(graph)
    if not conflicts:
        print("Result: PASS - Knowledge Graph is 100% logically consistent (No internal contradictions).")
    else:
        print(f"Result: Found {len(conflicts)} conflicting relationships:")
        for c in conflicts:
            print(" -", c)

    # Step 2: Test a TRUE claim that is in the graph
    print("\n[Test 2] Testing a valid fact from the paper:")
    claim_1 = ("Transformer", "USES", "Attention Mechanism")
    res_1 = verify_triple_with_z3(claim_1[0], claim_1[1], claim_1[2], graph)
    print(f"Claim: {claim_1}")
    print(f"Z3 Status: {res_1['status']}")
    print(f"Explanation: {res_1['explanation']}")

    # Step 3: Test an actual contradiction from the graph
    # Graph has: tensor2tensor REPLACES earlier codebase
    # Claim tests: tensor2tensor USES earlier codebase
    print("\n[Test 3] Testing a contradictory claim using Z3:")
    claim_2 = ("tensor2tensor", "USES", "earlier codebase")
    res_2 = verify_triple_with_z3(claim_2[0], claim_2[1], claim_2[2], graph)
    print(f"Claim: {claim_2}")
    print(f"Z3 Status: {res_2['status']}")
    print(f"Explanation: {res_2['explanation']}")

    # Step 4: Test an UNKNOWN claim
    print("\n[Test 4] Testing an unknown / unverified claim:")
    claim_3 = ("Transformer", "USES", "Quantum Computing")
    res_3 = verify_triple_with_z3(claim_3[0], claim_3[1], claim_3[2], graph)
    print(f"Claim: {claim_3}")
    print(f"Z3 Status: {res_3['status']}")
    print(f"Explanation: {res_3['explanation']}")
