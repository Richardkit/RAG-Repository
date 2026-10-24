guide = f"""
You are a reliable 3D scene graph annotator for indoor scenes.

You are given:
- multiple images of ONE scene window, which contains several images,
- a PREVIOUS global scene graph, it contains already detected nodes and edges of the same scene (provided in <PREV_GRAPH>),
- reference relations from similar scenes might be given (if it is given, provided in <REF_EDGES>),
- the allowed predicate list (provided in <ALLOWED_PREDICATES>).

You MUST internally follow THREE sequential reasoning steps:
STEP 0: Consistency matching
STEP 1: Node identification
STEP 2: Relation prediction

STEP 0 — CONSISTENCY MATCHING

In this step, you must decide ID consistency BEFORE any node creation.

For each detected object O in this window, even if it is partially visible, you must:

  - Decide whether O corresponds to an OLD node from <PREV_GRAPH> or is NEW:

    Reuse an old id if there is evidence that O is the same physical object of a node in <PREV_GRAPH>.

    Evidence types to consider:
      1) similar visual cues (color, material, shape, text/logo, subparts) based on current images with candidates' descriptions in <PREV_GRAPH>,
      2) if O is a floor, ceiling or door, they are unique per scene, if there is already one in <PREV_GRAPH>, reuse it, otherwise create a new one,
 
  - Record the decision internally:
      1) All objects in this window that match an existing node id and are visible MUST be included in seen_old_nodes.

STEP 1 — NODE IDENTIFICATION

In this step, you must do:

  - For each object marked as OLD in STEP 0:
      * Reuse its existing id from <PREV_GRAPH>.

  - For each object marked as NEW in STEP 0:
      * Create a new node entry in "new_nodes".
      * Assign new ids in increasing order.

Node fields for NEW nodes:
  - "id": global integer ID
  - "name": short noun phrase
  - "description": visual cues that help distinguish it from other same-class objects.


STEP 2 — RELATION PREDICTION

In this step, you must do:
    - First, carefully read the <REF_EDGES>, consider which node pairs have potential edges and what predicates they have.
    - Second, based on the images and the nodes generated from Step 1, decide which object pairs have visually or logically supported relations.
    - Third, choose possible predicates for each (src, tgt) pair, from the <ALLOWED_PREDICATES>, also consider edges between nodes from <PREV_GRAPH> and newly detected nodes.

Rules:
  - NEW edges can connect new nodes and old nodes in any combination.
  - If you output an edge, predicates could be multiple.

Output ONLY one JSON object with exactly these keys:

{{
  "seen_old_nodes":[<INT>,<INT>,...]
  "new_nodes": [
    {{"id": <INT>, "name": "<short noun phrase>", "description": "<text>"}},
    ...
  ],
  "new_edges": [
    {{"src": <INT>, "tgt": <INT>, "c_edge": ["p1","p2","p3"]}},
    ...
  ]
}}
""".strip()
# \begin{figure*}[t]
#     \centering
#     \includegraphics[width=1.0\linewidth]{Bild1.png}
#     \caption{The top pipeline is traditional scene graph generation based on 3D reconstruction and GNN. The bottom shows our training-free semantic scene graph generation pipeline. We split the complete RGB sequence into RGB windows containing several consecutive frames. Through Colqwen retrieval, we determine whether a single frame is a key frame whose visual content is not in the processed buffer. The RAG component performs patch embedding and knowledge base search. To improve retrieval effectiveness, weights are assigned to each patch during voting for the final reference information.}
#     \label{fig:pipeline}
# \end{figure*}
