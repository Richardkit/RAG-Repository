from typing import List, Dict, Any, Tuple, Optional

def build_predicate_agent_messages(reference_edges: List[Dict[str, Any]]):
    import json as _json

    edges_str = _json.dumps(
        reference_edges or [],
        ensure_ascii=False,
        separators=(",", ":")
    )

    content: List[Dict[str, Any]] = []

    guide = """
You are a scene-graph predicate policy compiler.

You are given retrieved <REFERENCE_EDGES> from similar scenes as a list of triples.
Each edge has:
- src: source object class
- tgt: target object class
- c_edge: predicate

Example:
If src is "chair", tgt is "floor", and c_edge is "standing on",
it means "the chair is standing on the floor".

Your task:
0) Extract ALL unique predicate strings appearing in <REFERENCE_EDGES> (from the `c_edge` field).
   You MUST list them in "all_predicates".

1) For EACH predicate in "all_predicates", provide:
   - typical situation ("when")
   - 2–6 typical object pairs "A->B" to CHECK (NOT to copy)

2) If a predicate is rare (frequency=1) or uncertain, you can keep "when"/"avoid" very short,
   but you MUST still include it (do NOT omit any predicate).

3) DO NOT invent predicates. The predicate string must EXACTLY match one of the `c_edge` values in <REFERENCE_EDGES>.

Return ONLY one JSON object with this schema:
{
  "all_predicates": ["p1","p2",...],
  "possible_predicates":[
    {
      "predicate":"<one predicate from all_predicates>",
      "when":"...",
      "check_pairs":["A->B","C->D"],
    }
  ],
  "global_rules":[ "...", "..." ]
}

""".strip()

    content.append({"type": "text", "text": guide})
    content.append({
        "type": "text",
        "text": f"<REFERENCE_EDGES>\n{edges_str}\n</REFERENCE_EDGES>"
    })

    return [{"role": "user", "content": content}]



