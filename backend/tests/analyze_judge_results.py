import os
import sys
import json
from typing import List, Dict, Any
import argparse
from collections import defaultdict, Counter


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def token_set(s: str) -> set:
    import re
    toks = re.findall(r"[A-Za-z0-9_]+", (s or '').lower())
    return set(toks)


def overlap_ratio(a: str, b: str) -> float:
    A = token_set(a)
    B = token_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return inter / float(len(A))


def analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregate judge scores if present
    dims = ["home_relevance", "related_relevance", "coverage", "hallucination_risk", "overall"]
    sums = defaultdict(float)
    counts = Counter()

    # Collect worst cases and simple overlap stats when available
    worst = []
    overlap_stats = []

    for r in rows:
        j = r.get("judge") or {}
        for k in dims:
            v = j.get(k)
            if isinstance(v, (int, float)):
                sums[k] += float(v)
                counts[k] += 1
        # Selection and home overlap if present in this row structure
        sel = r.get("selection") or j.get("selection") or ""
        # Try to find home text if embedded; some rows may only have URLs
        home_text = None
        # Some harness versions put a compact copy of home
        compact_home = r.get("home") or {}
        # No full text available here, we can only compute 0 overlap
        ov = 0.0
        if isinstance(sel, str) and isinstance(compact_home, dict):
            # home text may not be present; leave 0.0 if absent
            pass
        overlap_stats.append(ov)

        # Track worst by overall score
        overall = j.get("overall", 0)
        worst.append((overall if isinstance(overall, (int, float)) else 0, r))

    avg = {k: (sums[k] / counts[k] if counts[k] else None) for k in dims}
    worst_sorted = sorted(worst, key=lambda x: x[0])[:5]

    return {
        "averages": avg,
        "worst_cases": [w[1] for w in worst_sorted],
        "count": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to judge_results_*.jsonl")
    args = ap.parse_args()

    rows = read_jsonl(args.path)
    if not rows:
        print("No rows found or invalid file.")
        sys.exit(1)

    report = analyze(rows)
    print("Averages:")
    for k, v in report["averages"].items():
        if v is not None:
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: n/a")

    print("\nWorst 5 cases (truncated):")
    for i, r in enumerate(report["worst_cases"], 1):
        sel = (r.get("selection") or "")[:140]
        judge = r.get("judge") or {}
        overall = judge.get("overall")
        feedback = (str(judge.get("feedback", ""))[:240]) if judge else ""
        print(f"[{i}] overall={overall} | selection='{sel}...'\n    feedback={feedback}")


if __name__ == "__main__":
    main()
