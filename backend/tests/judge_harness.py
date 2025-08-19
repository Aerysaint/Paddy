import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import requests

# Optionally load GOOGLE_API_KEY from backend/.env when running outside server
try:
    from dotenv import load_dotenv  # type: ignore
    _has_dotenv = True
except Exception:
    _has_dotenv = False

# Reuse the project's LLM interface (Gemini preferred)
# Ensure LLM_PROVIDER=gemini and GOOGLE_API_KEY is set in your environment
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # add backend/ to path
from llm_call import get_llm_response  # type: ignore

BACKEND = os.environ.get("BACKEND_ORIGIN", "http://localhost:8000")
WORKDIR = os.path.join("backend", "tests", "_known_pdfs")
RESULTS_DIR = os.path.join("backend", "tests", "results")

KNOWN_PDFS = [
    {"name": "attention_is_all_you_need_1706.03762.pdf", "url": "https://arxiv.org/pdf/1706.03762.pdf"},
    {"name": "bert_1810.04805.pdf", "url": "https://arxiv.org/pdf/1810.04805.pdf"},
    {"name": "resnet_1512.03385.pdf", "url": "https://arxiv.org/pdf/1512.03385.pdf"},
    {"name": "gan_1406.2661.pdf", "url": "https://arxiv.org/pdf/1406.2661.pdf"},
]


def ensure_dirs():
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Best-effort: load env so GOOGLE_API_KEY is visible when running harness directly
    if _has_dotenv:
        backend_dir = os.path.join(os.path.dirname(__file__), "..")
        env_main = os.path.abspath(os.path.join(backend_dir, ".env"))
        env_local = os.path.abspath(os.path.join(backend_dir, ".env.local"))
        if _has_dotenv:
            for p in (env_main, env_local):
                if os.path.exists(p):
                    try:
                        load_dotenv(p, override=False)  # type: ignore[name-defined]
                    except Exception:
                        pass


def ensure_downloads() -> List[str]:
    ensure_dirs()
    paths = []
    for item in KNOWN_PDFS:
        out_path = os.path.join(WORKDIR, item["name"])
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 10000:
            print(f"Downloading {item['url']} -> {out_path}")
            r = requests.get(item["url"], timeout=120)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(r.content)
        paths.append(out_path)
    return paths


def ingest(paths: List[str]) -> Dict[str, Any]:
    url = f"{BACKEND}/api/ingest"
    files = [("files", (os.path.basename(p), open(p, "rb"), "application/pdf")) for p in paths]
    try:
        resp = requests.post(url, files=files, timeout=900)
        resp.raise_for_status()
        return resp.json()
    finally:
        for _, (name, fh, _) in files:
            try:
                fh.close()
            except Exception:
                pass


def get_library() -> Dict[str, Any]:
    url = f"{BACKEND}/api/library"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def run_insights(selection: str, top_k: int = 20, threshold: float = 0.2) -> Dict[str, Any]:
    url = f"{BACKEND}/api/insights"
    data = {"selection": selection, "top_k": str(top_k), "threshold": str(threshold), "stream": "false"}
    r = requests.post(url, data=data, timeout=600)
    r.raise_for_status()
    return r.json()


def run_insights_stream_meta(selection: str, top_k: int = 20, threshold: float = 0.2) -> Dict[str, Any]:
    """Use stream=true to retrieve meta first and avoid dependency on backend LLM creds."""
    url = f"{BACKEND}/api/insights"
    data = {"selection": selection, "top_k": str(top_k), "threshold": str(threshold), "stream": "true"}
    with requests.post(url, data=data, stream=True, timeout=600) as r:
        r.raise_for_status()
        meta_payload = None
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("event: meta"):
                continue
            if raw.startswith("data: "):
                payload = raw[len("data: ") :]
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                    if isinstance(obj, dict) and "home" in obj and "related" in obj:
                        meta_payload = obj
                        break
                except Exception:
                    pass
    if not meta_payload:
        raise RuntimeError("Failed to obtain meta payload from /api/insights stream")
    return meta_payload


def llm_json(messages: List[Dict[str, str]]) -> Any:
    """Call LLM and parse JSON; on failure, return raw text."""
    txt = get_llm_response(messages)
    if isinstance(txt, str):
        try:
            return json.loads(txt)
        except Exception:
            # Try to extract fenced JSON
            import re
            m = re.search(r"```(?:json)?\n(.*?)\n```", txt, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
            return txt
    return txt


def propose_selections_via_llm(library: Dict[str, Any], n_per_doc: int = 2) -> List[Dict[str, Any]]:
    # Build doc list summary
    docs = []
    for it in library.get("library", []):
        docs.append({"displayName": it.get("displayName"), "file": it.get("file"), "url": it.get("url")})
    sys_prompt = (
        "You are designing retrieval test inputs for research PDFs.\n"
        "Given a list of documents (filename + human display name), propose concise selection snippets (1-2 sentences) that are highly likely to appear or be paraphrased within the corresponding target paper.\n"
        "Sometimes a selection may be appropriate for more than one document. In that case, include multiple acceptable expectations.\n"
        "Return STRICT JSON only with this schema: {\"cases\":[{\"selection\": string, \"expected_file_contains\": string|null, \"expected_file_contains_any\": [string,...]|null, \"note\": string|null}]}.\n"
        "The values for expected_file_contains / expected_file_contains_any should be distinctive substrings that appear in the correct PDF URL or filename (e.g., an arXiv id like '1706.03762' or a short distinctive slug). Limit to about n_per_doc per document."
    )
    user_prompt = json.dumps({"documents": docs, "n_per_doc": n_per_doc}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = llm_json(messages)
    if isinstance(resp, dict) and isinstance(resp.get("cases"), list):
        cases = resp["cases"]
        # normalize and trim
        out = []
        for c in cases:
            sel = str(c.get("selection", "")).strip()
            exp_single = c.get("expected_file_contains")
            exp_list = c.get("expected_file_contains_any")
            note = c.get("note")
            # normalize expectations into a list of strings (may be empty)
            exp_any: List[str] = []
            if isinstance(exp_single, str) and exp_single.strip():
                exp_any.append(exp_single.strip())
            if isinstance(exp_list, list):
                for v in exp_list:
                    if isinstance(v, str) and v.strip():
                        exp_any.append(v.strip())
            # de-dup while preserving order
            seen = set()
            exp_any = [x for x in exp_any if not (x in seen or seen.add(x))]
            if sel and exp_any:
                out.append({
                    "selection": sel,
                    "expected_file_contains": exp_any[0],  # keep legacy single for compatibility
                    "expected_file_contains_any": exp_any,
                    "note": note,
                })
        if out:
            return out
    # fallback trivial cases
    return [
        {
            "selection": "Scaled dot-product attention and multi-head attention enable modeling relationships between all tokens.",
            "expected_file_contains": "1706.03762",
            "expected_file_contains_any": ["1706.03762"],
        },
        {
            "selection": "BERT uses masked language modeling and next sentence prediction during pretraining.",
            "expected_file_contains": "1810.04805",
            "expected_file_contains_any": ["1810.04805"],
        },
        {
            "selection": "Residual learning with identity shortcuts allows very deep networks.",
            "expected_file_contains": "1512.03385",
            "expected_file_contains_any": ["1512.03385"],
        },
        {
            "selection": "Generative Adversarial Networks train a generator and discriminator in a minimax game.",
            "expected_file_contains": "1406.2661",
            "expected_file_contains_any": ["1406.2661"],
        },
    ]


def judge_case_via_llm(
    selection: str,
    insights: Dict[str, Any],
    expected_file_contains: Optional[Union[str, List[str]]],
    doc_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Ask LLM to judge retrieval quality and optionally compare with expected document suffix."""
    home = insights.get("home") or {}
    related = insights.get("related") or []

    # Build a compact payload for the judge (limit text lengths)
    def trim(t: Optional[str], n: int = 1200) -> str:
        s = (t or "")
        return s[:n]

    # normalize expected list
    exp_any: List[str] = []
    if isinstance(expected_file_contains, str) and expected_file_contains.strip():
        exp_any.append(expected_file_contains.strip())
    elif isinstance(expected_file_contains, list):
        for v in expected_file_contains:
            if isinstance(v, str) and v.strip():
                exp_any.append(v.strip())

    judge_payload = {
        "selection": selection,
        "home": {
            "displayName": home.get("displayName"),
            "fileUrl": home.get("fileUrl"),
            "heading": home.get("heading"),
            "text": trim(home.get("text"), 1500),
        },
        "related": [
            {
                "displayName": r.get("displayName"),
                "fileUrl": r.get("fileUrl"),
                "heading": r.get("heading"),
                "text": trim(r.get("text"), 800),
            }
            for r in related[:3]
        ],
        "expected_file_contains": exp_any[0] if exp_any else None,
        "expected_file_contains_any": exp_any if exp_any else None,
        "documents": [{"displayName": d.get("displayName"), "file": d.get("file")} for d in doc_list],
        "instructions": (
            "Score the retrieval output on a 1-5 scale for: doc_match (does home doc seem correct?), relevance (home text matches selection), coverage (home+related cover the idea), related_support (related add helpful context).\n"
            "Return strict JSON: {\"expected_file_contains\": str|null, \"expected_file_contains_any\": [string,...]|null, \"predicted_file_url\": str|null, \"doc_match\": int, \"relevance\": int, \"coverage\": int, \"related_support\": int, \"overall\": int, \"feedback\": str}."
        ),
    }

    sys_prompt = (
        "You are an exacting evaluator for a retrieval system over PDFs."
        " Judge ONLY from the provided texts. Use the scoring rubric provided and return strict JSON."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(judge_payload, ensure_ascii=False)},
    ]
    resp = llm_json(messages)
    if isinstance(resp, dict):
        return resp
    # fallback minimal
    return {"overall": 3, "feedback": "Non-JSON response"}


def llm_multimodal_judge_gemini(
    google_api_key: str,
    pdf_paths: List[str],
    selection: str,
    system_home: Dict[str, Any],
    system_related: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Upload local PDFs to Gemini 2.5 Flash and request a strict-JSON judgment with expectations and scores.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency google-generativeai. Install with: pip install google-generativeai") from e

    # Configure with provided API key
    genai.configure(api_key='AIzaSyDpluHIhrQlx5LRDYKOeQWlPIbnYLr8F3E')

    # Upload a small set of PDFs to control latency/cost
    uploaded = []
    for p in pdf_paths:
        try:
            f = genai.upload_file(p)
            uploaded.append(f)
        except Exception as e:
            print(f"Gemini upload failed for {p}: {e}")
    if not uploaded:
        raise RuntimeError("No PDFs uploaded to Gemini judge.")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)

    # Compose prompt parts: instruction + selection + system outputs + attached files
    parts: List[Any] = []
    parts.append(
        (
            "You are an exact evaluation judge for a PDF retrieval system. "
            "You receive (a) a selection snippet, (b) the system's retrieved Home and Related section texts, "
            "and (c) the full PDFs as attachments.\n\n"
            "Return STRICT JSON only, matching this schema exactly (no prose, no extra keys):\n"
            "{\n  \"scores\": {\n    \"home_relevance\": 0-5,\n    \"related_relevance\": 0-5,\n    \"coverage\": 0-5,\n    \"hallucination_risk\": 0-5\n  },\n  \"expected\": {\n    \"expected_file_suffix\": string|null,\n    \"expected_keywords\": [string,...],\n    \"expected_heading_contains\": [string,...]\n  },\n  \"feedback\": string\n}"
        )
    )
    parts.append(f"Selection:\n{selection}")
    parts.append(
        (
            f"Home: {system_home.get('displayName')} | Heading: {system_home.get('heading')}\n"
            f"Home text (trunc):\n{(system_home.get('text') or '')[:2000]}\n\n"
            + "\n\n".join(
                [
                    f"Related {i+1}: {r.get('displayName')} | Heading: {r.get('heading')}\n{(r.get('text') or '')[:1000]}"
                    for i, r in enumerate(system_related[:3])
                ]
            )
        )
    )

    # Attach files as additional parts
    parts.extend(uploaded)

    try:
        # Newer google-generativeai expects a 'parts' array or simple list of parts
        resp = model.generate_content({"role": "user", "parts": parts})
        txt = resp.text or ""
        # Extract JSON if fenced or wrapped
        first = txt.find("{")
        last = txt.rfind("}")
        if first != -1 and last != -1 and last > first:
            txt = txt[first : last + 1]
        data = json.loads(txt)
    except Exception as e:
        raise RuntimeError(f"Gemini multimodal judge failed: {e}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_doc", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--ingest_known", action="store_true", help="Download and ingest known PDFs before judging")
    parser.add_argument("--multimodal", action="store_true", help="Use Gemini multimodal judge with PDF uploads")
    parser.add_argument("--limit_pdfs", type=int, default=3, help="Limit number of PDFs to upload to judge")
    args = parser.parse_args()

    ensure_dirs()

    # Optional ensure ingest
    if args.ingest_known:
        paths = ensure_downloads()
        print(f"Prepared {len(paths)} PDFs")
        print("Ingesting...")
        ing = ingest(paths)
        print("Ingest:", ing)

    # Library overview
    lib = get_library()
    doc_list = lib.get("library", [])
    if not doc_list:
        print("No library PDFs found. Ingest first.")
        sys.exit(1)

    # Ask LLM to propose selections + expected file suffixes (text-only LLM call)
    print("Proposing selections via LLM...")
    cases = propose_selections_via_llm(lib, n_per_doc=args.n_per_doc)

    results = []
    for i, c in enumerate(cases, 1):
        sel = c.get("selection")
        # support multiple expected file substrings
        exp_any = c.get("expected_file_contains_any")
        if isinstance(exp_any, list) and exp_any:
            exp: Optional[Union[str, List[str]]] = [str(v) for v in exp_any if isinstance(v, str) and v]
        else:
            exp_val = c.get("expected_file_contains")
            exp = str(exp_val).strip() if isinstance(exp_val, str) else None
        if not sel:
            continue
        print(f"\n=== Case {i} ===\nSelection: {sel}\nExpected file contains: {exp}")
        try:
            ins = run_insights(sel, top_k=args.top_k, threshold=args.threshold)
        except Exception as e:
            print("Insights error:", e)
            results.append({"selection": sel, "error": str(e)})
            continue
        # Choose judge: multimodal (Gemini with PDFs) or text-only (existing LLM)
        if args.multimodal:
            # Ensure we have an API key visible in this environment
            google_api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyDpluHIhrQlx5LRDYKOeQWlPIbnYLr8F3E").strip()
            if not google_api_key:
                print("Missing GOOGLE_API_KEY in environment for multimodal judge.")
                results.append({"selection": sel, "error": "GOOGLE_API_KEY not set for multimodal"})
                continue
            # Ensure local PDFs exist (download subset)
            paths = ensure_downloads()
            judge_pdfs = paths[: max(1, min(args.limit_pdfs, len(paths)))]
            try:
                judge = llm_multimodal_judge_gemini(
                    google_api_key=google_api_key,
                    pdf_paths=judge_pdfs,
                    selection=sel,
                    system_home=ins.get("home") or {},
                    system_related=ins.get("related") or [],
                )
            except Exception as e:
                print("Multimodal judge error:", e)
                judge = {"error": str(e)}
        else:
            judge = judge_case_via_llm(sel, ins, exp, doc_list)
        # Augment with quick automatic checks
        home = ins.get("home") or {}
        file_url = home.get("fileUrl") or ""
        # compute auto match against any expected suffix
        doc_match_auto = None
        match_hit: Optional[str] = None
        if isinstance(exp, str) and exp:
            doc_match_auto = exp in file_url
            match_hit = exp if doc_match_auto else None
        elif isinstance(exp, list):
            for v in exp:
                if isinstance(v, str) and v and (v in file_url):
                    doc_match_auto = True
                    match_hit = v
                    break
            if doc_match_auto is None:
                doc_match_auto = False
        judge_result = {
            "selection": sel,
            "expected_file_contains": exp if isinstance(exp, str) else (exp[0] if isinstance(exp, list) and exp else None),
            "expected_file_contains_any": exp if isinstance(exp, list) else ([exp] if isinstance(exp, str) and exp else None),
            "predicted_file_url": judge.get("predicted_file_url"),
            "doc_match_auto": doc_match_auto,
            "doc_match_hit": match_hit,
            "judge": judge,
        }
        print("Judge overall:", judge.get("overall"), "| doc_match_auto:", doc_match_auto)
        results.append(judge_result)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"judge_results_{ts}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved results to {out_path}")

    # Print a small summary
    try:
        scores = [int((r.get("judge") or {}).get("overall", 0)) for r in results if isinstance((r.get("judge") or {}).get("overall", None), (int, float))]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"Average overall score: {avg:.2f} over {len(scores)} cases")
    except Exception:
        pass


if __name__ == "__main__":
    main()
