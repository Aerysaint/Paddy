import os
import sys
import json
import time
from typing import List, Dict, Any

import argparse
import requests

BACKEND = os.environ.get("BACKEND_ORIGIN", "http://localhost:8000")
WORKDIR = os.path.join("backend", "tests", "_known_pdfs")

# Known public PDFs (arXiv)
KNOWN_PDFS = [
    {
        "name": "attention_is_all_you_need_1706.03762.pdf",
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
    },
    {
        "name": "bert_1810.04805.pdf",
        "url": "https://arxiv.org/pdf/1810.04805.pdf",
    },
    {
        "name": "resnet_1512.03385.pdf",
        "url": "https://arxiv.org/pdf/1512.03385.pdf",
    },
    {
        "name": "gan_1406.2661.pdf",
        "url": "https://arxiv.org/pdf/1406.2661.pdf",
    },
]

# Test cases: selection -> expected paper (by arXiv id suffix), across the whole corpus
TEST_CASES = [
    {
        "selection": "Scaled dot-product attention with softmax(QK^T / sqrt(d_k)) and multi-head attention",
        "expect_file_suffix": "1706.03762.pdf",
    },
    {
        "selection": "A generator and a discriminator trained in a minimax two-player game to produce realistic samples",
        "expect_file_suffix": "1406.2661.pdf",
    },
    {
        "selection": "Residual learning framework enables very deep networks via identity shortcut connections",
        "expect_file_suffix": "1512.03385.pdf",
    },
    {
        "selection": "BERT pre-training uses masked language model (MLM) and next sentence prediction (NSP)",
        "expect_file_suffix": "1810.04805.pdf",
    },
]


def ensure_downloads() -> List[str]:
    os.makedirs(WORKDIR, exist_ok=True)
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
        resp = requests.post(url, files=files, timeout=600)
        resp.raise_for_status()
        return resp.json()
    finally:
        for _, (name, fh, _) in files:
            try:
                fh.close()
            except Exception:
                pass


def run_insights(selection: str, top_k: int = 20, threshold: float = 0.2) -> Dict[str, Any]:
    url = f"{BACKEND}/api/insights"
    data = {
        "selection": selection,
        "top_k": str(top_k),
        "threshold": str(threshold),
        "stream": "false",
    }
    resp = requests.post(url, data=data, timeout=600)
    resp.raise_for_status()
    return resp.json()



def run_tests(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for i, tc in enumerate(cases, 1):
        sel = tc["selection"]
        expect_suffix = tc.get("expect_file_suffix")
        print(f"\n=== Test {i}: {sel[:80]}...")
        try:
            res = run_insights(sel)
            home = res.get("home") or {}
            home_text = home.get("text") or ""
            file_url = (home.get("fileUrl") or "")
            # Evaluate corpus-wide: the best matching section should come from the expected paper
            ok_file = (expect_suffix in file_url) if expect_suffix else True
            passed = ok_file and bool(home_text)
            results.append({
                "selection": sel,
                "passed": passed,
                "ok_file": ok_file,
                "fileUrl": file_url,
                "displayName": home.get("displayName"),
                "heading": home.get("heading"),
            })
            status = "PASS" if passed else "FAIL"
            print(f"Result: {status}")
            print(f"Home: {home.get('displayName')} | Heading: {home.get('heading')}")
            if not passed:
                print(f"File URL: {file_url}")
                print("Home text sample:")
                print((home_text[:500] + '...') if len(home_text) > 500 else home_text)
        except Exception as e:
            print("Error:", e)
            results.append({"selection": sel, "passed": False, "error": str(e)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading PDFs if already present")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingest step")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    if not args.skip_download:
        paths = ensure_downloads()
        print(f"Prepared {len(paths)} PDFs in {WORKDIR}")
    else:
        paths = [os.path.join(WORKDIR, it["name"]) for it in KNOWN_PDFS]

    if not args.skip_ingest:
        print("Ingesting PDFs...")
        ing = ingest(paths)
        print("Ingest:", ing)

    print("Running tests...")
    results = run_tests(TEST_CASES)
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)

    print(f"\nSummary: {passed}/{total} passed")
    # Emit JSON for machine reading if needed
    print(json.dumps({"passed": passed, "total": total, "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
