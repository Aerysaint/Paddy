import os
import sys
import time
import json
import threading
from typing import List, Dict, Any

import requests

# Simple harness to:
# 1) Search arXiv for PDFs by topic
# 2) Download a few PDFs
# 3) Ingest them via local FastAPI backend
# 4) Run insights on a few heuristic selections to check relevant sections
#
# Assumptions:
# - Backend server running at BACKEND_ORIGIN (default http://localhost:8000)
# - FRONTEND_ORIGIN used only to construct links in responses
# - No changes to API shape
#
# Usage (PowerShell):
#   python backend/tests/retrieval_harness.py --topic "large language models" --n 3
#
# If running from repo root, this path should work; else adjust.

import argparse

ARXIV_SEARCH = "https://export.arxiv.org/api/query"
DEFAULT_BACKEND = os.environ.get("BACKEND_ORIGIN", "http://localhost:8000")


def search_arxiv(topic: str, n: int = 3) -> List[Dict[str, Any]]:
    import feedparser
    query = f"search_query=all:{topic.replace(' ', '+')}&start=0&max_results={n}"
    url = f"{ARXIV_SEARCH}?{query}"

    feed = feedparser.parse(url)
    results = []
    for e in feed.entries:
        # Find PDF link among links
        pdf_url = None
        for l in e.get('links', []):
            if l.get('type') == 'application/pdf':
                pdf_url = l.get('href')
                break
        if not pdf_url:
            continue
        results.append({
            'title': e.get('title', 'untitled'),
            'pdf_url': pdf_url,
            'id': e.get('id'),
            'summary': e.get('summary', ''),
        })
    return results


def download_pdf(url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fn = url.split('/')[-1]
    if not fn.lower().endswith('.pdf'):
        fn += '.pdf'
    out_path = os.path.join(out_dir, fn)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        f.write(r.content)
    return out_path


def ingest_pdfs(paths: List[str]) -> Dict[str, Any]:
    url = f"{DEFAULT_BACKEND}/api/ingest"
    files = [('files', (os.path.basename(p), open(p, 'rb'), 'application/pdf')) for p in paths]
    try:
        resp = requests.post(url, files=files, timeout=300)
        return resp.json()
    finally:
        for _, (name, fh, _) in files:
            try:
                fh.close()
            except Exception:
                pass


def run_insights(selection: str, stream: bool = False, top_k: int = 20, threshold: float = 0.2) -> Dict[str, Any]:
    url = f"{DEFAULT_BACKEND}/api/insights"
    data = {
        'selection': selection,
        'top_k': str(top_k),
        'threshold': str(threshold),
        'stream': 'true' if stream else 'false',
    }
    if stream:
        # Stream meta + analysis; collect text
        with requests.post(url, data=data, stream=True, timeout=300) as r:
            r.raise_for_status()
            meta = None
            analysis_parts = []
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith('event: meta'):
                    # next data line contains meta
                    continue
                if line.startswith('data: '):
                    payload = line[len('data: '):]
                    if payload == '[DONE]':
                        break
                    try:
                        obj = json.loads(payload)
                        if isinstance(obj, dict) and meta is None and 'home' in obj and 'related' in obj:
                            meta = obj
                        else:
                            analysis_parts.append(payload)
                    except Exception:
                        analysis_parts.append(payload)
            return {
                'meta': meta,
                'analysis': ''.join(analysis_parts)
            }
    else:
        resp = requests.post(url, data=data, timeout=300)
        resp.raise_for_status()
        return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, required=True, help='Topic to search on arXiv')
    parser.add_argument('--n', type=int, default=3, help='Number of PDFs to fetch')
    parser.add_argument('--workdir', type=str, default='backend/tests/_tmp_arxiv', help='Working dir for downloaded PDFs')
    parser.add_argument('--stream', action='store_true', help='Use streaming insights')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()

    print(f"Searching arXiv for '{args.topic}' ...")
    try:
        results = search_arxiv(args.topic, args.n)
    except Exception as e:
        print(f"Search failed: {e}")
        sys.exit(1)

    if not results:
        print('No results found from arXiv search.')
        sys.exit(1)

    print(f"Found {len(results)} PDFs. Downloading...")
    pdf_paths = []
    for r in results:
        try:
            p = download_pdf(r['pdf_url'], args.workdir)
            pdf_paths.append(p)
            print(f"Downloaded: {p}")
        except Exception as e:
            print(f"Download failed for {r.get('pdf_url')}: {e}")

    if not pdf_paths:
        print('No PDFs downloaded; aborting.')
        sys.exit(1)

    print('Ingesting PDFs into backend...')
    ing = ingest_pdfs(pdf_paths)
    print('Ingest response:', ing)

    # Create a few heuristic selections from the arXiv summaries to probe retrieval
    selections = []
    for r in results[:min(3, len(results))]:
        summ = r.get('summary', '')
        if not summ:
            continue
        # take a mid-length sentence as selection
        parts = [s.strip() for s in summ.split('.') if len(s.split()) >= 6][:3]
        if parts:
            selections.append(parts[0])
    if not selections:
        # fallback: topic itself
        selections = [args.topic]

    print('\nRunning insights on sample selections...')
    for i, sel in enumerate(selections, 1):
        print(f"\n=== Selection {i} ===\n{sel}\n")
        try:
            res = run_insights(sel, stream=False, top_k=args.top_k, threshold=args.threshold)
            # Print a concise summary
            home = res.get('home')
            related = res.get('related', [])
            analysis = res.get('analysis')
            print('Home:', (home or {}).get('displayName'), '| Heading:', (home or {}).get('heading'))
            for j, rs in enumerate(related, 1):
                print(f"Rel-{j}:", rs.get('displayName'), '| Heading:', rs.get('heading'))
            print('\nAnalysis:\n', (analysis or '')[:400], '...')
        except Exception as e:
            print('Insights failed:', e)

    print('\nDone.')


if __name__ == '__main__':
    main()
