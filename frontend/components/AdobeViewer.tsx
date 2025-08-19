"use client";
import { useEffect, useRef, useState } from "react";

// Global from Adobe viewer.js
declare global {
  interface Window {
    AdobeDC?: any;
    adobe_dc_view_sdk?: any;
  }
}

export type SearchHit = {
  score: number;
  file: string;
  fileUrl: string;
  page: number;
  heading?: string | null;
  text?: string;
};

// Remove repeated/overlapping results by normalizing snippet text and keeping highest score
function dedupeHits(hits: SearchHit[], limit = 5): SearchHit[] {
  const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();
  // Sort by score desc so we keep best when deduping
  const sorted = [...hits].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const seen = new Set<string>();
  const out: SearchHit[] = [];
  for (const h of sorted) {
    const keyText = h.text ? norm(h.text) : "";
    const key = keyText || norm(`${h.file}|${h.page}|${h.heading ?? ""}`);
    if (key.length === 0) continue;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(h);
    if (out.length >= limit) break;
  }
  return out;
}

export default function AdobeViewer({ currentUrl, onOpenPdf, initialPage, initialHighlight }: { currentUrl?: string; onOpenPdf?: (url: string, title?: string, page?: number, snippet?: string) => void; initialPage?: number; initialHighlight?: string }) {
  const divRef = useRef<HTMLDivElement | null>(null);
  const [divId] = useState(() => `adobe-dc-view-${Math.random().toString(36).slice(2)}`);
  const [adobeViewer, setAdobeViewer] = useState<any>(null);
  const [viewerError, setViewerError] = useState<string | null>(null);
  const [hasSelection, setHasSelection] = useState(false);
  const selectionRef = useRef<string>("");
  const pollRef = useRef<number | null>(null);
  const lastQueriedRef = useRef<string>("");

  // Stacked mini viewers panel state
  const [panelOpen, setPanelOpen] = useState(false);
  const [insightsOpen, setInsightsOpen] = useState(false);
  const [hitsState, setHitsState] = useState<SearchHit[] | null>(null);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [panelError, setPanelError] = useState<string | null>(null);
  const [miniIds, setMiniIds] = useState<string[]>([]); // per-hit container ids (state so DOM updates)
  const miniViewsRef = useRef<any[]>([]); // per-hit AdobeDC.View
  const miniViewersRef = useRef<any[]>([]); // per-hit viewer objects
  const curPagesRef = useRef<number[]>([]); // per-hit current page tracker
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsError, setInsightsError] = useState<string | null>(null);
  const [insightsData, setInsightsData] = useState<any | null>(null);
  const [insightsText, setInsightsText] = useState<string>("");
  const [streaming, setStreaming] = useState<boolean>(false);
  // Audio synthesis state
  const [audioLoading, setAudioLoading] = useState<boolean>(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [audioData, setAudioData] = useState<{ audioUrl: string; script: string; mode: string } | null>(null);

  useEffect(() => {
    setViewerError(null);
    if (!divRef.current) return;

    const scriptId = "adobe-viewer-sdk";
    if (!document.getElementById(scriptId)) {
      const s = document.createElement("script");
      s.id = scriptId;
      s.src = "https://acrobatservices.adobe.com/view-sdk/viewer.js";
      s.onerror = () => setViewerError("Failed to load Adobe viewer script.");
      document.body.appendChild(s);
    }

    async function init() {
      try {
        if (!window.AdobeDC || !currentUrl) return;
        const clientId = process.env.NEXT_PUBLIC_ADOBE_CLIENT_ID;
        if (!clientId) {
          setViewerError("Missing NEXT_PUBLIC_ADOBE_CLIENT_ID. Set it in your environment.");
          return;
        }
  const view = new window.AdobeDC.View({ clientId, divId });
        const previewConfig = {
          embedMode: "FULL_WINDOW",
          showDownloadPDF: false,
          showZoomControl: true,
          enableSearchAPIs: true
        } as const;
  const previewPromise = view.previewFile(
          {
            content: { location: { url: currentUrl } },
            metaData: { fileName: currentUrl.split("/").pop() }
          },
          previewConfig
        );

        previewPromise
          .then((viewer: any) => {
            setAdobeViewer(viewer);
            // After viewer ready, optionally navigate to a page and highlight selection
            const afterReady = async () => {
              try {
                const apis = await viewer.getAPIs();
                if (initialPage && Number.isFinite(initialPage)) {
                  const p = Math.max(1, Math.floor(initialPage));
                  await apis.gotoLocation(p, 0, 0).catch(() => {});
                }
                if (initialHighlight && initialHighlight.trim()) {
                  // Use search API to highlight the snippet; requires enableSearchAPIs
                  try {
                    const reduce = (s: string) => {
                      const clean = s.replace(/\s+/g, ' ').trim();
                      const words = clean.split(' ');
                      return words.slice(0, 20).join(' ');
                    };
                    const needle = reduce(initialHighlight);
                    const searchObj = await apis.search(needle);
                    await searchObj
                      .onResultsUpdate((sr: any) => {
                        // When search completes and we have results, navigate to the first highlight
                        if (sr && sr.status === 'COMPLETED' && sr.totalResults > 0) {
                          return searchObj.next().then(() => true).catch(() => true);
                        }
                        return Promise.resolve(true);
                      })
                      .catch(() => {});
                  } catch {}
                }
              } catch {}
            };
            afterReady();
            // Poll for selection via supported API (iframe blocks host mouse events)
      const startPolling = async () => {
              try {
                const apis = await viewer.getAPIs();
                // Clear any old timer first
                if (pollRef.current) {
                  window.clearInterval(pollRef.current);
                  pollRef.current = null;
                }
        pollRef.current = window.setInterval(async () => {
                  try {
                    const sel = await apis.getSelectedContent();
                    const data = (sel?.data || "").trim();
                    if (data) {
                      // track current selection
                      if (data !== selectionRef.current) {
                        selectionRef.current = data;
            // Do not auto-search; just show the action button
                      }
                      setHasSelection(true);
                    } else {
                      selectionRef.current = "";
                      setHasSelection(false);
                    }
                  } catch {
                    // ignore polling errors
                  }
                }, 400);
              } catch (err) {
                console.warn("Failed to initialize selection polling", err);
              }
            };
            startPolling();
          })
          .catch((e: any) => {
            console.error("Adobe previewFile failed", e);
            setViewerError("Adobe viewer failed to open the PDF. Check allowed domains and client ID.");
          });
      } catch (err: any) {
        console.error("Adobe init error", err);
        setViewerError("Failed to initialize Adobe viewer.");
      }
    }

    const ready = () => init();
    document.addEventListener("adobe_dc_view_sdk.ready", ready);
    const t = setTimeout(init, 1200);

    return () => {
      document.removeEventListener("adobe_dc_view_sdk.ready", ready);
      clearTimeout(t);
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [currentUrl]);

  async function doSearch(q: string): Promise<SearchHit[]> {
    const fd = new FormData();
    fd.append("query", q);
    fd.append("k", "5");
    const res = await fetch("http://localhost:8000/api/search", { method: "POST", body: fd });
    const json = await res.json();
  const raw = (json?.results || []) as SearchHit[];
  return dedupeHits(raw, 5);
  }

  // Helper to trim snippet for reliable highlighting
  const reduceSnippet = (s: string) => {
    const clean = s.replace(/\s+/g, ' ').trim();
    const words = clean.split(' ');
    return words.slice(0, 20).join(' ');
  };

  // Wait for an element by ID to exist and have non-zero size
  async function waitForElementById(id: string, timeoutMs = 2000): Promise<HTMLElement | null> {
    const start = Date.now();
    return new Promise((resolve) => {
      const tick = () => {
        const el = document.getElementById(id) as HTMLElement | null;
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.width > 10 && rect.height > 10) {
            resolve(el);
            return;
          }
        }
        if (Date.now() - start > timeoutMs) {
          resolve(el);
        } else {
          requestAnimationFrame(tick);
        }
      };
      tick();
    });
  }

  // Load and render a specific hit into its own mini container
  async function renderHit(i: number) {
    if (!hitsState) return;
    const h = hitsState[i];
    if (!h) return;
    try {
      const id = miniIds[i];
      if (!id) return;
      const el = await waitForElementById(id);
      if (!el) return;
      const clientId = process.env.NEXT_PUBLIC_ADOBE_CLIENT_ID;
      if (!clientId || !window.AdobeDC) return;
      if (!miniViewsRef.current[i]) {
        miniViewsRef.current[i] = new window.AdobeDC.View({ clientId, divId: id });
      }
      const fileUrl = `http://localhost:8000${h.fileUrl}`;
      // Start preview
      miniViewersRef.current[i] = await miniViewsRef.current[i].previewFile(
        {
          content: { location: { url: fileUrl } },
          metaData: { fileName: h.file?.split("/").pop() }
        },
        { embedMode: 'SIZED_CONTAINER', showDownloadPDF: false, showZoomControl: false, enableSearchAPIs: true }
      );
      const apis = await miniViewersRef.current[i].getAPIs();
      const basePage = (h.page && Number.isFinite(h.page)) ? Math.max(1, Math.floor(h.page)) : 1;
      curPagesRef.current[i] = basePage;
      await apis.gotoLocation(basePage, 0, 0).catch(() => {});
      const snippet = (h.text || '').trim();
      if (snippet) {
        try {
          const needle = reduceSnippet(snippet);
          const searchObj = await apis.search(needle);
          await searchObj.onResultsUpdate((sr: any) => {
            if (sr && sr.status === 'COMPLETED' && sr.totalResults > 0) {
              return searchObj.next().then(() => true).catch(() => true);
            }
            return Promise.resolve(true);
          }).catch(() => {});
        } catch {}
      }
    } catch (e) {
      console.warn('Failed to render hit', i, e);
    }
  }

  // When panel opens or hits load, initialize per-hit containers and render all
  useEffect(() => {
    if (!panelOpen || !hitsState || hitsState.length === 0) return;
    // Initialize ids and per-hit trackers
    const ids = hitsState.map((_, i) => `adobe-mini-${i}-${Math.random().toString(36).slice(2)}`);
    setMiniIds(ids);
    miniViewsRef.current = Array(hitsState.length).fill(null);
    miniViewersRef.current = Array(hitsState.length).fill(null);
    curPagesRef.current = hitsState.map(h => (h.page && Number.isFinite(h.page) ? Math.max(1, Math.floor(h.page)) : 1));
    setActiveIndex(0);
  }, [panelOpen, hitsState]);

  // Once IDs are applied to the DOM, render each hit progressively
  useEffect(() => {
    if (!panelOpen || !hitsState || hitsState.length === 0) return;
    if (miniIds.length !== hitsState.length) return;
    (async () => {
      const initialBatch = Math.min(3, hitsState.length);
      for (let i = 0; i < initialBatch; i++) {
        await renderHit(i);
      }
      // queue the rest without blocking UI
      setTimeout(async () => {
        for (let i = initialBatch; i < hitsState.length; i++) {
          await renderHit(i);
        }
      }, 0);
    })();
  }, [miniIds, panelOpen, hitsState]);

  return (
  <div style={{ position: "relative", height: "100%" }}>
      {!currentUrl && (
        <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: '#9ca3af' }}>
          Upload a current PDF to start.
        </div>
      )}
  <div id={divId} ref={divRef} style={{ height: "100%" }} />
      {viewerError && (
        <div style={{ position: 'absolute', top: 12, left: 12, background: '#1f2937', color: '#fecaca', border: '1px solid #b91c1c', borderRadius: 8, padding: '8px 12px' }}>
          {viewerError}
        </div>
      )}
  {hasSelection && (
        <div style={{ position: 'absolute', right: 16, bottom: 16, display: 'flex', gap: 8, zIndex: 9998 }}>
          <button
            onClick={async () => {
              const q = selectionRef.current.trim();
              if (!q) return;
              lastQueriedRef.current = q;
              const hits = await doSearch(q);
              setHitsState(hits);
              setInsightsOpen(false);
              setPanelOpen(true);
            }}
            style={{
              background: 'linear-gradient(135deg, #0ea5e9, #2563eb)',
              color: 'white',
              border: 'none',
              borderRadius: 20,
              padding: '10px 14px',
              fontWeight: 600,
              boxShadow: '0 6px 20px rgba(14,165,233,0.4)',
              cursor: 'pointer',
            }}
            title="Find related passages for selected text"
          >
            Related
          </button>
          <button
            onClick={async () => {
              const sel = selectionRef.current.trim();
              if (!sel) return;
              setPanelOpen(false);
              setInsightsOpen(true);
              setInsightsError(null);
              setInsightsLoading(true);
              setInsightsText("");
              try {
                const fd = new FormData();
                fd.append('selection', sel);
                fd.append('top_k', '20');
                fd.append('threshold', '0.2');
                if (currentUrl) fd.append('current_url', currentUrl);
                // Try SSE streaming first
                setStreaming(true);
                const url = 'http://localhost:8000/api/insights';
                const controller = new AbortController();
                const params = new URLSearchParams();
                // Use fetch with text/event-stream emulation via ReadableStream
                fd.append('stream', 'true');
                const resp = await fetch(url, { method: 'POST', body: fd, signal: controller.signal });
                if (resp.ok && resp.headers.get('content-type')?.includes('text/event-stream')) {
                  const reader = resp.body!.getReader();
                  const dec = new TextDecoder();
                  let buffer = '';
                  const pump = async () => {
                    const { value, done } = await reader.read();
                    if (done) return;
                    buffer += dec.decode(value, { stream: true });
                    // Split on double newlines per SSE
                    const parts = buffer.split('\n\n');
                    buffer = parts.pop() || '';
                    for (const part of parts) {
                      const lines = part.split('\n');
                      let ev = 'message';
                      let data = '';
                      for (const line of lines) {
                        if (line.startsWith('event:')) ev = line.slice(6).trim();
                        else if (line.startsWith('data:')) data += line.slice(5).trim();
                      }
                      if (ev === 'meta') {
                        try { setInsightsData(JSON.parse(data)); } catch {}
                      } else if (data === '[DONE]') {
                        setStreaming(false);
                        return;
                      } else if (data) {
                        setInsightsText((prev) => prev + data);
                      }
                    }
                    await pump();
                  };
                  await pump();
                } else {
                  // Fallback non-stream JSON
                  const json = await resp.json();
                  if (!resp.ok) throw new Error(json?.error || 'Failed to get insights');
                  setInsightsData(json);
                  setInsightsText(json.analysis || '');
                }
              } catch (e: any) {
                setInsightsError(e?.message || 'Failed to get insights');
              } finally {
                setInsightsLoading(false);
                setStreaming(false);
              }
            }}
            style={{
              background: 'linear-gradient(135deg, #22c55e, #16a34a)',
              color: 'white',
              border: 'none',
              borderRadius: 20,
              padding: '10px 14px',
              fontWeight: 600,
              boxShadow: '0 6px 20px rgba(34,197,94,0.4)',
              cursor: 'pointer',
            }}
            title="Generate grounded insights for the selected text"
          >
            Insights
          </button>
        </div>
      )}
      {/* Floating stacked mini PDF windows for results */}
      {panelOpen && (
        <div
          style={{
            position: 'fixed',
            right: 16,
            bottom: 70,
            width: 540,
            height: 520,
            background: 'rgba(17,25,40,0.92)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 14,
            boxShadow: '0 10px 40px rgba(0,0,0,0.5)',
            color: '#e6edf3',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 9999
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
            <div style={{ fontWeight: 600, flex: 1 }}>Related results</div>
            <button
              onClick={() => setPanelOpen(false)}
              style={{ background: 'transparent', color: '#fca5a5', border: '1px solid rgba(252,165,165,0.4)', borderRadius: 8, padding: '6px 8px', cursor: 'pointer' }}
              title="Close panel"
            >
              ✕
            </button>
          </div>
          <div style={{ position: 'relative', flex: 1, overflow: 'auto', padding: 10, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {panelError && (
              <div style={{ background: '#1f2937', color: '#fecaca', border: '1px solid #b91c1c', borderRadius: 8, padding: '6px 10px' }}>{panelError}</div>
            )}
            {hitsState && hitsState.length > 0 ? (
              hitsState.map((h, i) => (
                <div key={miniIds[i] || i} style={{ border: `2px solid ${activeIndex === i ? '#60a5fa' : 'rgba(255,255,255,0.08)'}`, borderRadius: 10, background: 'rgba(0,0,0,0.2)' }}>
                  <div
                    onClick={() => setActiveIndex(i)}
                    style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 8px', borderBottom: '1px solid rgba(255,255,255,0.06)', cursor: 'pointer' }}
                  >
                    <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      <span style={{ fontWeight: 600, color: '#7dd3fc' }}>{h.file?.split('_').slice(1).join('_') || h.file}</span>
                      <span style={{ opacity: 0.8, marginLeft: 8 }}>{h.heading || `Page ${h.page}`}</span>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      {activeIndex === i && (
                        <>
                          <button
                            onClick={async (e) => {
                              e.stopPropagation();
                              const viewer = miniViewersRef.current[i];
                              if (!viewer) return;
                              try {
                                const apis = await viewer.getAPIs();
                                const nextPage = (curPagesRef.current[i] || 1) - 1;
                                if (nextPage >= 1) {
                                  await apis.gotoLocation(nextPage, 0, 0).catch(() => {});
                                  curPagesRef.current[i] = nextPage;
                                }
                              } catch {}
                            }}
                            style={{ background: 'transparent', color: '#cbd5e1', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, padding: '4px 6px', cursor: 'pointer' }}
                            title="Previous page"
                          >
                            ◀
                          </button>
                          <button
                            onClick={async (e) => {
                              e.stopPropagation();
                              const viewer = miniViewersRef.current[i];
                              if (!viewer) return;
                              try {
                                const apis = await viewer.getAPIs();
                                const nextPage = (curPagesRef.current[i] || 1) + 1;
                                await apis.gotoLocation(nextPage, 0, 0).catch(() => {});
                                curPagesRef.current[i] = nextPage;
                              } catch {}
                            }}
                            style={{ background: 'transparent', color: '#cbd5e1', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, padding: '4px 6px', cursor: 'pointer' }}
                            title="Next page"
                          >
                            ▶
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              const url = `http://localhost:8000${h.fileUrl}`;
                              const page = h.page;
                              const snippet = (h.text || '').slice(0, 200);
                              const title = h.file?.split('_').slice(1).join('_') || h.file;
                              if (onOpenPdf) {
                                onOpenPdf(url, title, page, snippet);
                              } else {
                                // Broadcast an app-level event so the main page can open a split pane
                                try {
                                  window.dispatchEvent(new CustomEvent('adobe:open-pdf', { detail: { url, title, page, snippet } }));
                                } catch {}
                              }
                            }}
                            style={{ background: 'transparent', color: '#93c5fd', border: '1px solid rgba(147,197,253,0.4)', borderRadius: 8, padding: '4px 8px', cursor: 'pointer' }}
                            title="Open in split tab"
                          >
                            Open
                          </button>
                          <button
                            onClick={async (e) => {
                              e.stopPropagation();
                              const viewer = miniViewersRef.current[i];
                              if (!viewer) return;
                              try {
                                const apis = await viewer.getAPIs();
                                const snippet = (h.text || '').trim();
                                if (snippet) {
                                  const needle = reduceSnippet(snippet);
                                  const searchObj = await apis.search(needle);
                                  await searchObj.onResultsUpdate((sr: any) => {
                                    if (sr && sr.status === 'COMPLETED' && sr.totalResults > 0) {
                                      return searchObj.next().then(() => true).catch(() => true);
                                    }
                                    return Promise.resolve(true);
                                  }).catch(() => {});
                                }
                              } catch {}
                            }}
                            style={{ background: 'transparent', color: '#a7f3d0', border: '1px solid rgba(167,243,208,0.4)', borderRadius: 8, padding: '4px 8px', cursor: 'pointer' }}
                            title="Re-highlight snippet"
                          >
                            Highlight
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                  <div style={{ position: 'relative', height: 200 }} onClick={() => setActiveIndex(i)}>
                    <div id={miniIds[i]} style={{ position: 'absolute', inset: 0 }} />
                  </div>
                </div>
              ))
            ) : (
              <div style={{ opacity: 0.7 }}>No related passages found.</div>
            )}
          </div>
        </div>
      )}

      {/* Insights panel */}
      {insightsOpen && (
        <div
          style={{
            position: 'fixed',
            right: 16,
            bottom: 70,
            width: 640,
            height: 560,
            background: 'rgba(17,25,40,0.96)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 14,
            boxShadow: '0 10px 40px rgba(0,0,0,0.5)',
            color: '#e6edf3',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 9999
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
            <div style={{ fontWeight: 600, flex: 1 }}>Insights</div>
            {/* Audio actions */}
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                onClick={async () => {
                  const sel = selectionRef.current.trim();
                  if (!sel) return;
                  setAudioError(null);
                  setAudioData(null);
                  setAudioLoading(true);
                  try {
                    const fd = new FormData();
                    fd.append('selection', sel);
                    fd.append('mode', 'podcast');
                    const resp = await fetch('http://localhost:8000/api/audio', { method: 'POST', body: fd });
                    const json = await resp.json();
                    if (!resp.ok) throw new Error(json?.error || 'Failed to generate podcast');
                    setAudioData(json);
                  } catch (e: any) {
                    setAudioError(e?.message || 'Failed to generate podcast');
                  } finally {
                    setAudioLoading(false);
                  }
                }}
                style={{ background: 'transparent', color: '#a7f3d0', border: '1px solid rgba(167,243,208,0.4)', borderRadius: 8, padding: '6px 8px', cursor: 'pointer' }}
                title="Generate a 2–5 min podcast"
              >
                Audio Podcast
              </button>
              <button
                onClick={async () => {
                  const sel = selectionRef.current.trim();
                  if (!sel) return;
                  setAudioError(null);
                  setAudioData(null);
                  setAudioLoading(true);
                  try {
                    const fd = new FormData();
                    fd.append('selection', sel);
                    fd.append('mode', 'overview');
                    const resp = await fetch('http://localhost:8000/api/audio', { method: 'POST', body: fd });
                    const json = await resp.json();
                    if (!resp.ok) throw new Error(json?.error || 'Failed to generate audio overview');
                    setAudioData(json);
                  } catch (e: any) {
                    setAudioError(e?.message || 'Failed to generate audio overview');
                  } finally {
                    setAudioLoading(false);
                  }
                }}
                style={{ background: 'transparent', color: '#fde68a', border: '1px solid rgba(253,230,138,0.4)', borderRadius: 8, padding: '6px 8px', cursor: 'pointer' }}
                title="Generate a 2–3 min overview"
              >
                Audio Overview
              </button>
            </div>
            <button
              onClick={() => setInsightsOpen(false)}
              style={{ background: 'transparent', color: '#fca5a5', border: '1px solid rgba(252,165,165,0.4)', borderRadius: 8, padding: '6px 8px', cursor: 'pointer' }}
              title="Close panel"
            >
              ✕
            </button>
          </div>
          <div style={{ position: 'relative', flex: 1, overflow: 'auto', padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {insightsLoading && <div style={{ opacity: 0.8 }}>Analyzing…</div>}
            {insightsError && (
              <div style={{ background: '#1f2937', color: '#fecaca', border: '1px solid #b91c1c', borderRadius: 8, padding: '6px 10px' }}>{insightsError}</div>
            )}
            {/* Audio status and player */}
            {audioLoading && <div style={{ opacity: 0.9 }}>Generating audio…</div>}
            {audioError && (
              <div style={{ background: '#1f2937', color: '#fecaca', border: '1px solid #b91c1c', borderRadius: 8, padding: '6px 10px' }}>{audioError}</div>
            )}
            {audioData && (
              <div style={{ background: '#0b0f14', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 8, padding: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{ fontWeight: 600 }}>Audio {audioData.mode === 'podcast' ? 'Podcast' : 'Overview'}</div>
                <audio controls src={`http://localhost:8000${audioData.audioUrl}`} style={{ width: '100%' }} />
                <details>
                  <summary style={{ cursor: 'pointer' }}>Show script</summary>
                  <div style={{ marginTop: 6 }}>
                    {(() => {
                      const md: string = audioData.script || '';
                      const html = md
                        .replace(/&/g, '&amp;')
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;')
                        .replace(/\n/g, '<br/>');
                      return <div dangerouslySetInnerHTML={{ __html: html }} />;
                    })()}
                  </div>
                </details>
              </div>
            )}
            {insightsData && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, minHeight: 0 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10, minHeight: 0 }}>
                  <div style={{ fontWeight: 600 }}>Analysis</div>
                  <div
                    onClick={(e) => {
                      // Intercept links to open inside app split panes instead of new tab
                      const target = e.target as HTMLElement;
                      if (target && target.tagName === 'A') {
                        e.preventDefault();
                        const href = (target as HTMLAnchorElement).href;
                        try {
                          const u = new URL(href);
                          if (u.pathname.endsWith('/viewer')) {
                            const url = u.searchParams.get('url') || '';
                            const page = Number(u.searchParams.get('page') || '') || undefined;
                            const q = u.searchParams.get('q') || undefined;
                            const title = decodeURIComponent(url.split('/').pop() || 'PDF');
                            if (onOpenPdf) onOpenPdf(decodeURIComponent(url), title, page, q);
                            else window.dispatchEvent(new CustomEvent('adobe:open-pdf', { detail: { url: decodeURIComponent(url), title, page, snippet: q } }));
                          } else {
                            // If it's a direct file URL, still open via split pane
                            const url = href;
                            const title = decodeURIComponent(url.split('/').pop() || 'PDF');
                            if (onOpenPdf) onOpenPdf(url, title);
                            else window.dispatchEvent(new CustomEvent('adobe:open-pdf', { detail: { url, title } }));
                          }
                        } catch {}
                      }
                    }}
                    style={{ background: '#0b0f14', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 8, padding: 10, minHeight: 140, maxHeight: 300, overflow: 'auto', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace' }}
                  >
                    {/* Minimal markdown: convert [text](url) incrementally */}
                    {(() => {
                      const md: string = insightsText || insightsData?.analysis || 'No analysis.';
                      const html = md
                        .replace(/&/g, '&amp;')
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;')
                        .replace(/\n/g, '<br/>')
                        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
                      return <div dangerouslySetInnerHTML={{ __html: html }} />;
                    })()}
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10, minHeight: 0 }}>
                  <div style={{ fontWeight: 600 }}>Citations</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8, overflow: 'auto' }}>
                    <div style={{ background: '#0b0f14', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 8, padding: 8 }}>
                      <div style={{ fontWeight: 600, color: '#7dd3fc' }}>{insightsData.home?.displayName || insightsData.home?.file}</div>
                      <div style={{ opacity: 0.8 }}>{insightsData.home?.heading || (insightsData.home?.pages ? `Pages ${insightsData.home.pages.join(', ')}` : '')}</div>
                      <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                        <button
                          onClick={() => {
                            const url = `http://localhost:8000${insightsData.home?.fileUrl}`;
                            const title = insightsData.home?.displayName || insightsData.home?.file;
                            const page = Array.isArray(insightsData.home?.pages) ? insightsData.home.pages[0] : undefined;
                            const snippet = (insightsData.home?.text || '').slice(0, 200);
                            if (onOpenPdf) onOpenPdf(url, title, page, snippet);
                            else window.dispatchEvent(new CustomEvent('adobe:open-pdf', { detail: { url, title, page, snippet } }));
                          }}
                          style={{ background: 'transparent', color: '#93c5fd', border: '1px solid rgba(147,197,253,0.4)', borderRadius: 8, padding: '4px 8px', cursor: 'pointer' }}
                        >
                          Open
                        </button>
                      </div>
                    </div>
                    {(insightsData.related || []).map((r: any, i: number) => (
                      <div key={i} style={{ background: '#0b0f14', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 8, padding: 8 }}>
                        <div style={{ fontWeight: 600, color: '#7dd3fc' }}>{r.displayName || r.file}</div>
                        <div style={{ opacity: 0.8 }}>{r.heading || (r.pages ? `Pages ${r.pages.join(', ')}` : '')}</div>
                        <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                          <button
                            onClick={() => {
                              const url = `http://localhost:8000${r.fileUrl}`;
                              const title = r.displayName || r.file;
                              const page = Array.isArray(r.pages) ? r.pages[0] : undefined;
                              const snippet = (r.text || '').slice(0, 200);
                              if (onOpenPdf) onOpenPdf(url, title, page, snippet);
                              else window.dispatchEvent(new CustomEvent('adobe:open-pdf', { detail: { url, title, page, snippet } }));
                            }}
                            style={{ background: 'transparent', color: '#93c5fd', border: '1px solid rgba(147,197,253,0.4)', borderRadius: 8, padding: '4px 8px', cursor: 'pointer' }}
                          >
                            Open
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
