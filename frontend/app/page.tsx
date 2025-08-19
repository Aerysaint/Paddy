"use client";
import { useEffect, useMemo, useState } from "react";
import AdobeViewer from "../components/AdobeViewer";
import SplitPane from "../components/SplitPane";

export default function Page() {
  const [currentUrl, setCurrentUrl] = useState<string | undefined>();
  const [ingesting, setIngesting] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  type LibItem = { file: string; url: string; displayName: string };
  const [library, setLibrary] = useState<{ library: LibItem[]; current: LibItem[] }>({ library: [], current: [] });
  const [loadingLib, setLoadingLib] = useState(false);
  // Multi-subtab state
  type Pane = { id: string; title: string; url: string; page?: number; snippet?: string };
  const [panes, setPanes] = useState<Pane[]>([]);
  const addPane = (url: string, title?: string, page?: number, snippet?: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    setPanes((prev) => {
      // If no panes yet and a currentUrl is being viewed, seed the split layout
      // with the current PDF as the first pane so opening creates a true split.
      if (prev.length === 0 && currentUrl) {
        const baseId = `base-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
        const baseTitle = currentUrl.split("/").pop() || "PDF";
        return [
          { id: baseId, title: baseTitle, url: currentUrl },
          { id, title: title || (url.split("/").pop() || "PDF"), url, page, snippet }
        ];
      }
      return [...prev, { id, title: title || (url.split("/").pop() || "PDF"), url, page, snippet }];
    });
  };
  const closePane = (id: string) => setPanes((prev) => prev.filter((p) => p.id !== id));

  async function refreshLibrary() {
    setLoadingLib(true);
    try {
      const res = await fetch("http://localhost:8000/api/library");
      const json = await res.json();
      setLibrary({ library: json.library || [], current: json.current || [] });
    } catch (e) {
      // ignore
    } finally {
      setLoadingLib(false);
    }
  }

  useEffect(() => {
    refreshLibrary();
  }, []);

  // Listen for open events from AdobeViewer instances that do not receive onOpenPdf
  useEffect(() => {
    function onOpenEvt(e: Event) {
      const ce = e as CustomEvent<{ url: string; title?: string; page?: number; snippet?: string }>;
      const { url, title, page, snippet } = ce.detail || ({} as any);
      if (url) addPane(url, title, page, snippet);
    }
    window.addEventListener('adobe:open-pdf', onOpenEvt as EventListener);
    return () => window.removeEventListener('adobe:open-pdf', onOpenEvt as EventListener);
  }, []);

  async function uploadLibrary(files: FileList | null) {
    if (!files || files.length === 0) return;
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    setIngesting(true);
    try {
      await fetch("http://localhost:8000/api/ingest", { method: "POST", body: fd });
    } finally {
      setIngesting(false);
  refreshLibrary();
    }
  }

  async function uploadCurrent(file: File | null) {
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("http://localhost:8000/api/current", { method: "POST", body: fd });
    const json = await res.json();
    const url = `http://localhost:8000${json.url}`;
    setCurrentUrl(url);
    // Also open as a subtab
    addPane(url, file.name);
    refreshLibrary();
  }

  return (
    <div style={{ display: "grid", gridTemplateRows: "auto 1fr", height: "100vh" }}>
      <div style={{ padding: 12, background: "#0b0f14", borderBottom: "1px solid rgba(255,255,255,0.08)", display: "flex", gap: 12, alignItems: 'center' }}>
        <div style={{ fontWeight: 700 }}>PDF Selection-to-Search</div>
        <button
          onClick={() => setSidebarOpen((s) => !s)}
          style={{ marginLeft: 12, background: '#111827', color: '#cbd5e1', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, padding: '6px 10px', cursor: 'pointer' }}
          title={sidebarOpen ? 'Hide library' : 'Show library'}
        >
          ☰ Library
        </button>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 12 }}>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: '#111827', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10, padding: '8px 12px', cursor: 'pointer' }}>
            <span>Bulk Ingest PDFs</span>
            <input type="file" accept="application/pdf" multiple style={{ display: 'none' }} onChange={(e) => uploadLibrary(e.target.files)} />
          </label>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: '#111827', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10, padding: '8px 12px', cursor: 'pointer' }}>
            <span>Set Current PDF</span>
            <input type="file" accept="application/pdf" style={{ display: 'none' }} onChange={(e) => uploadCurrent(e.target.files?.[0] ?? null)} />
          </label>
          {ingesting && <span style={{ opacity: 0.8 }}>Indexing…</span>}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: sidebarOpen ? '280px 1fr' : '1fr', minHeight: 0 }}>
        {sidebarOpen && (
          <aside style={{ borderRight: '1px solid rgba(255,255,255,0.08)', background: '#0b0f14', padding: 10, overflow: 'auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <div style={{ fontWeight: 700, color: '#cbd5e1' }}>Library</div>
              <button onClick={refreshLibrary} style={{ background: '#111827', color: '#cbd5e1', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 6, padding: '4px 8px', cursor: 'pointer' }}>
                {loadingLib ? '...' : 'Refresh'}
              </button>
            </div>
            {library.current.length > 0 && (
              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 4 }}>Current</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {library.current.map((it) => {
                    const url = `http://localhost:8000${it.url}`;
                    const active = currentUrl === url;
                    return (
                      <button
                        key={it.url}
                        onClick={() => {
                          // Make this the current PDF and switch to single-view mode
                          setPanes([]);
                          setCurrentUrl(url);
                        }}
                        style={{ textAlign: 'left', background: active ? 'rgba(96,165,250,0.15)' : 'transparent', color: '#e5e7eb', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: '8px', cursor: 'pointer' }}
                        title={it.displayName}
                      >
                        {it.displayName}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
            <div>
              <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 4 }}>All PDFs</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {library.library.length === 0 ? (
                  <div style={{ opacity: 0.6, fontSize: 12 }}>No PDFs ingested yet.</div>
                ) : (
                  library.library.map((it) => {
                    const url = `http://localhost:8000${it.url}`;
                    const active = currentUrl === url;
                    return (
                      <button
                        key={it.url}
                        onClick={() => {
                          setPanes([]);
                          setCurrentUrl(url);
                        }}
                        style={{ textAlign: 'left', background: active ? 'rgba(96,165,250,0.15)' : 'transparent', color: '#e5e7eb', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: '8px', cursor: 'pointer' }}
                        title={it.displayName}
                      >
                        {it.displayName}
                      </button>
                    );
                  })
                )}
              </div>
            </div>
          </aside>
        )}
        <div style={{ minHeight: 0 }}>
          {panes.length === 0 ? (
            currentUrl ? (
              <AdobeViewer currentUrl={currentUrl} onOpenPdf={(url, title, page, snippet) => addPane(url, title, page, snippet)} />
            ) : (
              <div style={{ height: '100%', display: 'grid', placeItems: 'center', color: '#9ca3af' }}>
                Upload a current PDF to start.
              </div>
            )
          ) : (
            <SplitPane
              panes={panes.map((p) => ({
                id: p.id,
                title: p.title,
                content: (
                  <AdobeViewer
                    key={p.id}
                    currentUrl={p.url}
                    initialPage={p.page}
                    initialHighlight={p.snippet}
                    onOpenPdf={(url, title, page, snippet) => addPane(url, title, page, snippet)}
                  />
                )
              }))}
              onClose={closePane}
            />
          )}
        </div>
      </div>
    </div>
  );
}
