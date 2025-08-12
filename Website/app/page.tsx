"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";

const PDFViewer = dynamic(() => import("@/components/PDFViewer"), { ssr: false });

type UploadedFile = { name: string; url: string };

export default function Page() {
  const [uploaded, setUploaded] = useState<UploadedFile[]>([]);
  const [selectedUrl, setSelectedUrl] = useState<string | null>(null);
  const [freshFile, setFreshFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const loadList = async () => {
    const res = await fetch("/api/list", { cache: "no-store" });
    if (res.ok) {
      const data = (await res.json()) as UploadedFile[];
      setUploaded(data);
    }
  };

  useEffect(() => {
    void loadList();
  }, []);

  const onUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    const res = await fetch("/api/upload", { method: "POST", body: form });
    if (res.ok) {
      await loadList();
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const onOpenFresh = (file: File | null) => {
    setSelectedUrl(null);
    setFreshFile(file);
  };

  const onSelectUploaded = (url: string) => {
    setFreshFile(null);
    setSelectedUrl(url);
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="section">
          <h3>Bulk upload PDFs</h3>
          <input ref={inputRef} type="file" accept="application/pdf" multiple onChange={(e) => onUpload(e.target.files)} />
          <p className="small">Uploaded documents</p>
          <div>
            {uploaded.length === 0 && <div className="small">No files yet</div>}
            {uploaded.map((f) => (
              <div key={f.url} className="fileItem" onClick={() => onSelectUploaded(f.url)} title={f.name}>
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 220 }}>{f.name}</span>
                <span className="small">View</span>
              </div>
            ))}
          </div>
        </div>

        <div className="section">
          <h3>Open fresh PDF</h3>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => onOpenFresh(e.target.files?.[0] ?? null)}
          />
          <p className="small">Opens without uploading; ideal for a first-time document</p>
        </div>
      </aside>

      <main className="viewer">
        <PDFViewer url={selectedUrl ?? undefined} file={freshFile ?? undefined} />
      </main>
    </div>
  );
}


