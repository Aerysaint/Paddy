"use client";
import { useMemo } from "react";
import AdobeViewer from "../../components/AdobeViewer";

export default function ViewerPage({ searchParams }: { searchParams: { url?: string; page?: string; q?: string } }) {
  const url = searchParams.url ? decodeURIComponent(searchParams.url) : undefined;
  const initPage = useMemo(() => {
    const p = Number(searchParams.page);
    return Number.isFinite(p) && p > 0 ? p : undefined;
  }, [searchParams.page]);
  const initHighlight = searchParams.q ? decodeURIComponent(searchParams.q) : undefined;

  return (
    <div style={{ height: "100vh" }}>
      {url ? (
        <AdobeViewer currentUrl={url} initialPage={initPage} initialHighlight={initHighlight} />
      ) : (
        <div style={{ height: '100%', display: 'grid', placeItems: 'center', color: '#9ca3af' }}>Missing or invalid URL.</div>
      )}
    </div>
  );
}
