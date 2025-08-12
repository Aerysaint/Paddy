"use client";

import { useEffect, useId, useRef } from "react";

declare global {
  interface Window {
    AdobeDC?: any;
  }
}

type Props = {
  url?: string;
  file?: File;
};

export default function PDFViewer({ url, file }: Props) {
  const divId = useId().replace(/[:]/g, "");
  const lastKeyRef = useRef<string | null>(null);

  useEffect(() => {
    const clientId = process.env.NEXT_PUBLIC_ADOBE_EMBED_API_KEY;
    if (!clientId) return;

    function render() {
      if (!window.AdobeDC) return;
      const adobeDCView = new window.AdobeDC.View({ clientId, divId });
      const viewerConfig = {
        embedMode: "FULL_WINDOW",
        defaultViewMode: "FIT_WIDTH",
        showZoomControl: true
      } as const;

      if (file) {
        const key = `file:${file.name}:${file.size}:${file.lastModified}`;
        lastKeyRef.current = key;
        const promise = file.arrayBuffer();
        adobeDCView.previewFile(
          {
            content: { promise },
            metaData: { fileName: file.name }
          },
          viewerConfig
        );
      } else if (url) {
        const key = `url:${url}`;
        lastKeyRef.current = key;
        adobeDCView.previewFile(
          {
            content: { location: { url } },
            metaData: { fileName: url.split("/").pop() || "document.pdf" }
          },
          viewerConfig
        );
      }
    }

    const readyHandler = () => render();

    if (window.AdobeDC) {
      render();
    } else {
      document.addEventListener("adobe_dc_view_sdk.ready", readyHandler);
      return () => document.removeEventListener("adobe_dc_view_sdk.ready", readyHandler);
    }
  }, [url, file, divId]);

  return <div id={divId} style={{ width: "100%", height: "100%" }} />;
}


