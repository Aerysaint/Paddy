"use client";
import React, { useRef, useState, useEffect } from "react";

type Pane = {
  id: string;
  title: string;
  content: React.ReactNode;
};

type Props = {
  panes: Pane[];
  onClose?: (id: string) => void;
};

// Simple horizontal split supporting multiple resizable panes with drag handles
export default function SplitPane({ panes, onClose }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [sizes, setSizes] = useState<number[]>([]);
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [startX, setStartX] = useState<number>(0);
  const [startSizes, setStartSizes] = useState<number[]>([]);

  useEffect(() => {
    if (sizes.length === panes.length) return;
    // Initialize equally sized panes
    const n = panes.length || 1;
    const base = Math.floor(100 / n);
    const arr = new Array(n).fill(base);
    arr[arr.length - 1] = 100 - base * (n - 1);
    setSizes(arr);
  }, [panes.length]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (dragIdx === null) return;
      if (!containerRef.current) return;
      const dx = e.clientX - startX;
      const rect = containerRef.current.getBoundingClientRect();
      const w = rect.width;
      if (w <= 0) return;
      const deltaPct = (dx / w) * 100;
      const next = [...startSizes];
      const i = dragIdx;
      // Adjust sizes of pane i and i+1
      let left = Math.max(10, Math.min(90, startSizes[i] + deltaPct));
      let right = Math.max(10, Math.min(90, startSizes[i + 1] - deltaPct));
      // Keep total stable
      const sum = startSizes[i] + startSizes[i + 1];
      if (left + right !== sum) {
        if (left + right > sum) right = sum - left;
        else left = sum - right;
      }
      next[i] = left;
      next[i + 1] = right;
      setSizes(next);
    };
    const onUp = () => setDragIdx(null);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [dragIdx, startX, startSizes]);

  return (
    <div ref={containerRef} style={{ display: "flex", width: "100%", height: "100%", overflow: "hidden" }}>
      {panes.map((p, idx) => (
        <React.Fragment key={p.id}>
          <div style={{ width: `${sizes[idx] ?? 0}%`, height: "100%", display: "flex", flexDirection: "column", borderRight: idx < panes.length - 1 ? "1px solid rgba(255,255,255,0.08)" : undefined }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 8px", background: "#0b0f14", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
              <div style={{ fontWeight: 600, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p.title}</div>
              {onClose && (
                <button onClick={() => onClose(p.id)} title="Close" style={{ background: "transparent", color: "#94a3b8", border: "none", cursor: "pointer" }}>✕</button>
              )}
            </div>
            <div style={{ flex: 1, minHeight: 0 }}>{p.content}</div>
          </div>
          {idx < panes.length - 1 && (
            <div
              onMouseDown={(e) => {
                setDragIdx(idx);
                setStartX(e.clientX);
                setStartSizes([...sizes]);
              }}
              style={{ width: 6, cursor: "col-resize", background: "rgba(255,255,255,0.06)", userSelect: "none" }}
            />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}
