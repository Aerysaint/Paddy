import { NextRequest } from "next/server";
import path from "node:path";
import { getUploadsDir } from "@/lib/server/fs";
import { readFile, stat } from "node:fs/promises";

export const runtime = "nodejs";

export async function GET(req: NextRequest, { params }: { params: { path: string[] } }) {
  const dir = getUploadsDir();
  const fileName = params.path.join("/");
  const filePath = path.join(dir, fileName);
  try {
    const s = await stat(filePath);
    if (!s.isFile()) return new Response("Not found", { status: 404 });
    const data = await readFile(filePath);
    return new Response(data, {
      headers: {
        "Content-Type": "application/pdf",
        "Content-Length": String(data.byteLength)
      }
    });
  } catch (e) {
    return new Response("Not found", { status: 404 });
  }
}


