import { listPDFs } from "@/lib/server/fs";

export const runtime = "nodejs";

export async function GET() {
  const files = await listPDFs();
  return Response.json(files.map((f) => ({ name: f.name, url: `/api/files/${encodeURIComponent(f.name)}` })));
}


