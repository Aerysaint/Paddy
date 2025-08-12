import { NextRequest } from "next/server";
import { savePDF } from "@/lib/server/fs";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const form = await req.formData();
  const files = form.getAll("files").filter((v): v is File => v instanceof File);

  const stored = [] as { name: string; url: string }[];
  for (const f of files) {
    const buf = await f.arrayBuffer();
    const { name } = await savePDF(f.name, buf);
    stored.push({ name, url: `/api/files/${encodeURIComponent(name)}` });
  }

  return Response.json(stored);
}


