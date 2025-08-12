import { mkdir, writeFile, stat, readdir, readFile } from "node:fs/promises";
import path from "node:path";

export const getUploadsDir = () => path.join(process.cwd(), "uploads");

export async function ensureUploadsDir(): Promise<string> {
  const dir = getUploadsDir();
  try {
    await mkdir(dir, { recursive: true });
  } catch (err) {
    // already exists or cannot be created
  }
  return dir;
}

export async function savePDF(filename: string, data: ArrayBuffer) {
  const dir = await ensureUploadsDir();
  const safeName = filename.replace(/[^a-zA-Z0-9_.-]/g, "_");
  const filePath = path.join(dir, safeName);
  await writeFile(filePath, Buffer.from(data));
  return { filePath, name: safeName } as const;
}

export async function listPDFs() {
  const dir = await ensureUploadsDir();
  try {
    const items = await readdir(dir, { withFileTypes: true });
    return items
      .filter((d) => d.isFile() && d.name.toLowerCase().endsWith(".pdf"))
      .map((d) => ({ name: d.name, path: path.join(dir, d.name) }));
  } catch {
    return [] as { name: string; path: string }[];
  }
}


