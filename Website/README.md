### PDF Reading Experience (Next.js)

Implements:
- Displays PDFs using Adobe PDF Embed SDK
- Bulk upload PDFs and list them
- Open a fresh PDF (without upload) for first-time viewing

Setup:
1. Create `.env.local` with your Adobe Embed API key:
   ```
   NEXT_PUBLIC_ADOBE_EMBED_API_KEY=your_client_id_here
   ```
2. Install and run:
   ```bash
   npm install
   npm run dev
   ```

Use the UI to upload PDFs or open a fresh file. Click an uploaded file to view it. Zoom/pan are available in the viewer controls.


