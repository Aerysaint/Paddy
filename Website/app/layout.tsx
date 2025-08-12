import Script from "next/script";
import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "PDF Reader",
  description: "PDF reading experience with Adobe PDF Embed API"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <Script
          src="https://documentcloud.adobe.com/view-sdk/main.js"
          strategy="beforeInteractive"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}


