import './globals.css'
import React from 'react'

export const metadata = {
  title: 'PDF Selection-to-Search',
  description: 'Adobe Embed + Vector Search Bubble'
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: '#0b0f14', color: '#e6edf3', fontFamily: 'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto' }}>
        {children}
      </body>
    </html>
  )
}
