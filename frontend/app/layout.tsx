import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Anti-Covid AI — Long COVID Diagnostic Platform",
  description: "Multi-modal agentic AI diagnostic system for Post-Acute Sequelae of COVID-19 (PASC). Powered by MRI and blood RNA-seq analysis.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
