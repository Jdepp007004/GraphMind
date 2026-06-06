import type { Metadata } from "next";
import "./globals.css";
import Navigation from "@/components/Navigation";

export const metadata: Metadata = {
  title: "GraphMind — Samsung EnnovateX AX Hackathon",
  description: "GraphMindRL_V5: Reinforcement learning on Markov graphs for intelligent Android app prefetching. F1=0.7745, 31 users, UbiqLog dataset.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      </head>
      <body style={{ background: "#f7f7f8", color: "#111827" }}>
        <div className="flex min-h-screen">
          <Navigation />
          <main className="flex-1" style={{ marginLeft: "224px", minHeight: "100vh" }}>
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
