import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "GraphMind — AI App Prefetch | Samsung EnnovateX AX Hackathon",
  description:
    "GraphMindRL_V5: Reinforcement Learning on Markov graphs for intelligent Android app prefetching. F1=0.7745, 31 users, Samsung Galaxy A23. Samsung EnnovateX AX Hackathon 2025.",
  keywords: "GraphMind, app prefetch, reinforcement learning, Markov graph, Samsung, UbiqLog",
  authors: [{ name: "GraphMind Team" }],
  openGraph: {
    title: "GraphMind — AI App Prefetch Cache",
    description: "GraphMindRL_V5: F1=0.7745, ΔF1=+0.0321, 31 users, 208K transitions",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-[#050917] text-[#f0f4ff] antialiased">
        <div className="flex min-h-screen">
          <Navigation />
          <main className="flex-1 ml-64 min-h-screen">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
