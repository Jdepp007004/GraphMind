"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, BarChart3, GitBranch, Network,
  Cpu, PlayCircle, FlaskConical,
} from "lucide-react";

const navItems = [
  { href: "/",          label: "Overview",          icon: LayoutDashboard },
  { href: "/benchmark", label: "Benchmark",          icon: BarChart3 },
  { href: "/journey",   label: "Optimization",       icon: GitBranch },
  { href: "/graph",     label: "Graph Explorer",     icon: Network },
  { href: "/simulator", label: "Cache Simulator",    icon: Cpu },
  { href: "/playback",  label: "User Playback",      icon: PlayCircle },
  { href: "/research",  label: "Research",           icon: FlaskConical },
];

export default function Navigation() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-56 z-50 flex flex-col bg-white"
           style={{ borderRight: "1px solid #e5e7eb" }}>

      {/* Brand */}
      <div className="px-5 py-5" style={{ borderBottom: "1px solid #f3f4f6" }}>
        <div className="flex items-center gap-2.5 mb-0.5">
          <div className="w-6 h-6 rounded flex items-center justify-center bg-gray-900">
            <span className="text-white font-bold text-xs">G</span>
          </div>
          <span className="font-semibold text-sm text-gray-900">GraphMind</span>
        </div>
        <p className="text-xs text-gray-400 ml-8">V5 · Samsung AX 2025</p>
      </div>

      {/* Score pill */}
      <div className="px-4 py-3" style={{ borderBottom: "1px solid #f3f4f6" }}>
        <div className="bg-gray-50 rounded-lg px-3 py-2.5" style={{ border: "1px solid #e5e7eb" }}>
          <div className="text-xs text-gray-400 mb-1">Production F1</div>
          <div className="text-lg font-bold text-gray-900 tracking-tight">0.7745</div>
          <div className="text-xs text-green-600 font-medium">+0.0321 vs baseline</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-3 overflow-y-auto">
        <div className="space-y-0.5">
          {navItems.map((item) => {
            const active = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link key={item.href} href={item.href}>
                <div className={`nav-item ${active ? "active" : ""}`}>
                  <Icon size={15} strokeWidth={active ? 2 : 1.5} className={active ? "text-gray-700" : "text-gray-400"} />
                  <span>{item.label}</span>
                </div>
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Footer */}
      <div className="px-4 py-3" style={{ borderTop: "1px solid #f3f4f6" }}>
        <div className="text-xs text-gray-400">UbiqLog · 31 users · 208K transitions</div>
      </div>
    </aside>
  );
}
