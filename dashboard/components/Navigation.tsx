"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, BarChart3, GitBranch, Network,
  Cpu, PlayCircle, FlaskConical, ChevronRight,
} from "lucide-react";

const navItems = [
  { href: "/",            label: "Executive Overview",   icon: LayoutDashboard, id: "overview" },
  { href: "/benchmark",   label: "Benchmark Explorer",   icon: BarChart3,       id: "benchmark" },
  { href: "/journey",     label: "Optimization Journey", icon: GitBranch,       id: "journey" },
  { href: "/graph",       label: "Graph Explorer",       icon: Network,         id: "graph" },
  { href: "/simulator",   label: "Cache Simulator",      icon: Cpu,             id: "simulator" },
  { href: "/playback",    label: "User Playback",        icon: PlayCircle,      id: "playback" },
  { href: "/research",    label: "Research Validation",  icon: FlaskConical,    id: "research" },
];

export default function Navigation() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 z-50 flex flex-col"
           style={{ background: "rgba(5, 9, 23, 0.95)", borderRight: "1px solid rgba(255,255,255,0.06)", backdropFilter: "blur(20px)" }}>
      
      {/* Logo */}
      <div className="px-6 py-6 border-b border-white/5">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
               style={{ background: "linear-gradient(135deg, #1428a0, #00b0ff)" }}>
            <span className="text-white font-black text-sm">G</span>
          </div>
          <span className="font-bold text-lg tracking-tight text-white">GraphMind</span>
        </div>
        <p className="text-xs text-[#7a8fbf] ml-11">GraphMindRL_V5</p>
      </div>

      {/* F1 badge */}
      <div className="mx-4 mt-4 px-4 py-3 rounded-xl"
           style={{ background: "linear-gradient(135deg, rgba(0,230,118,0.12), rgba(0,176,255,0.08))", border: "1px solid rgba(0,230,118,0.2)" }}>
        <div className="text-xs text-[#7a8fbf] mb-0.5">Production F1-Score</div>
        <div className="text-2xl font-black text-[#00e676] metric-number">0.7745</div>
        <div className="text-xs text-[#00b0ff]">ΔF1 = +0.0321 vs baseline</div>
      </div>

      {/* Nav items */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link key={item.id} href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-200 group
                ${active
                  ? "nav-active text-[#00b0ff] font-medium"
                  : "text-[#7a8fbf] hover:text-white hover:bg-white/5"
                }`}>
              <Icon size={17} className={active ? "text-[#00b0ff]" : "text-current group-hover:text-[#00b0ff]"} />
              <span className="flex-1">{item.label}</span>
              {active && <ChevronRight size={14} className="text-[#00b0ff]/60" />}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-white/5">
        <div className="text-xs text-[#7a8fbf] text-center">Samsung EnnovateX AX 2025</div>
        <div className="text-xs text-center mt-0.5"
             style={{ color: "rgba(20,40,160,0.8)" }}>UbiqLog · 31 users · 208K transitions</div>
      </div>
    </aside>
  );
}
