"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, BarChart2, Target, GitBranch, Play, FlaskConical, Cpu } from "lucide-react";

const NAV = [
  { href: "/",          label: "Overview",           icon: LayoutDashboard },
  { href: "/kpi",       label: "KPI Dashboard",      icon: Target },
  { href: "/benchmark", label: "Benchmark Explorer", icon: BarChart2 },
  { href: "/simulator", label: "Cache Simulator",    icon: Cpu },
  { href: "/graph",     label: "Graph Explorer",     icon: GitBranch },
  { href: "/playback",  label: "User Playback",      icon: Play },
  { href: "/research",  label: "Research",           icon: FlaskConical },
];

export default function Navigation() {
  const path = usePathname();
  return (
    <nav style={{
      width: 200, flexShrink: 0, borderRight: "1px solid #e5e7eb",
      background: "#ffffff", padding: "20px 12px", display: "flex", flexDirection: "column", gap: 2,
    }}>
      <div style={{ marginBottom: 16, paddingLeft: 10 }}>
        <div className="text-sm font-semibold text-gray-900">GraphMind V6</div>
        <div className="text-xs text-gray-400 mt-0.5">Samsung · PS03</div>
      </div>
      {NAV.map(({ href, label, icon: Icon }) => (
        <Link key={href} href={href}>
          <div className={`nav-item ${path === href ? "active" : ""}`}>
            <Icon size={14} />
            {label}
          </div>
        </Link>
      ))}
      <div style={{ marginTop: "auto", paddingLeft: 10, paddingTop: 16, borderTop: "1px solid #f3f4f6" }}>
        <div className="text-xs text-gray-400">7/7 KPIs PASS</div>
        <div className="text-xs text-gray-400">Real UbiqLog · 31 users</div>
      </div>
    </nav>
  );
}
