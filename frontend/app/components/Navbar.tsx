'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <Link href="/" className="navbar-logo">
          🧠 <span>Anti-Covid</span>&nbsp;AI
        </Link>
        <div className="navbar-links">
          <Link href="/dashboard" className={`navbar-link ${pathname === '/dashboard' ? 'active' : ''}`}>
            Dashboard
          </Link>
          <Link href="/analysis" className={`navbar-link ${pathname === '/analysis' ? 'active' : ''}`}>
            New Analysis
          </Link>
          <Link href="/login" className="btn btn-primary btn-sm" style={{ marginLeft: 8 }}>
            Doctor Login
          </Link>
        </div>
      </div>
    </nav>
  );
}
