'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import styles from './login.module.css';
import { login } from '../lib/api';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login(email, password);
      router.push('/dashboard');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Login failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.glow} />

      <div className={styles.card}>
        <div className={styles.logoWrap}>
          <Link href="/" className={styles.logo}>🧠 <span className="gradient-text">Anti-Covid</span></Link>
        </div>

        <h1 className={styles.title}>Doctor Login</h1>
        <p className={styles.subtitle}>Sign in with your institutional account to access the system.</p>

        {error && (
          <div className={styles.errorBox}>
            ⚠️ {error}
          </div>
        )}

        <form onSubmit={handleLogin} className={styles.form}>
          <div className="form-group">
            <label className="form-label">Email Address</label>
            <input
              id="email"
              type="email"
              className="form-input"
              placeholder="doctor@hospital.edu"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">Password</label>
            <input
              id="password"
              type="password"
              className="form-input"
              placeholder="••••••••"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
            />
          </div>

          <button
            id="login-btn"
            type="submit"
            className="btn btn-primary"
            style={{ width: '100%', marginTop: 8 }}
            disabled={loading}
          >
            {loading ? <><span className="spinner" /> Signing in...</> : 'Sign In →'}
          </button>
        </form>

        <div className={styles.dividerRow}>
          <div className="divider" style={{ flex: 1 }} />
          <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', padding: '0 12px' }}>or</span>
          <div className="divider" style={{ flex: 1 }} />
        </div>

        <button
          className="btn btn-secondary"
          style={{ width: '100%' }}
          onClick={() => { setEmail('berke@anti-covid.ai'); setPassword('test1234'); }}
        >
          🎯 Fill with Demo Account
        </button>

        <p className={styles.footerText}>
          Don&apos;t have an account?{' '}
          <Link href="/" style={{ color: 'var(--accent-blue)' }}>Contact the administrator</Link>
        </p>
      </div>
    </div>
  );
}
