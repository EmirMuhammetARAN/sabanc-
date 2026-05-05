'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import Navbar from '../components/Navbar';
import styles from './dashboard.module.css';
import { getDashboard } from '../lib/api';

function StatusBadge({ status }: { status: string }) {
  if (status === 'positive') return <span className="badge badge-red">🔴 POSITIVE</span>;
  if (status === 'negative') return <span className="badge badge-teal">🟢 NEGATIVE</span>;
  return <span className="badge badge-amber">🟡 Pending</span>;
}

export default function DashboardPage() {
  const [doctorEmail, setDoctorEmail] = useState('');
  const [dashboard, setDashboard] = useState<{
    total_analyses: number;
    positive_count: number;
    negative_count: number;
    positive_rate: number;
    avg_ensemble_score: number;
    recent_analyses: Array<{
      id: string;
      patient_id?: string;
      decision: string;
      ensemble_score: number;
      created_at: string;
      top_blood_genes?: string[];
      top_mri_features?: string[];
    }>;
  } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setDoctorEmail(localStorage.getItem('doctor_email') || 'demo@anti-covid.ai');
    async function loadDashboard() {
      try {
        const data = await getDashboard();
        setDashboard(data);
      } finally {
        setLoading(false);
      }
    }
    loadDashboard();
  }, []);

  const totalAnalyses = dashboard?.total_analyses ?? 0;
  const positive = dashboard?.positive_count ?? 0;
  const negative = dashboard?.negative_count ?? 0;
  const avgEnsemble = Math.round(dashboard?.avg_ensemble_score ?? 0);
  const recentAnalyses = dashboard?.recent_analyses ?? [];

  return (
    <div className="page-wrapper">
      <Navbar />

      <main className={styles.main}>
        <div className="container">

          {/* Header */}
          <div className={styles.header}>
            <div>
              <h1>Dashboard</h1>
              <p style={{ color: 'var(--text-secondary)', marginTop: 4 }}>
                Welcome, <strong style={{ color: 'var(--text-primary)' }}>{doctorEmail}</strong>
              </p>
            </div>
            <Link href="/analysis" className="btn btn-primary">
              + New Analysis
            </Link>
          </div>

          {/* Summary Cards */}
          <div className={styles.summaryGrid}>
            {[
              { label: 'Total Analyses', value: totalAnalyses, icon: '📊', color: 'var(--accent-blue)' },
              { label: 'POSITIVE Cases', value: positive, icon: '🔴', color: 'var(--accent-red)' },
              { label: 'NEGATIVE Cases', value: negative, icon: '🟢', color: 'var(--accent-teal)' },
              { label: 'Avg. Ensemble Score', value: `%${avgEnsemble}`, icon: '🤖', color: 'var(--accent-purple)' },
            ].map(card => (
              <div key={card.label} className={`glass-card ${styles.summaryCard}`}>
                <div className={styles.summaryIcon} style={{ color: card.color }}>{card.icon}</div>
                <div className={styles.summaryValue} style={{ color: card.color }}>{card.value}</div>
                <div className={styles.summaryLabel}>{card.label}</div>
              </div>
            ))}
          </div>

          {/* Recent Analyses */}
          <div className={styles.tableSection}>
            <div className={styles.tableHeader}>
              <h2>Recent Analyses</h2>
              <span className="badge badge-blue">{recentAnalyses.length} records</span>
            </div>

            <div className={`glass-card ${styles.tableCard}`}>
              {loading ? (
                <div style={{ padding: 24, color: 'var(--text-secondary)' }}>Loading records...</div>
              ) : recentAnalyses.length === 0 ? (
                <div style={{ padding: 24, color: 'var(--text-secondary)' }}>No analysis records yet.</div>
              ) : (
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Analysis</th>
                      <th>Date</th>
                      <th>Ensemble</th>
                      <th>Result</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentAnalyses.map((a) => {
                      const ensemble = Number(a.ensemble_score) || 0;
                      const status = a.decision === 'POSITIVE' ? 'positive' : 'negative';
                      return (
                        <tr key={a.id}>
                          <td>
                            <div className={styles.patientCell}>
                              <div className={styles.patientAvatar}>{a.id.slice(-2).toUpperCase()}</div>
                              <div>
                                <div style={{ fontWeight: 600 }}>Analysis #{a.id.slice(0, 8)}</div>
                                <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                                  Patient: {a.patient_id ? a.patient_id.slice(0, 8) : 'unknown'}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td style={{ color: 'var(--text-secondary)' }}>
                            {a.created_at ? new Date(a.created_at).toLocaleDateString('en-US') : '-'}
                          </td>
                          <td>
                            <span style={{ fontWeight: 700, color: ensemble >= 50 ? 'var(--accent-red)' : 'var(--accent-teal)' }}>
                              %{Math.round(ensemble)}
                            </span>
                          </td>
                          <td><StatusBadge status={status} /></td>
                          <td>
                            <Link href={`/results/${a.id}`} className="btn btn-secondary btn-sm">
                              Detail →
                            </Link>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
