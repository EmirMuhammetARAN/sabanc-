'use client';

import React, { Suspense, useEffect, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import styles from './results.module.css';
import { getAnalysis, generateReport } from '../../lib/api';

function ScoreGauge({ value, label, color }: { value: number; label: string; color: string }) {
  return (
    <div className={styles.gaugeCard}>
      <div className={styles.gaugeCircle} style={{ '--pct': value, '--color': color } as React.CSSProperties}>
        <div className={styles.gaugeInner}>
          <span className={styles.gaugeValue} style={{ color }}>{Math.round(value)}%</span>
        </div>
      </div>
      <div className={styles.gaugeLabel}>{label}</div>
    </div>
  );
}

// ══════════════════════════════════════════════
// PRINT REPORT — A4 layout for physicians
// ══════════════════════════════════════════════
function PrintReport({ analysis, id }: { analysis: Record<string, unknown>; id: string }) {
  const mri      = Number(analysis.mri_score) || 0;
  const blood    = Number(analysis.blood_score) || 0;
  const ensemble = Number(analysis.ensemble_score) || 0;
  const isPos    = analysis.decision === 'POSITIVE';
  const topMri   = (analysis.top_mri_features as string[]) || [];
  const topBlood = (analysis.top_blood_genes as string[]) || [];
  const report   = analysis.gemini_report as string | null;
  const dateStr  = analysis.created_at
    ? new Date(analysis.created_at as string).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
    : new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

  const posColor  = '#e03c5a';
  const negColor  = '#00b894';
  const mainColor = isPos ? posColor : negColor;

  return (
    <div className={styles.printReport}>

      {/* HEADER */}
      <div className={styles.printHeader}>
        <div className={styles.printLogo}>
          <div className={styles.printLogoIcon}>A</div>
          <div>
            <div className={styles.printLogoText}>Anti-Covid AI</div>
            <div className={styles.printLogoSub}>Neurology Diagnostic Assistant</div>
          </div>
        </div>
        <div className={styles.printHeaderRight}>
          <div className={styles.printTitle}>Diagnostic Report</div>
          <div className={styles.printDate}>{dateStr} · ID: {id.slice(0, 12)}...</div>
        </div>
      </div>

      {/* METRIC CARDS */}
      <div className={styles.printMetrics}>
        <div className={styles.printMetricCard}>
          <div className={styles.printMetricLabel}>Brain MRI Score</div>
          <div className={styles.printMetricValue} style={{ color: mri >= 50 ? posColor : negColor }}>
            {Math.round(mri)}%
          </div>
          <div className={styles.printMetricSub}>Structural Analysis · 74% Acc</div>
        </div>
        <div className={styles.printMetricCard}>
          <div className={styles.printMetricLabel}>Blood RNA-Seq Score</div>
          <div className={styles.printMetricValue} style={{ color: blood >= 50 ? posColor : negColor }}>
            {Math.round(blood)}%
          </div>
          <div className={styles.printMetricSub}>Gene Expression · 97% Acc</div>
        </div>
        <div className={styles.printMetricCard}>
          <div className={styles.printMetricLabel}>Ensemble Meta-Model</div>
          <div className={styles.printMetricValue} style={{ color: mainColor }}>
            {Math.round(ensemble)}%
          </div>
          <div className={styles.printMetricSub}>MRI×0.35 + Blood×0.65</div>
        </div>
        <div className={styles.printMetricCard} style={{ background: isPos ? '#fff0f3' : '#f0fff8', borderColor: isPos ? '#f5c6cf' : '#b2f0df' }}>
          <div className={styles.printMetricLabel}>System Decision</div>
          <div className={styles.printMetricValue} style={{ color: mainColor, fontSize: 18, paddingTop: 4 }}>
            {isPos ? 'POSITIVE' : 'NEGATIVE'}
          </div>
          <div className={styles.printMetricSub}>Long COVID {isPos ? 'Detected' : 'Not Detected'}</div>
        </div>
      </div>

      {/* DECISION BAND */}
      <div
        className={styles.printDecisionBand}
        style={{ background: isPos ? '#fff0f3' : '#f0fff8', border: `1px solid ${isPos ? '#f5c6cf' : '#b2f0df'}` }}
      >
        <div className={styles.printDecisionDot} style={{ background: mainColor }} />
        <span style={{ color: mainColor }}>
          {isPos ? '🔴 POSITIVE — Long COVID (PASC) Detected' : '🟢 NEGATIVE — Long COVID Not Detected'}
        </span>
        <span style={{ marginLeft: 'auto', color: '#999', fontSize: 11 }}>
          Ensemble Score: {Math.round(ensemble)}% · Threshold: 50%
        </span>
      </div>

      {/* FINDINGS TABLES */}
      <div className={styles.printMidGrid}>

        {/* MRI findings */}
        <div className={styles.printSection}>
          <div className={styles.printSectionHead}>🧲 MRI — Structural Brain Findings</div>
          <table className={styles.printTable}>
            <thead>
              <tr>
                <th>#</th>
                <th>Brain Region</th>
                <th>Effect</th>
                <th>MRI Score</th>
              </tr>
            </thead>
            <tbody>
              {topMri.length > 0 ? topMri.map((f, i) => (
                <tr key={f}>
                  <td style={{ color: '#999', fontSize: 11 }}>{i + 1}.</td>
                  <td style={{ fontWeight: 500 }}>{f}</td>
                  <td>
                    <span style={{ color: isPos ? posColor : negColor, fontSize: 10, fontWeight: 600 }}>
                      {isPos ? 'Abnormal' : 'Normal'}
                    </span>
                  </td>
                  <td>
                    <div className={styles.printScoreBar}>
                      <div className={styles.printBarTrack}>
                        <div className={styles.printBarFill} style={{ width: `${mri}%`, background: mainColor }} />
                      </div>
                      <span className={styles.printBarVal} style={{ color: mainColor }}>{Math.round(mri)}%</span>
                    </div>
                  </td>
                </tr>
              )) : (
                <tr><td colSpan={4} style={{ color: '#aaa', textAlign: 'center', padding: 16 }}>No significant findings detected</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Blood findings */}
        <div className={styles.printSection}>
          <div className={styles.printSectionHead}>🧬 Blood — Gene Expression Findings</div>
          <table className={styles.printTable}>
            <thead>
              <tr>
                <th>#</th>
                <th>Gene Name</th>
                <th>Type</th>
                <th>Blood Score</th>
              </tr>
            </thead>
            <tbody>
              {topBlood.length > 0 ? topBlood.map((f, i) => (
                <tr key={f}>
                  <td style={{ color: '#999', fontSize: 11 }}>{i + 1}.</td>
                  <td style={{ fontWeight: 500 }}>{f}</td>
                  <td><span style={{ fontSize: 10, color: '#888' }}>RNA-Seq</span></td>
                  <td>
                    <div className={styles.printScoreBar}>
                      <div className={styles.printBarTrack}>
                        <div className={styles.printBarFill} style={{ width: `${blood}%`, background: mainColor }} />
                      </div>
                      <span className={styles.printBarVal} style={{ color: mainColor }}>{Math.round(blood)}%</span>
                    </div>
                  </td>
                </tr>
              )) : (
                <tr><td colSpan={4} style={{ color: '#aaa', textAlign: 'center', padding: 16 }}>No significant findings detected</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* AI REPORT */}
      {report && (
        <div className={styles.printReportSection}>
          <div className={styles.printReportHead}>
            🤖 AI Clinical Evaluation Report
          </div>
          <div className={styles.printReportBody}>
            {report.split('\n\n').map((p, i) => (
              <p key={i} dangerouslySetInnerHTML={{ __html: p.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
            ))}
          </div>
        </div>
      )}

      {/* FOOTER */}
      <div className={styles.printFooter}>
        <span className={styles.printFooterLogo}>Anti-Covid AI — Sabancı University × DTC</span>
        <span>This report was generated by an AI-assisted diagnostic system. Final diagnosis belongs to the physician.</span>
        <span>Page 1/1</span>
      </div>

    </div>
  );
}

// ══════════════════════════════════════════════
// MAIN RESULTS CONTENT
// ══════════════════════════════════════════════
function ResultsContent({ id }: { id: string }) {
  const searchParams = useSearchParams();
  const [analysis, setAnalysis] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showReport, setShowReport] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const tempData = searchParams.get('data');
        if (id === 'temp' && tempData) {
          setAnalysis(JSON.parse(decodeURIComponent(tempData)));
          setLoading(false);
          return;
        }
        const data = await getAnalysis(id);
        setAnalysis(data);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Analysis could not be loaded.');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id, searchParams]);

  async function handleGenerateReport() {
    if (!analysis || id === 'temp') return;
    setReportLoading(true);
    try {
      const res = await generateReport(id);
      setAnalysis(prev => prev ? { ...prev, gemini_report: res.report } : prev);
      setShowReport(true);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Report could not be generated.');
    } finally {
      setReportLoading(false);
    }
  }

  if (loading) {
    return (
      <div className={styles.loadingState}>
        <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
        <p>Loading results...</p>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className={styles.loadingState}>
        <p style={{ color: 'var(--accent-red)' }}>⚠️ {error || 'Analysis not found.'}</p>
        <Link href="/dashboard" className="btn btn-secondary" style={{ marginTop: 16 }}>← Dashboard</Link>
      </div>
    );
  }

  const mri      = Number(analysis.mri_score) || 0;
  const blood    = Number(analysis.blood_score) || 0;
  const ensemble = Number(analysis.ensemble_score) || 0;
  const isPositive = analysis.decision === 'POSITIVE';
  const topMri   = (analysis.top_mri_features as string[]) || [];
  const topBlood = (analysis.top_blood_genes as string[]) || [];
  const report   = analysis.gemini_report as string | null;

  return (
    <>
      {/* Normal screen view */}
      <main className={styles.main}>
        <div className="container">

          <div className={styles.breadcrumb}>
            <Link href="/dashboard" style={{ color: 'var(--text-muted)' }}>Dashboard</Link>
            <span style={{ color: 'var(--text-muted)' }}> / </span>
            <span>Analysis Result</span>
          </div>

          <div className={styles.resultHeader}>
            <div>
              <h1>Analysis Result</h1>
              <div className={styles.meta}>
                <span style={{ color: 'var(--text-secondary)' }}>ID: {id.slice(0, 8)}...</span>
                <span style={{ color: 'var(--text-muted)' }}>·</span>
                <span style={{ color: 'var(--text-secondary)' }}>
                  {analysis.created_at
                    ? new Date(analysis.created_at as string).toLocaleDateString('en-US')
                    : new Date().toLocaleDateString('en-US')}
                </span>
              </div>
            </div>
            <div className={styles.verdictBadge} style={{
              background: isPositive ? 'rgba(255,77,109,0.1)' : 'rgba(0,212,170,0.1)',
              border: `1px solid ${isPositive ? 'rgba(255,77,109,0.3)' : 'rgba(0,212,170,0.3)'}`,
              color: isPositive ? 'var(--accent-red)' : 'var(--accent-teal)',
            }}>
              <span className={`status-dot ${isPositive ? 'positive' : 'negative'}`} />
              {isPositive ? '🔴 POSITIVE — Long COVID Detected' : '🟢 NEGATIVE — Long COVID Not Detected'}
            </div>
          </div>

          <div className={styles.gaugesRow}>
            <ScoreGauge value={mri} label="Brain MRI Model" color={mri >= 50 ? 'var(--accent-red)' : 'var(--accent-teal)'} />
            <ScoreGauge value={blood} label="Blood RNA-Seq Model" color={blood >= 50 ? 'var(--accent-red)' : 'var(--accent-teal)'} />
            <div className={styles.ensembleCard} style={{
              borderColor: isPositive ? 'rgba(255,77,109,0.3)' : 'rgba(0,212,170,0.3)',
            }}>
              <div className={styles.ensembleLabel}>Ensemble Meta-Model</div>
              <div className={styles.ensembleValue} style={{ color: isPositive ? 'var(--accent-red)' : 'var(--accent-teal)' }}>
                {Math.round(ensemble)}%
              </div>
              <div className="progress-bar" style={{ marginTop: 16 }}>
                <div className={`progress-fill ${isPositive ? 'danger' : ''}`} style={{ width: `${ensemble}%` }} />
              </div>
              <div className={styles.ensembleWeights}>
                <span>MRI × 0.35</span>
                <span>Blood × 0.65</span>
              </div>
            </div>
          </div>

          <div className={styles.findingsGrid}>
            <div className={`glass-card ${styles.findingCard}`}>
              <h3>🧲 MRI Abnormalities</h3>
              <div className={styles.findingList}>
                {topMri.length > 0 ? topMri.map(f => (
                  <div key={f} className={styles.findingItem}>
                    <div className={`status-dot ${isPositive ? 'positive' : 'negative'}`} />
                    <span>{f}</span>
                  </div>
                )) : <p style={{ color: 'var(--text-muted)' }}>No significant findings</p>}
              </div>
            </div>
            <div className={`glass-card ${styles.findingCard}`}>
              <h3>🧬 Gene Expression Abnormalities</h3>
              <div className={styles.findingList}>
                {topBlood.length > 0 ? topBlood.map(f => (
                  <div key={f} className={styles.findingItem}>
                    <div className={`status-dot ${isPositive ? 'positive' : 'negative'}`} />
                    <span>{f}</span>
                    <span className={styles.geneCode}>RNA-Seq</span>
                  </div>
                )) : <p style={{ color: 'var(--text-muted)' }}>No significant findings</p>}
              </div>
            </div>
          </div>

          <div className={`glass-card ${styles.reportCard}`}>
            <div className={styles.reportHeader}>
              <h2>🤖 AI Clinical Report</h2>
              {!report && !showReport && (
                <button className="btn btn-primary btn-sm" onClick={handleGenerateReport} disabled={reportLoading}>
                  {reportLoading ? <><span className="spinner" /> Generating...</> : '✨ Generate Report'}
                </button>
              )}
              {report && !showReport && (
                <button className="btn btn-secondary btn-sm" onClick={() => setShowReport(true)}>Show Report</button>
              )}
            </div>
            {(showReport && report) && (
              <div className={styles.reportBody}>
                {report.split('\n\n').map((paragraph, i) => (
                  <p key={i} style={{ marginBottom: 16 }}
                    dangerouslySetInnerHTML={{ __html: paragraph.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }}
                  />
                ))}
              </div>
            )}
          </div>

          <div className={styles.actions}>
            <Link href="/analysis" className="btn btn-secondary">+ New Analysis</Link>
            <Link href="/dashboard" className="btn btn-secondary">← Dashboard</Link>
            <button className="btn btn-primary" onClick={() => window.print()}>
              🖨️ Print Report
            </button>
          </div>

        </div>
      </main>

      {/* Print report — only visible on Ctrl+P */}
      <PrintReport analysis={analysis} id={id} />
    </>
  );
}

export default function ResultsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = React.use(params);
  return (
    <div className="page-wrapper">
      <Suspense fallback={
        <div style={{ padding: 40, textAlign: 'center' }}>
          <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3, margin: '0 auto' }} />
        </div>
      }>
        <ResultsContent id={id} />
      </Suspense>
    </div>
  );
}
