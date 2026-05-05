'use client';

import { useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Navbar from '../components/Navbar';
import styles from './analysis.module.css';
import { createPatient, runPredict } from '../lib/api';

type FileState = File | null;

export default function AnalysisPage() {
  const router = useRouter();
  const [mriFile, setMriFile] = useState<FileState>(null);
  const [bloodFile, setBloodFile] = useState<FileState>(null);
  const [patientName, setPatientName] = useState('');
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState<string>('');
  const [error, setError] = useState('');

  const mriRef = useRef<HTMLInputElement>(null);
  const bloodRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: React.DragEvent, type: 'mri' | 'blood') {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    if (type === 'mri') setMriFile(file);
    else setBloodFile(file);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!mriFile || !bloodFile || !patientName) return;

    setLoading(true);
    setError('');

    try {
      // Step 1: Create or find patient
      setStep('Creating patient record...');
      const patientRes = await createPatient({
        tc_no: `DEMO_${Date.now()}`,
        full_name: patientName,
      });
      const patientId = patientRes.patient.id;

      // Step 2: Run models
      setStep('Processing MRI data...');
      await new Promise(r => setTimeout(r, 500));

      setStep('Analyzing blood RNA-Seq...');
      await new Promise(r => setTimeout(r, 500));

      setStep('Calculating meta-model score...');
      const result = await runPredict(mriFile, bloodFile, patientId);

      // Step 3: Redirect to results page
      setStep('Saving to Supabase...');
      await new Promise(r => setTimeout(r, 300));

      if (result.analysis_id) {
        router.push(`/results/${result.analysis_id}`);
      } else {
        // If DB save failed, put result in URL temporarily
        router.push(`/results/temp?data=${encodeURIComponent(JSON.stringify(result))}`);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'An error occurred during analysis.');
      setLoading(false);
      setStep('');
    }
  }

  const canSubmit = mriFile && bloodFile && patientName && !loading;

  return (
    <div className="page-wrapper">
      <Navbar />

      <main className={styles.main}>
        <div className="container">
          <div className={styles.header}>
            <h1>New Analysis</h1>
            <p style={{ color: 'var(--text-secondary)', marginTop: 6 }}>
              Start the diagnostic process by uploading the patient's MRI and blood data.
            </p>
          </div>

          {error && (
            <div style={{
              background: 'rgba(255,77,109,0.1)',
              border: '1px solid rgba(255,77,109,0.3)',
              color: 'var(--accent-red)',
              borderRadius: 12,
              padding: '12px 16px',
              marginBottom: 20,
            }}>
              ⚠️ {error}
            </div>
          )}

          {loading ? (
            <div className={styles.loadingCard}>
              <div className={styles.loadingSpinner}>
                <div className={styles.spinRing} />
                <span>🧠</span>
              </div>
              <h2>Analysis in Progress</h2>
              <p className={styles.stepText}>{step}</p>
              <div className={styles.stepList}>
                {['Patient Record', 'MRI Processing', 'Blood Analysis', 'Meta-Model', 'Saving'].map((s, i) => (
                  <div key={s} className={styles.stepItem}>
                    <div className={`${styles.stepDot} ${i < 4 ? styles.stepDone : ''}`} />
                    <span>{s}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className={styles.form}>
              <div className={styles.grid}>
                {/* Left panel */}
                <div className={styles.formSection}>
                  <h2 className={styles.sectionTitle}>
                    <span>①</span> Patient Information
                  </h2>

                  <div className={styles.fields}>
                    <div className="form-group">
                      <label className="form-label">Patient Full Name *</label>
                      <input
                        id="patient-id"
                        type="text"
                        className="form-input"
                        placeholder="e.g. John Doe"
                        value={patientName}
                        onChange={e => setPatientName(e.target.value)}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label className="form-label">Clinical Notes (optional)</label>
                      <textarea
                        id="notes"
                        className="form-input"
                        style={{ minHeight: 100, resize: 'vertical' }}
                        placeholder="Symptoms, anamnesis, previous diagnoses..."
                        value={notes}
                        onChange={e => setNotes(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className={styles.infoBox}>
                    <div className={styles.infoIcon}>💡</div>
                    <div>
                      <strong>Supported Formats</strong>
                      <p>MRI: Excel (.xlsx) — Dryad_final format</p>
                      <p>Blood: CSV (.csv) — blood_X format</p>
                    </div>
                  </div>
                </div>

                {/* Right panel */}
                <div className={styles.formSection}>
                  <h2 className={styles.sectionTitle}>
                    <span>②</span> Data Upload
                  </h2>

                  <div
                    className={`${styles.dropzone} ${mriFile ? styles.dropzoneFilled : ''}`}
                    onDragOver={e => e.preventDefault()}
                    onDrop={e => handleDrop(e, 'mri')}
                    onClick={() => mriRef.current?.click()}
                  >
                    <input ref={mriRef} type="file" hidden accept=".xlsx" onChange={e => setMriFile(e.target.files?.[0] ?? null)} />
                    <div className={styles.dropzoneIcon}>{mriFile ? '✅' : '🧲'}</div>
                    <div className={styles.dropzoneTitle}>
                      {mriFile ? mriFile.name : 'Brain MRI Data'}
                    </div>
                    <div className={styles.dropzoneHint}>
                      {mriFile ? `${(mriFile.size / 1024).toFixed(1)} KB` : 'Click or drag · .xlsx'}
                    </div>
                  </div>

                  <div
                    className={`${styles.dropzone} ${bloodFile ? styles.dropzoneFilled : ''}`}
                    onDragOver={e => e.preventDefault()}
                    onDrop={e => handleDrop(e, 'blood')}
                    onClick={() => bloodRef.current?.click()}
                    style={{ marginTop: 16 }}
                  >
                    <input ref={bloodRef} type="file" hidden accept=".csv" onChange={e => setBloodFile(e.target.files?.[0] ?? null)} />
                    <div className={styles.dropzoneIcon}>{bloodFile ? '✅' : '🧬'}</div>
                    <div className={styles.dropzoneTitle}>
                      {bloodFile ? bloodFile.name : 'Blood RNA-Seq Data'}
                    </div>
                    <div className={styles.dropzoneHint}>
                      {bloodFile ? `${(bloodFile.size / 1024).toFixed(1)} KB` : 'Click or drag · .csv'}
                    </div>
                  </div>
                </div>
              </div>

              <div className={styles.submitRow}>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  {!mriFile && '⚠ MRI file required  '}
                  {!bloodFile && '⚠ Blood file required  '}
                  {!patientName && '⚠ Patient name required'}
                </div>
                <button
                  id="start-analysis"
                  type="submit"
                  className="btn btn-primary btn-lg"
                  disabled={!canSubmit}
                >
                  🚀 Start Analysis
                </button>
              </div>
            </form>
          )}
        </div>
      </main>
    </div>
  );
}
