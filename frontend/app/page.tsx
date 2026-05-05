import Link from 'next/link';
import Navbar from './components/Navbar';
import styles from './page.module.css';

export default function LandingPage() {
  return (
    <div className="page-wrapper">
      <Navbar />

      {/* Hero */}
      <section className={styles.hero}>
        <div className={styles.heroGlow} />
        <div className={styles.heroGlow2} />

        <div className="container" style={{ position: 'relative', zIndex: 1 }}>
          <div className={styles.heroBadge}>
            <span className="status-dot positive" />
            <span>Active · Sabancı University × DTC</span>
          </div>

          <h1 className={styles.heroTitle}>
            Detect Long COVID<br />
            <span className="gradient-text">Early</span>
          </h1>

          <p className={styles.heroSubtitle}>
            A multi-modal AI system combining brain MRI and blood RNA-Seq data.
            Identify Post-Acute Sequelae with 97% accuracy, accelerating treatment.
          </p>

          <div className={styles.heroCtas}>
            <Link href="/analysis" className="btn btn-primary btn-lg">
              ✨ Start New Analysis
            </Link>
            <Link href="/dashboard" className="btn btn-secondary btn-lg">
              📊 Go to Dashboard
            </Link>
          </div>

          <div className={styles.heroStats}>
            {[
              { value: '97%', label: 'Blood Model Accuracy' },
              { value: '74%', label: 'MRI Model Accuracy' },
              { value: '2', label: 'Data Modalities' },
              { value: '<30s', label: 'Analysis Time' },
            ].map((stat) => (
              <div key={stat.label} className={styles.statBox}>
                <div className={styles.statValue}>{stat.value}</div>
                <div className={styles.statLabel}>{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className={styles.features}>
        <div className="container">
          <div className={styles.sectionHeader}>
            <h2>How It Works?</h2>
            <p>Comprehensive diagnosis in 3 steps</p>
          </div>

          <div className={styles.featureGrid}>
            {[
              {
                icon: '🔬',
                step: '01',
                title: 'Upload Data',
                desc: 'Upload the patient\'s brain MRI measurements and blood RNA-Seq gene expression data.',
              },
              {
                icon: '🤖',
                step: '02',
                title: 'AI Analysis',
                desc: 'Two independent models (MRI + Blood) run in parallel. The meta-model combines both.',
              },
              {
                icon: '📋',
                step: '03',
                title: 'Clinical Report',
                desc: 'AI interprets abnormal findings and generates a professional report for the physician.',
              },
            ].map((feat) => (
              <div key={feat.step} className={`glass-card ${styles.featureCard}`}>
                <div className={styles.featureStep}>{feat.step}</div>
                <div className={styles.featureIcon}>{feat.icon}</div>
                <h3>{feat.title}</h3>
                <p>{feat.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Banner */}
      <section className={styles.ctaBanner}>
        <div className="container">
          <div className={styles.ctaInner}>
            <div>
              <h2>Ready for Clinical Use</h2>
              <p>Fast integration · HIPAA compliant · Secure data processing</p>
            </div>
            <Link href="/login" className="btn btn-primary btn-lg">
              Create Doctor Account →
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className={styles.footer}>
        <div className="container">
          <p>© 2025 Anti-Covid AI · Sabancı University × Digital Technology Center</p>
        </div>
      </footer>
    </div>
  );
}
