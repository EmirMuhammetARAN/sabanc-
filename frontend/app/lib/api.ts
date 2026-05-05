const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

// ── Token yönetimi ────────────────────────────
export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('neuroveil_token');
}

export function setToken(token: string) {
  localStorage.setItem('neuroveil_token', token);
}

export function clearToken() {
  localStorage.removeItem('neuroveil_token');
  localStorage.removeItem('neuroveil_doctor');
}

export function setDoctor(doctor: object) {
  localStorage.setItem('neuroveil_doctor', JSON.stringify(doctor));
}

export function getDoctor() {
  if (typeof window === 'undefined') return null;
  const d = localStorage.getItem('neuroveil_doctor');
  return d ? JSON.parse(d) : null;
}

// ── Base fetch ────────────────────────────────
async function apiFetch(path: string, options: RequestInit = {}) {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const res = await fetch(`${API_URL}${path}`, { ...options, headers });

  if (res.status === 401) {
    clearToken();
    window.location.href = '/login';
    throw new Error('Oturum süresi doldu.');
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Bir hata oluştu.');
  }

  return res.json();
}

// ══════════════════════════════════════════════
// AUTH
// ══════════════════════════════════════════════

export async function login(email: string, password: string) {
  const data = await apiFetch('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  setToken(data.token);
  setDoctor(data.doctor);
  return data;
}

export async function register(payload: {
  email: string;
  password: string;
  full_name: string;
  hospital?: string;
  department?: string;
}) {
  const data = await apiFetch('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  setToken(data.token);
  setDoctor(data.doctor);
  return data;
}

export async function getMe() {
  return apiFetch('/api/auth/me');
}

// ══════════════════════════════════════════════
// HASTALAR
// ══════════════════════════════════════════════

export async function createPatient(payload: {
  tc_no: string;
  full_name: string;
  birth_date?: string;
  gender?: string;
}) {
  return apiFetch('/api/patients', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function searchPatients(q: string) {
  return apiFetch(`/api/patients/search?q=${encodeURIComponent(q)}`);
}

export async function getPatient(patientId: string) {
  return apiFetch(`/api/patients/${patientId}`);
}

// ══════════════════════════════════════════════
// ANALİZ — ANA FONKSİYON
// ══════════════════════════════════════════════

export async function runPredict(
  mriFile: File,
  bloodFile: File,
  patientId: string,
) {
  const form = new FormData();
  form.append('mri_file', mriFile);
  form.append('blood_file', bloodFile);
  form.append('patient_id', patientId);
  form.append('save_to_db', 'true');

  const token = getToken();
  const res = await fetch(`${API_URL}/api/predict`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Analiz başarısız.');
  }

  return res.json();
}

export async function generateReport(analysisId: string) {
  return apiFetch('/api/report', {
    method: 'POST',
    body: JSON.stringify({ analysis_id: analysisId }),
  });
}

// ══════════════════════════════════════════════
// SONUÇLAR
// ══════════════════════════════════════════════

export async function getAnalysis(analysisId: string) {
  return apiFetch(`/api/analyses/${analysisId}`);
}

export async function getDashboard() {
  return apiFetch('/api/dashboard');
}
