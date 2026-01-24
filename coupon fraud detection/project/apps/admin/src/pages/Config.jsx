import { useEffect, useState } from 'react';
import { api } from '../api/client';

export default function Config() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState('');
  const [err, setErr] = useState('');
  const [configId, setConfigId] = useState(null);

  const [challengeRisk, setChallengeRisk] = useState(0.6);
  const [blockRisk, setBlockRisk] = useState(0.8);
  const [wAnom, setWAnom] = useState(0.3);

  async function load() {
    try {
      setLoading(true);
      setErr('');
      const r = await api('/admin/rules-config/active', { method: 'GET', auth: true });
      const cfg = r?.config;
      if (!cfg) throw new Error('Missing config');

      setConfigId(cfg._id || cfg.id || null);
      setChallengeRisk(cfg.thresholds?.challengeRisk ?? 0.6);
      setBlockRisk(cfg.thresholds?.blockRisk ?? 0.8);
      setWAnom(cfg.w_anom ?? 0.3);
    } catch (e) {
      setErr(e.message || 'Failed to load config');
    } finally {
      setLoading(false);
    }
  }

  async function save() {
    try {
      setSaving(true);
      setMsg('');
      setErr('');
      const body = {
        thresholds: {
          challengeRisk: Number(challengeRisk),
          blockRisk: Number(blockRisk),
        },
        w_anom: Number(wAnom),
      };

      const path = configId
        ? `/admin/rules-config/${configId}`
        : '/admin/rules-config';

      const method = configId ? 'PUT' : 'POST';

      const r = await api(path, { method, body, auth: true });
      if (r?.config?._id) {
        setConfigId(r.config._id);
      }

      setMsg('Saved.');
    } catch (e) {
      setErr(e.message || 'Failed to save config');
    } finally {
      setSaving(false);
    }
  }

  useEffect(() => { load(); }, []);

  return (
    <div className="container">
      <div className="h1">Risk Config</div>

      {loading ? (
        <div>Loading…</div>
      ) : (
        <div className="card">
          <div className="row" style={{ marginBottom: 12 }}>
            <label style={{ width: 180 }}>ML Weight (w_anom)</label>
            <input
              className="input"
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={wAnom}
              onChange={e => setWAnom(e.target.value)}
            />
          </div>

          <div className="row" style={{ marginBottom: 12 }}>
            <label style={{ width: 180 }}>Challenge ≥</label>
            <input
              className="input"
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={challengeRisk}
              onChange={e => setChallengeRisk(e.target.value)}
            />
          </div>

          <div className="row" style={{ marginBottom: 12 }}>
            <label style={{ width: 180 }}>Block ≥</label>
            <input
              className="input"
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={blockRisk}
              onChange={e => setBlockRisk(e.target.value)}
            />
          </div>

          <button className="button" onClick={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>

          {msg && <div style={{ marginTop: 10, color: '#166534' }}>{msg}</div>}
          {err && <div style={{ marginTop: 10, color: '#b91c1c' }}>{err}</div>}
        </div>
      )}
    </div>
  );
}