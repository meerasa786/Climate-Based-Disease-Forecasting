// apps/admin/src/pages/RulesConfigPage.jsx
import { useEffect, useState } from 'react';
import { api } from '../api/client';

export default function RulesConfigPage() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [msg, setMsg] = useState('');
  const [configId, setConfigId] = useState(null);

  const [challengeRisk, setChallengeRisk] = useState(0.6);
  const [blockRisk, setBlockRisk] = useState(0.8);
  const [wAnom, setWAnom] = useState(0.3);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError('');
      setMsg('');
      try {
        const res = await api('/admin/rules-config/active', { method: 'GET', auth: true });
        const cfg = res?.config;
        if (!cfg) throw new Error('No config found');

        if (!cancelled) {
          setConfigId(cfg._id || cfg.id || null);
          setChallengeRisk(cfg.thresholds?.challengeRisk ?? 0.6);
          setBlockRisk(cfg.thresholds?.blockRisk ?? 0.8);
          setWAnom(cfg.w_anom ?? 0.3);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e.message || 'Failed to load rules config');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleSave(e) {
    e.preventDefault();
    setSaving(true);
    setError('');
    setMsg('');

    const payload = {
      thresholds: {
        challengeRisk: Number(challengeRisk),
        blockRisk: Number(blockRisk),
      },
      w_anom: Number(wAnom),
    };

    try {
      let res;
      if (configId) {
        res = await api(`/admin/rules-config/${configId}`, {
          method: 'PUT',
          auth: true,
          body: payload,
        });
      } else {
        res = await api('/admin/rules-config', {
          method: 'POST',
          auth: true,
          body: { ...payload, enable: true, disableOthers: true },
        });
      }

      const cfg = res?.config;
      setConfigId(cfg?._id || cfg?.id || configId);
      setMsg('Saved.');
    } catch (e) {
      setError(e.message || 'Failed to save config');
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="admin-page">
      <div className="admin-page-header">
        <h1>Risk & Rules Configuration</h1>
        <p className="admin-page-subtitle">
          Tune when redemptions are ALLOW / CHALLENGE / BLOCK without changing code.
        </p>
      </div>

      {loading ? (
        <div>Loading…</div>
      ) : (
        <form className="admin-form" onSubmit={handleSave}>
          <div className="admin-card">
            <h2>Thresholds</h2>
            <p className="admin-help">
              Risk is a value in [0, 1]. Below challenge threshold → ALLOW. Between challenge and block →
              CHALLENGE. Above block → BLOCK.
            </p>

            <div className="admin-field">
              <label>Challenge risk threshold</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={challengeRisk}
                onChange={e => setChallengeRisk(e.target.value)}
              />
            </div>

            <div className="admin-field">
              <label>Block risk threshold</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={blockRisk}
                onChange={e => setBlockRisk(e.target.value)}
              />
            </div>
          </div>

          <div className="admin-card" style={{ marginTop: 16 }}>
            <h2>Anomaly weight</h2>
            <p className="admin-help">
              Controls how much the Isolation Forest score influences risk (0 = rules only, 1 = mostly anomaly).
            </p>
            <div className="admin-field">
              <label>Anomaly weight (w_anom)</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={wAnom}
                onChange={e => setWAnom(e.target.value)}
              />
            </div>
          </div>

          <div style={{ marginTop: 20, display: 'flex', alignItems: 'center', gap: 12 }}>
            <button className="button" type="submit" disabled={saving}>
              {saving ? 'Saving…' : 'Save config'}
            </button>
            {msg && <span className="admin-msg">{msg}</span>}
            {error && <span className="admin-error">{error}</span>}
          </div>
        </form>
      )}
    </div>
  );
}