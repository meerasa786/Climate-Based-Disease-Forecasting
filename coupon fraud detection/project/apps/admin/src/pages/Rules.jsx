

import { useEffect, useState } from 'react';
import { api } from '../api/client';

export default function Rules() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState('');
  const [msg, setMsg] = useState('');
  const [configId, setConfigId] = useState(null);

  const [rules, setRules] = useState([]);

  // Fetch active rules config
  async function load() {
    try {
      setLoading(true);
      setErr('');
      setMsg('');

      const res = await api('/admin/rules-config/active', {
        method: 'GET',
        auth: true,
      });

      const cfg = res?.config;
      if (!cfg) throw new Error('No active rule config found');

      setConfigId(cfg._id);
      setRules(cfg.rules || []);
    } catch (e) {
      setErr(e.message || 'Failed to load rule config');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  // Save updated rule list
  async function save() {
    try {
      setSaving(true);
      setErr('');
      setMsg('');

      const body = {
        rules,
      };

      const res = await api(`/admin/rules-config/${configId}`, {
        method: 'PUT',
        auth: true,
        body,
      });

      if (res?.config) {
        setMsg('Saved.');
      }
    } catch (e) {
      setErr(e.message || 'Failed to save rules');
    } finally {
      setSaving(false);
    }
  }

  function updateRule(index, field, value) {
    const next = [...rules];
    next[index] = { ...next[index], [field]: value };
    setRules(next);
  }

  function updateRuleParam(index, pkey, value) {
    const next = [...rules];
    next[index] = {
      ...next[index],
      params: {
        ...(next[index].params || {}),
        [pkey]: value,
      },
    };
    setRules(next);
  }

  return (
    <div className="container">
      <div className="h1">Rules Engine</div>
      <div className="small" style={{ marginBottom: 12 }}>
        Modify rule weights, parameters, and types.
      </div>

      {loading ? (
        <div>Loading…</div>
      ) : (
        <>
          {rules.map((r, i) => (
            <div
              key={i}
              className="card"
              style={{ marginBottom: 16, padding: 16 }}
            >
              <div className="h2" style={{ marginBottom: 8 }}>
                {r.label || r.id}
              </div>

              <div className="row" style={{ marginBottom: 8 }}>
                <label style={{ width: 120 }}>Score</label>
                <input
                  className="input"
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={r.score}
                  onChange={e => updateRule(i, 'score', Number(e.target.value))}
                />
              </div>

              <div className="row" style={{ marginBottom: 8 }}>
                <label style={{ width: 120 }}>Type</label>
                <select
                  className="input"
                  value={r.type}
                  onChange={e => updateRule(i, 'type', e.target.value)}
                >
                  <option value="soft">soft</option>
                  <option value="hard">hard</option>
                </select>
              </div>

              {r.params &&
                Object.keys(r.params).map((p, idx) => (
                  <div className="row" style={{ marginBottom: 8 }} key={idx}>
                    <label style={{ width: 120 }}>{p}</label>
                    <input
                      className="input"
                      type="number"
                      step="0.5"
                      value={r.params[p]}
                      onChange={e =>
                        updateRuleParam(i, p, Number(e.target.value))
                      }
                    />
                  </div>
                ))}
            </div>
          ))}

          <button className="button" onClick={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save Rules'}
          </button>

          {msg && (
            <div style={{ marginTop: 8, color: '#166534' }}>{msg}</div>
          )}
          {err && (
            <div style={{ marginTop: 8, color: '#b91c1c' }}>{err}</div>
          )}
        </>
      )}
    </div>
  );
}