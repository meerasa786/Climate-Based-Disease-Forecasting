import { useEffect, useMemo, useState } from 'react';
import { api } from '../api/client';

// Lightweight modal
function Modal({ open, onClose, children }) {
  if (!open) return null;
  return (
    <div style={{
      position:'fixed', inset:0, background:'rgba(0,0,0,.25)',
      display:'grid', placeItems:'center', padding:20, zIndex:50
    }}>
      <div className="card" style={{ width:'min(840px, 96vw)' }}>
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:8 }}>
          <div className="h1" style={{ margin:0 }}>Decision details</div>
          <button className="button ghost" onClick={onClose}>Close</button>
        </div>
        {children}
      </div>
    </div>
  );
}

export default function Decisions() {
  // filters
  const [decision, setDecision] = useState('');
  const [couponCode, setCouponCode] = useState('');
  const [from, setFrom] = useState(''); // ISO local like 2025-11-18T00:00
  const [to, setTo] = useState('');
  const [limit, setLimit] = useState(100);

  // live auto-refresh + presets
  const [auto, setAuto] = useState(false);
  const [refreshMs, setRefreshMs] = useState(3000);

  function toLocalISO(dt) {
    // returns YYYY-MM-DDTHH:MM for <input type="datetime-local">
    const t = new Date(dt.getTime() - dt.getTimezoneOffset() * 60000);
    return t.toISOString().slice(0,16);
  }
  function setPreset(range) {
    const now = new Date();
    let fromDt = null;
    if (range === '10m') fromDt = new Date(now.getTime() - 10 * 60 * 1000);
    if (range === '1h')  fromDt = new Date(now.getTime() - 60 * 60 * 1000);
    if (range === '24h') fromDt = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    if (fromDt) {
      setFrom(toLocalISO(fromDt));
      setTo('');
    }
  }

  // data
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);

  // selection
  const [sel, setSel] = useState(null);
  const [msg, setMsg] = useState('');

  async function load(opts = {}) {
    const silent = !!opts.silent;
    if (!silent) setLoading(true);
    setMsg('');
    try {
      const qs = new URLSearchParams();
      if (decision) qs.set('decision', decision);
      if (couponCode) qs.set('couponCode', couponCode);
      if (from) qs.set('from', new Date(from).toISOString());
      if (to) qs.set('to', new Date(to).toISOString());
      if (limit) qs.set('limit', String(limit));
      const r = await api(`/admin/redemptions?${qs.toString()}`);
      setRows(r.rows || []);
    } catch (e) {
      if (!silent) setMsg('Failed to load redemptions');
    } finally {
      if (!silent) setLoading(false);
    }
  }

  useEffect(() => { load(); /* initial */ }, []);

  // Auto-refresh when "Live" is on. Also re-run when filters change.
  useEffect(() => {
    if (!auto) return;
    const id = setInterval(() => load({ silent: true }), refreshMs);
    return () => clearInterval(id);
  }, [auto, refreshMs, decision, couponCode, from, to, limit]);

  // Actions (using your /api/admin/lists endpoints)
  async function addToList(type, kind, value, note='') {
    try {
      await api('/admin/lists', { method:'POST', body:{ type, items:[{ kind, value, note }] } });
      alert(`${kind}:${value} added to ${type}`);
    } catch {
      alert(`Failed to add to ${type}`);
    }
  }

  // Nice derived text for WHY
  const whyText = useMemo(() => {
    if (!sel) return '';
    if (Array.isArray(sel.why) && sel.why.length) return sel.why.join('; ');
    return sel.narration || '';
  }, [sel]);

  return (
    <div className="container">
      <div className="h1">Recent Decisions</div>

      {/* Filters */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row" style={{ flexWrap:'wrap', alignItems:'center' }}>
          <select className="select" value={decision} onChange={e=>setDecision(e.target.value)}>
            <option value="">All decisions</option>
            <option value="ALLOW">ALLOW</option>
            <option value="CHALLENGE">CHALLENGE</option>
            <option value="BLOCK">BLOCK</option>
          </select>

          <input className="input" placeholder="coupon code" value={couponCode} onChange={e=>setCouponCode(e.target.value)} />

          <input className="input" type="datetime-local" value={from} onChange={e=>setFrom(e.target.value)} />
          <input className="input" type="datetime-local" value={to} onChange={e=>setTo(e.target.value)} />

          <div className="row" style={{ gap:8 }}>
            <button className="button ghost" onClick={()=>setPreset('10m')}>Last 10m</button>
            <button className="button ghost" onClick={()=>setPreset('1h')}>1h</button>
            <button className="button ghost" onClick={()=>setPreset('24h')}>24h</button>
          </div>

          <input className="input" style={{ width: 96 }} value={limit} onChange={e=>setLimit(Number(e.target.value||100))} />

          <button className="button" onClick={()=>load()} disabled={loading}>{loading ? 'Loading...' : 'Refresh'}</button>

          <div className="row" style={{ gap:6, marginLeft:'auto' }}>
            <label className="h2" style={{ margin:0 }}>Live</label>
            <input type="checkbox" checked={auto} onChange={e=>setAuto(e.target.checked)} />
            {auto && <span style={{ width:10, height:10, borderRadius:999, background:'#10b981' }} />}
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="card">
        {msg && <div style={{ color:'#b91c1c', marginBottom:8 }}>{msg}</div>}
        <table className="table">
          <thead><tr>
            <th>Time</th>
            <th>User</th>
            <th>Device</th>
            <th>IP</th>
            <th>Coupon</th>
            <th>Decision</th>
            <th>Risk</th>
          </tr></thead>
          <tbody>
            {rows.map(r => (
              <tr key={r._id || r.id} onClick={()=>setSel(r)} style={{ cursor:'pointer' }}>
                <td>{new Date(r.createdAt).toLocaleString()}</td>
                <td>{r.userId || '-'}</td>
                <td>{r.deviceId || '-'}</td>
                <td>{r.ip || '-'}</td>
                <td>{r.couponCode || '-'}</td>
                <td><span className={`badge ${r.decision}`}>{r.decision}</span></td>
                <td>{(r.riskScore ?? r.risk ?? 0).toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && <div style={{ color:'#6b7280', marginTop:8 }}>No data for the current filters.</div>}
      </div>

      {/* Details + actions */}
      <Modal open={!!sel} onClose={()=>setSel(null)}>
        {sel && (
          <div className="col">
            <div className="row" style={{ gap:20, flexWrap:'wrap' }}>
              <div><div className="h2">Decision</div><span className={`badge ${sel.decision}`}>{sel.decision}</span></div>
              <div><div className="h2">Risk</div>{(sel.riskScore ?? sel.risk ?? 0).toFixed(2)}</div>
              <div><div className="h2">Coupon</div>{sel.couponCode || '-'}</div>
              <div><div className="h2">User</div>{sel.userId || '-'}</div>
              <div><div className="h2">Device</div>{sel.deviceId || '-'}</div>
              <div><div className="h2">IP</div>{sel.ip || '-'}</div>
              <div><div className="h2">Time</div>{new Date(sel.createdAt).toLocaleString()}</div>
            </div>

            <div style={{ marginTop:8 }}>
              <div className="h2">Why</div>
              <div style={{ whiteSpace:'pre-wrap' }}>{whyText || '-'}</div>
            </div>

            <div className="row" style={{ marginTop:12, flexWrap:'wrap' }}>
              {/* Whitelist quickly for demo */}
              {sel.couponCode && (
                <button className="button" onClick={()=>addToList('whitelist','couponCode',sel.couponCode,'demo allow')}>Whitelist coupon</button>
              )}
              {sel.userId && (
                <button className="button" onClick={()=>addToList('whitelist','userId',sel.userId,'demo allow')}>Whitelist user</button>
              )}
              {sel.deviceId && (
                <button className="button" onClick={()=>addToList('whitelist','deviceId',sel.deviceId,'demo allow')}>Whitelist device</button>
              )}
              {sel.ip && (
                <button className="button" onClick={()=>addToList('whitelist','ip',sel.ip,'demo allow')}>Whitelist IP</button>
              )}
              <button className="button ghost" onClick={()=>setSel(null)}>Done</button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}