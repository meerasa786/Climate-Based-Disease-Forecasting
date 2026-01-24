import { useEffect, useState } from "react";
import { api } from "../api/client";

export default function Coupons() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");
  const [coupons, setCoupons] = useState([]);

  // form state
  const [name, setName] = useState("");
  const [code, setCode] = useState("");
  const [type, setType] = useState("fixed");
  const [value, setValue] = useState(10);
  const [maxRedemptions, setMaxRedemptions] = useState("");
  const [singleUse, setSingleUse] = useState(false);
  const [status, setStatus] = useState("active");
  const [endAt, setEndAt] = useState("");

  async function load() {
    try {
      setLoading(true);
      setErr("");
      const r = await api("/admin/coupons", { method: "GET", auth: true });
      setCoupons(r.coupons || []);
    } catch (e) {
      setErr(e.message || "Failed to load coupons");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleCreate(e) {
    e.preventDefault();
    try {
      setSaving(true);
      setErr("");
      setMsg("");

      const body = {
        name: name.trim(),
        code: code.trim(),
        type,
        value: Number(value),
        maxRedemptions: maxRedemptions ? Number(maxRedemptions) : null,
        singleUse,
        status,
        endAt: endAt || null,
      };

      const r = await api("/admin/coupons", {
        method: "POST",
        auth: true,
        body,
      });

      if (!r.ok) {
        throw new Error(r.msg || "Failed to create coupon");
      }

      setMsg("Coupon created.");
      setName("");
      setCode("");
      setType("fixed");
      setValue(10);
      setMaxRedemptions("");
      setSingleUse(false);
      setStatus("active");
      setEndAt("");

      await load();
    } catch (e) {
      setErr(e.message || "Failed to create coupon");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="container">
      <div className="h1">Coupons</div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="h2" style={{ marginBottom: 12 }}>
          Create coupon
        </div>

        <form className="col" onSubmit={handleCreate}>
          <div className="row" style={{ marginBottom: 8 }}>
            <label style={{ width: 140 }}>Name</label>
            <input
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="WELCOME10"
            />
          </div>

          <div className="row" style={{ marginBottom: 8 }}>
            <label style={{ width: 140 }}>Public code</label>
            <input
              className="input"
              value={code}
              onChange={(e) => setCode(e.target.value)}
              placeholder="WELCOME10"
            />
          </div>

          <div className="row" style={{ marginBottom: 8 }}>
            <label style={{ width: 140 }}>Type</label>
            <select
              className="input"
              value={type}
              onChange={(e) => setType(e.target.value)}
            >
              <option value="fixed">Fixed amount</option>
              <option value="percent">Percent</option>
            </select>
          </div>

          <div className="row" style={{ marginBottom: 8 }}>
            <label style={{ width: 140 }}>Value</label>
            <input
              className="input"
              type="number"
              value={value}
              onChange={(e) => setValue(e.target.value)}
            />
          </div>

          <div className="row" style={{ marginBottom: 8 }}>
            <label style={{ width: 140 }}>Max redemptions</label>
            <input
              className="input"
              type="number"
              value={maxRedemptions}
              onChange={(e) => setMaxRedemptions(e.target.value)}
              placeholder="leave blank for unlimited"
            />
          </div>

          <div className="row" style={{ marginBottom: 8 }}>
            <label style={{ width: 140 }}>Single use per customer</label>
            <input
              type="checkbox"
              checked={singleUse}
              onChange={(e) => setSingleUse(e.target.checked)}
              style={{ width: 16, height: 16 }}
            />
          </div>

          <div className="row" style={{ marginBottom: 8 }}>
            <label style={{ width: 140 }}>Status</label>
            <select
              className="input"
              value={status}
              onChange={(e) => setStatus(e.target.value)}
            >
              <option value="active">active</option>
              <option value="paused">paused</option>
              <option value="draft">draft</option>
              <option value="archived">archived</option>
            </select>
          </div>

          <div className="row" style={{ marginBottom: 12 }}>
            <label style={{ width: 140 }}>Expires at</label>
            <input
              className="input"
              type="datetime-local"
              value={endAt}
              onChange={(e) => setEndAt(e.target.value)}
            />
          </div>

          <button className="button" type="submit" disabled={saving}>
            {saving ? "Creating…" : "Create coupon"}
          </button>

          {msg && <div style={{ marginTop: 8, color: "#166534" }}>{msg}</div>}
          {err && <div style={{ marginTop: 8, color: "#b91c1c" }}>{err}</div>}
        </form>
      </div>

      <div className="card">
        <div className="h2" style={{ marginBottom: 12 }}>
          Existing coupons
        </div>

        {loading ? (
          <div>Loading…</div>
        ) : coupons.length === 0 ? (
          <div className="small">No coupons yet.</div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Code</th>
                <th>Type</th>
                <th>Value</th>
                <th>Status</th>
                <th>Total redemptions</th>
                <th>Last redemption</th>
              </tr>
            </thead>
            <tbody>
              {coupons.map((c) => (
                <tr key={c._id}>
                  <td>{c.name}</td>
                  <td>{c.code}</td>
                  <td>{c.type}</td>
                  <td>{c.value}</td>
                  <td>{c.status}</td>
                  <td>{c.stats?.totalRedemptions ?? 0}</td>
                  <td>
                    {c.stats?.lastRedemptionAt
                      ? new Date(c.stats.lastRedemptionAt).toLocaleString()
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}