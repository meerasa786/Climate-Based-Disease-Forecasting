import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, getToken, setToken } from '../api/client';

export default function CheckoutPage() {
  const [orderTotal, setOrderTotal] = useState('');
  const [couponCode, setCouponCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const [result, setResult] = useState(null);
  const [email, setEmail] = useState(
    localStorage.getItem('checkout_user_email') || ''
  );

  const navigate = useNavigate();

  useEffect(() => {
    if (!getToken()) {
      navigate('/login');
    }
  }, [navigate]);

  function handleLogout() {
    setToken(null);
    localStorage.removeItem('checkout_user_email');
    setResult(null);
    setOrderTotal('');
    setCouponCode('');
    navigate('/login');
  }

  async function handleRedeem(e) {
    e.preventDefault();
    setErr('');
    setResult(null);

    if (!getToken()) {
      setErr('Please login first.');
      return;
    }

    if (!orderTotal || !couponCode) {
      setErr('Enter order total and coupon code.');
      return;
    }

    const totalNum = Number(orderTotal);
    if (Number.isNaN(totalNum) || totalNum <= 0) {
      setErr('Order total should be a positive number.');
      return;
    }

    setLoading(true);
    try {
      const r = await api('/public/redemptions', {
        method: 'POST',
        auth: true,
        body: {
          couponCode,
          orderTotal: totalNum,
        },
      });
      setResult(r);
    } catch (e2) {
      setErr(e2.message || 'Failed to redeem coupon.');
    } finally {
      setLoading(false);
    }
  }

  const decision =
    result?.decision || result?.redemption?.decision || null;

  const risk =
    typeof result?.risk === 'number'
      ? result.risk
      : typeof result?.riskScore === 'number'
      ? result.riskScore
      : typeof result?.redemption?.riskScore === 'number'
      ? result.redemption.riskScore
      : null;

  let why = '';
  if (Array.isArray(result?.why) && result.why.length) {
    why = result.why.join('\n');
  } else if (typeof result?.narration === 'string') {
    why = result.narration;
  } else if (Array.isArray(result?.narration)) {
    why = result.narration.join('\n');
  }

  return (
    <div className="screen">
      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div>
            <div className="h1">Demo Checkout</div>
            <div className="tagline">
              Apply a coupon and see the fraud decision (ALLOW / CHALLENGE / BLOCK).
            </div>
          </div>
          <div className="col" style={{ alignItems: 'flex-end' }}>
            {email && (
              <div className="small" style={{ marginBottom: 4 }}>
                Logged in as {email}
              </div>
            )}
            <button
              className="button ghost"
              type="button"
              style={{ padding: '6px 12px', fontSize: 12 }}
              onClick={handleLogout}
            >
              Logout
            </button>
          </div>
        </div>

        <form className="col" onSubmit={handleRedeem} style={{ marginTop: 16 }}>
          <div className="row">
            <div style={{ flex: 1, marginRight: 8 }}>
              <label className="h2">Order total</label>
              <input
                className="input"
                type="number"
                min="0"
                step="0.01"
                placeholder="e.g. 49.99"
                value={orderTotal}
                onChange={e => setOrderTotal(e.target.value)}
              />
            </div>
            <div style={{ flex: 1, marginLeft: 8 }}>
              <label className="h2">Coupon code</label>
              <input
                className="input"
                placeholder="e.g. WELCOME10"
                value={couponCode}
                onChange={e => setCouponCode(e.target.value)}
              />
            </div>
          </div>

          <button className="button" type="submit" disabled={loading}>
            {loading ? 'Checking…' : 'Apply coupon'}
          </button>

          {err && <div className="error-text">{err}</div>}
        </form>

        {decision && (
          <div className="result-box">
            <div className="result-title">
              <div className="h2" style={{ marginBottom: 0 }}>
                Result
              </div>
              <span className={`badge ${decision}`}>{decision}</span>
            </div>
            {risk !== null && (
              <div style={{ fontSize: 13, marginBottom: 4 }}>
                Risk score: <strong>{risk.toFixed(2)}</strong>
              </div>
            )}
            {why && (
              <>
                <div className="h2" style={{ marginTop: 6 }}>
                  Why
                </div>
                <div className="result-why">{why}</div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}