import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { api, setToken } from '../api/client';

export default function RegisterPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [msg, setMsg] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setMsg('');

    if (!email || !password) {
      setMsg('Email and password are required.');
      return;
    }

    setLoading(true);
    try {
      const r = await api('/public/auth/register', {
        method: 'POST',
        auth: false,
        body: { email, password },
      });

      if (r.ok && r.token) {
        setToken(r.token);
        localStorage.setItem(
          'checkout_user_email',
          r.user?.email || email.trim().toLowerCase()
        );
        navigate('/checkout');
      } else {
        setMsg(r.msg || 'Registration failed.');
      }
    } catch (err) {
      setMsg(err.message || 'Registration failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="screen">
      <div className="card">
        <div className="h1">Demo Checkout</div>
        <div className="tagline">Create a customer account to redeem coupons.</div>

        <div className="h2" style={{ marginTop: 16 }}>
          Register
        </div>

        <form className="col" onSubmit={handleSubmit}>
          <input
            className="input"
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={e => setEmail(e.target.value)}
          />
          <input
            className="input"
            type="password"
            placeholder="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
          />

          <button className="button" type="submit" disabled={loading}>
            {loading ? 'Creating account…' : 'Register'}
          </button>

          {msg && <div className="small" style={{ marginTop: 6 }}>{msg}</div>}

          <div className="small" style={{ marginTop: 10 }}>
            Already have an account?{' '}
            <Link to="/login" className="link">
              Login
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}