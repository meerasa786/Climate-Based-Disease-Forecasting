import { useState } from 'react';
import { api, setToken } from '../api/client';

export default function Login({ onAuthed }) {
  const [mode, setMode] = useState('login'); // 'login' | 'register'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [msg, setMsg] = useState('');

  async function doLogin(e) {
    e.preventDefault();
    setMsg('');
    try {
      const r = await api('/admin/auth/login', { method:'POST', body:{ email, password }, auth:false });
      if (r.ok && r.token) { setToken(r.token); onAuthed(); } else setMsg(r.msg || 'Login failed');
    } catch { setMsg('Login failed'); }
  }
  async function doRegister(e) {
    e.preventDefault();
    setMsg('');
    try {
      const r = await api('/admin/auth/register', { method:'POST', body:{ email, password }, auth:false });
      if (r.ok && r.token) { setToken(r.token); onAuthed(); } else setMsg(r.msg || 'Register failed');
    } catch { setMsg('Register failed'); }
  }

  return (
    <div className="screen">
      <div className="card" style={{ width: 380 }}>
        <div className="h1">{mode === 'login' ? 'Admin Login' : 'Create Admin'}</div>

        <div className="row" style={{ marginBottom: 10 }}>
          <button className={`button ${mode==='login'?'':'ghost'}`} onClick={()=>setMode('login')}>Login</button>
          <button className={`button ${mode==='register'?'':'ghost'}`} onClick={()=>setMode('register')}>Register</button>
        </div>

        <form onSubmit={mode==='login' ? doLogin : doRegister} className="col">
          <input className="input" placeholder="email" value={email} onChange={e=>setEmail(e.target.value)} />
          <input className="input" placeholder="password" type="password" value={password} onChange={e=>setPassword(e.target.value)} />
          <button className="button" type="submit" style={{ marginTop: 6 }}>
            {mode==='login' ? 'Login' : 'Register'}
          </button>
          {msg && <div style={{ color:'#b91c1c', marginTop:8 }}>{msg}</div>}
        </form>
      </div>
    </div>
  );
}