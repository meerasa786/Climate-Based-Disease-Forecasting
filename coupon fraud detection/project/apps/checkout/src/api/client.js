// apps/checkout/src/api/client.js

// Point this to your Node API
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:4000/api';

// IMPORTANT: checkout has its own token key
const STORAGE_KEY = 'checkout_token';

let token = localStorage.getItem(STORAGE_KEY) || null;

export function setToken(nextToken) {
  token = nextToken || null;
  if (token) {
    localStorage.setItem(STORAGE_KEY, token);
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
}

export function getToken() {
  return token;
}

export async function api(path, { method = 'GET', body, auth = false } = {}) {
  const headers = { 'Content-Type': 'application/json' };

  if (auth && token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const opts = { method, headers };
  if (body) {
    opts.body = JSON.stringify(body);
  }

  const res = await fetch(`${API_BASE}${path}`, opts);

  let data = null;
  try {
    data = await res.json();
  } catch (_) {
    // ignore
  }

  if (!res.ok) {
    const msg =
      (data && (data.msg || data.error || data.message)) || 'Request failed';
    throw new Error(msg);
  }

  return data;
}