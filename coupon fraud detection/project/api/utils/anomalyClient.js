const fetch = (...args) => import('node-fetch').then(({ default: f }) => f(...args));

// Env compat: prefer ML_URL/USE_ML, but accept ML_BASE_URL/ML_ENABLED too
const ML_URL =
  process.env.ML_URL ||
  process.env.ML_BASE_URL ||
  'http://127.0.0.1:8000';
const USE_ML =
  (process.env.USE_ML ?? process.env.ML_ENABLED ?? 'true') === 'true';
const ML_TIMEOUT_MS = parseInt(process.env.ML_TIMEOUT_MS || '2000', 10);

/**
 * getAnomalyScore(ctx): primary API
 * Expects the enriched context object from utils/enrich.js:
 *   ctx.couponValue, ctx.acctAgeHours, ctx.counters.device_redemptions24h, ctx.counters.ip_uniqueAccounts10m
 * Returns a Number in [0.0, 1.0]. Falls back to 0.0 if ML is disabled/unavailable.
 */
async function getAnomalyScore(ctx = {}) {
  if (!USE_ML) return 0.0;

  const payload = {
    couponValue: Number(ctx.couponValue ?? 0),
    acctAgeHours: Number(ctx.acctAgeHours ?? 0),
    device_redemptions24h: Number(ctx.counters?.device_redemptions24h ?? 0),
    ip_uniqueAccounts10m: Number(ctx.counters?.ip_uniqueAccounts10m ?? 0),
  };

  let controller;
  let timer;
  try {
    if (typeof AbortController !== 'undefined') {
      controller = new AbortController();
      timer = setTimeout(() => controller.abort(), ML_TIMEOUT_MS);
    }
    const res = await fetch(`${ML_URL}/score`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller?.signal,
    });
    if (timer) clearTimeout(timer);

    if (!res.ok) {
      console.warn('[ML] non-OK response:', res.status);
      return 0.0;
    }
    const data = await res.json();
    return typeof data.score === 'number' ? data.score : 0.0;
  } catch (e) {
    if (e?.name === 'AbortError') {
      console.warn(`[ML] timeout after ${ML_TIMEOUT_MS} ms`);
    } else {
      console.warn('[ML] scorer error:', e.message);
    }
    return 0.0;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * scoreAnomaly(features): backward-compatible wrapper
 * Accepts either an array [couponValue, acctAgeHours, device24h, ip10m]
 * or an object with the same named properties. Returns { score, top }.
 */
async function scoreAnomaly(features) {
  if (!USE_ML) return { score: 0, top: [] };

  let payload;
  if (Array.isArray(features)) {
    payload = {
      couponValue: Number(features[0] ?? 0),
      acctAgeHours: Number(features[1] ?? 0),
      device_redemptions24h: Number(features[2] ?? 0),
      ip_uniqueAccounts10m: Number(features[3] ?? 0),
    };
  } else if (features && typeof features === 'object') {
    payload = {
      couponValue: Number(features.couponValue ?? 0),
      acctAgeHours: Number(features.acctAgeHours ?? 0),
      device_redemptions24h: Number(features.device_redemptions24h ?? 0),
      ip_uniqueAccounts10m: Number(features.ip_uniqueAccounts10m ?? 0),
    };
  } else {
    payload = {
      couponValue: 0,
      acctAgeHours: 0,
      device_redemptions24h: 0,
      ip_uniqueAccounts10m: 0,
    };
  }

  let controller;
  let timer;
  try {
    if (typeof AbortController !== 'undefined') {
      controller = new AbortController();
      timer = setTimeout(() => controller.abort(), ML_TIMEOUT_MS);
    }
    const res = await fetch(`${ML_URL}/score`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller?.signal,
    });
    if (timer) clearTimeout(timer);

    if (!res.ok) {
      return { score: 0, top: [] };
    }
    const data = await res.json();
    return {
      score: typeof data.score === 'number' ? data.score : 0,
      top: Array.isArray(data.top) ? data.top : [],
    };
  } catch (e) {
    if (timer) clearTimeout(timer);
    console.error('ML score error', e.message);
    return { score: 0, top: [] };
  }
}

module.exports = { getAnomalyScore, scoreAnomaly };
