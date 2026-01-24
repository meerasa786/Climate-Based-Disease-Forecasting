
/**
 * Human-friendly narration and WHY list for decisions.
 * - Supports both { rulesHits } and { hits } (either name is fine).
 * - Prints clear rule reasons using friendly labels.
 * - Mentions anomaly score only when it is meaningful.
 */

const ANOMALY_MENTION_THRESHOLD =
  Number(process.env.ANOMALY_MENTION_THRESHOLD || 0.6); // show ML only if >= 0.6

// Map internal rule ids to friendly text
const RULE_LABELS = {
  new_acct_high_value: 'new account used high-value coupon',
  device_duplicate: 'too many redemptions from same device (24h)',
  ip_burst: 'many unique accounts from the same IP (10m)',
};

const FEATURE_LABELS = {
  acctAgeHours: 'account age',
  device24h: 'device activity (24h)',
  ip10m: 'IP fan-out (10m)',
  user24h: 'user activity (24h)',
  value: 'coupon value',
};

/**
 * Build an array of short "why" reasons.
 * @param {{hits?: Array, rulesHits?: Array, anomaly?: {score?:number, top?:string[]}}} args
 * @returns {string[]}
 */
function buildWhyList({ hits, rulesHits, anomaly } = {}) {
  const list = [];
  const ruleHits = Array.isArray(rulesHits) ? rulesHits : Array.isArray(hits) ? hits : [];

  if (ruleHits.length) {
    for (const h of ruleHits) {
      const id = h.id || h.ruleId || 'rule';
      const label = RULE_LABELS[id] || id;
      const d = h.detail || {};
      const extras = Object.keys(d)
        .map((k) => `${k}=${d[k]}`)
        .join(', ');
      list.push(extras ? `${label} (${extras})` : label);
    }
  } else {
    list.push('no rule hits');
  }

  if (anomaly && typeof anomaly.score === 'number' && anomaly.score >= ANOMALY_MENTION_THRESHOLD) {
    let msg = `anomalous behavior (score ${anomaly.score.toFixed(2)})`;
    const top = Array.isArray(anomaly.top) ? anomaly.top : [];
    const topPretty = top
      .map((k) => FEATURE_LABELS[k] || k)
      .slice(0, 3)
      .join(', ');
    if (topPretty) msg += `; drivers: ${topPretty}`;
    list.push(msg);
  }

  return list;
}

/**
 * Produce a single narration string.
 * @param {{ decision:string, risk:number, rulesHits?:Array, hits?:Array, anomaly?:{score?:number, top?:string[]} }} args
 * @returns {string}
 */
function narrateDecision({ decision, risk, rulesHits, hits, anomaly } = {}) {
  const why = buildWhyList({ rulesHits, hits, anomaly });
  const riskStr = isFinite(risk) ? Number(risk).toFixed(2) : '0.00';
  return `${cap(decision)} (risk ${riskStr}): ${why.join('; ')}`;
}

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

module.exports = { narrateDecision, buildWhyList };
