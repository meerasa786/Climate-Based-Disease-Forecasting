/**
 * Evaluate configured rules on the given context.
 */
function evaluateRules(ctx, cfg) {
  const rules = (cfg && cfg.rules) || defaultRules();
  const hits = [];

  for (const r of rules) {
    const { id, points, type='soft', params={} } = r;
    let hit = false, detail = {};

    switch (id) {
      case 'new_acct_high_value':
        hit = ctx.acctAgeHours < (params.ageHours ?? 24) &&
              ctx.couponValue  > (params.minValue ?? 20);
        detail = { acctAgeHours: ctx.acctAgeHours, couponValue: ctx.couponValue };
        break;

      case 'device_duplicate':
        hit = ctx.counters.device_redemptions24h > (params.maxPerDevice24h ?? 5);
        detail = { device24h: ctx.counters.device_redemptions24h };
        break;

      case 'ip_burst':
        hit = ctx.counters.ip_uniqueAccounts10m > (params.maxAccounts10m ?? 8);
        detail = { ip10m: ctx.counters.ip_uniqueAccounts10m };
        break;

      case 'redemption_velocity':
        hit = ctx.counters.user_redemptions24h > (params.maxUser24h ?? 3);
        detail = { user24h: ctx.counters.user_redemptions24h };
        break;

      case 'code_guessing':
        hit = (ctx.failedCouponAttempts10m || 0) > (params.maxFailed10m ?? 6);
        detail = { failed10m: ctx.failedCouponAttempts10m || 0 };
        break;

      default:
        hit = false;
    }

    if (hit) hits.push({ id, score: points, type, detail });
  }

  const rulesPoints = hits.reduce((s, h) => s + h.score, 0);
  return { hits, rulesPoints };
}

/** Fallback default if no RulesConfig doc exists yet */
function defaultRules() {
  return [
    { id: 'new_acct_high_value', points: 0.4, type: 'soft', params: { ageHours: 24, minValue: 20 } },
    { id: 'device_duplicate',    points: 0.5, type: 'soft', params: { maxPerDevice24h: 5 } },
    { id: 'ip_burst',            points: 0.5, type: 'soft', params: { maxAccounts10m: 8 } },
    { id: 'redemption_velocity', points: 0.4, type: 'soft', params: { maxUser24h: 3 } },
    { id: 'code_guessing',       points: 0.8, type: 'hard', params: { maxFailed10m: 6 } }
  ];
}

module.exports = { evaluateRules };
