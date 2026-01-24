const router = require('express').Router();
const { adminAuth } = require('../../middleware/auth');

// Sub-routers
router.use('/auth', require('./auth'));
router.use('/coupons', adminAuth, require('./coupons'));
router.use('/lists', adminAuth, require('./lists'));
router.use('/rules-config', require('./rulesConfig'));


const RulesConfig = require('../../models/RulesConfig');
const Redemption = require('../../models/Redemption');


const DEFAULT_CONFIG = Object.freeze({
  w_anom: 0.3,
  thresholds: { blockRisk: 0.8, challengeRisk: 0.6 },
  rules: [],
});


async function getLatestEnabled() {
  return RulesConfig.findOne({ enabled: true })
    .sort({ createdAt: -1, version: -1 })
    .lean()
    .exec();
}


router.get('/config', adminAuth, async (req, res) => {
  try {
    const cfg = await getLatestEnabled();
    if (!cfg) {
      return res.json({ source: 'default', config: DEFAULT_CONFIG });
    }
    return res.json({ source: 'db', config: cfg });
  } catch (err) {
    console.error('[admin/config GET] error:', err);
    return res.status(500).json({ error: 'failed_to_fetch_config' });
  }
});


router.put('/config', adminAuth, async (req, res) => {
  try {
    const latest = await getLatestEnabled();
    const base = latest || DEFAULT_CONFIG;

    const body = req.body || {};
    const next = {
      enabled: true,
      w_anom: typeof body.w_anom === 'number' ? body.w_anom : base.w_anom,
      thresholds: {
        blockRisk: body?.thresholds?.blockRisk ?? base.thresholds?.blockRisk ?? 0.8,
        challengeRisk: body?.thresholds?.challengeRisk ?? base.thresholds?.challengeRisk ?? 0.6,
      },
      rules: Array.isArray(body.rules) ? body.rules : base.rules || [],
    };

    // Validation
    const bn = Number(next.thresholds.blockRisk);
    const cn = Number(next.thresholds.challengeRisk);
    const wn = Number(next.w_anom);

    if (!Number.isFinite(bn) || bn < 0 || bn > 1) {
      return res.status(400).json({ error: 'invalid_blockRisk_range_0_1' });
    }
    if (!Number.isFinite(cn) || cn < 0 || cn > 1) {
      return res.status(400).json({ error: 'invalid_challengeRisk_range_0_1' });
    }
    if (bn <= cn) {
      return res.status(400).json({ error: 'blockRisk_must_be_greater_than_challengeRisk' });
    }
    if (!Number.isFinite(wn) || wn < 0 || wn > 1) {
      return res.status(400).json({ error: 'invalid_w_anom_range_0_1' });
    }

    const doc = new RulesConfig({
      ...next,

      version: latest && typeof latest.version === 'number' ? latest.version + 1 : 1,
      createdAt: new Date(),
    });
    await doc.save();

    return res.json({ ok: true, config: doc });
  } catch (err) {
    console.error('[admin/config PUT] error:', err);
    return res.status(500).json({ error: 'failed_to_update_config' });
  }
});


router.get('/redemptions', adminAuth, async (req, res) => {
  try {
    const {
      limit: limitRaw,
      decision,
      couponCode,
      userId,
      deviceId,
      ip,
      from,
      to,
    } = req.query || {};

    const limit = Math.min(Math.max(parseInt(limitRaw || '100', 10), 1), 500);

    const q = {};
    if (decision) {
      const d = String(decision).toUpperCase();
      if (['ALLOW', 'CHALLENGE', 'BLOCK'].includes(d)) {
        q.decision = d;
      }
    }
    if (couponCode) q.couponCode = String(couponCode);
    if (userId) q.userId = String(userId);
    if (deviceId) q.deviceId = String(deviceId);
    if (ip) q.ip = String(ip);

    if (from || to) {
      q.createdAt = {};
      if (from) q.createdAt.$gte = new Date(from);
      if (to) q.createdAt.$lte = new Date(to);
    }

    const docs = await Redemption.find(q, {
      userId: 1,
      deviceId: 1,
      ip: 1,
      couponCode: 1,
      riskScore: 1,
      risk: 1,
      decision: 1,
      why: 1,
      narration: 1,
      createdAt: 1,
    })
      .sort({ createdAt: -1 })
      .limit(limit)
      .lean()
      .exec();

    const rows = docs.map((d) => {
      const riskVal =
        typeof d.riskScore === 'number'
          ? d.riskScore
          : typeof d.risk === 'number'
          ? d.risk
          : null;

      return {
        ...d,
        riskScore: riskVal,
        risk: riskVal,
        why: d.why || d.narration || null,
      };
    });

    return res.json({ ok: true, count: rows.length, rows });
  } catch (err) {
    console.error('[admin/redemptions GET] error:', err);
    return res.status(500).json({ error: 'failed_to_list_redemptions' });
  }
});

module.exports = router;
