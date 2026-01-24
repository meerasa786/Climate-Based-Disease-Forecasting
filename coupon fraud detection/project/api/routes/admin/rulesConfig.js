// api/routes/admin/rulesConfig.js
const router = require('express').Router();
const RulesConfig = require('../../models/RulesConfig');

/**
 * Helper: build a safe default config if DB is empty.
 * Adjust defaults if your schema has more fields.
 */
function buildDefaultConfig() {
  return {
    name: 'Default risk config',
    enabled: true,
    thresholds: {
      blockRisk: 0.8,
      challengeRisk: 0.6,
    },
    w_anom: 0.3,
    rules: [
      {
        id: 'new_acct_high_value',
        label: 'New account using high-value coupon',
        type: 'soft', // can be "soft" or "hard"
        score: 0.4,
        params: {
          ageHours: 24,
          minValue: 20,
        },
      },
      {
        id: 'device_duplicate',
        label: 'Too many redemptions from same device (24h)',
        type: 'soft',
        score: 0.3,
        params: {
          maxPerDevice24h: 5,
        },
      },
      {
        id: 'ip_burst',
        label: 'Too many unique accounts from same IP (10m)',
        type: 'soft',
        score: 0.3,
        params: {
          maxAccounts10m: 8,
        },
      },
    ],
  };
}

/**
 * GET /api/admin/rules-config/active
 *
 * Returns the latest enabled config, or creates a default one if none exists.
 */
router.get('/active', async (req, res) => {
  try {
    let cfg = await RulesConfig.findOne({ enabled: true })
      .sort({ createdAt: -1 })
      .lean()
      .exec();

    if (!cfg) {
      const defaults = buildDefaultConfig();
      const created = await RulesConfig.create(defaults);
      cfg = created.toObject();
    }

    return res.json({ ok: true, config: cfg });
  } catch (err) {
    console.error('[rules-config] GET /active error:', err);
    return res.status(500).json({ ok: false, msg: 'Failed to load rules config' });
  }
});

/**
 * POST /api/admin/rules-config
 *
 * Creates a new config and (optionally) disables previous ones.
 * Body: { thresholds, w_anom, rules, name, enable }
 */
router.post('/', async (req, res) => {
  try {
    const {
      name,
      thresholds = {},
      w_anom,
      rules,
      enable = true,
      disableOthers = true,
    } = req.body || {};

    // basic sanitization
    const blockRisk = Math.min(
      1,
      Math.max(0, Number(thresholds.blockRisk ?? 0.8)),
    );
    const challengeRisk = Math.min(
      1,
      Math.max(0, Number(thresholds.challengeRisk ?? 0.6)),
    );
    const wAnom = w_anom != null ? Math.min(1, Math.max(0, Number(w_anom))) : 0.3;

    if (disableOthers) {
      await RulesConfig.updateMany({}, { $set: { enabled: false } }).exec();
    }

    const doc = await RulesConfig.create({
      name: name || 'Rules config ' + new Date().toISOString(),
      enabled: !!enable,
      thresholds: {
        blockRisk,
        challengeRisk,
      },
      w_anom: wAnom,
      // if your schema has a different field name for rules, adjust here
      rules,
    });

    return res.status(201).json({ ok: true, config: doc });
  } catch (err) {
    console.error('[rules-config] POST / error:', err);
    return res.status(500).json({ ok: false, msg: 'Failed to create rules config' });
  }
});

/**
 * PUT /api/admin/rules-config/:id
 *
 * Updates an existing config (e.g., thresholds, w_anom, rules).
 */
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const {
      name,
      thresholds,
      w_anom,
      rules,
      enabled,
    } = req.body || {};

    const update = {};

    if (name != null) update.name = name;
    if (enabled != null) update.enabled = !!enabled;

    if (thresholds) {
      const patch = {};
      if (thresholds.blockRisk != null) {
        patch['thresholds.blockRisk'] = Math.min(
          1,
          Math.max(0, Number(thresholds.blockRisk)),
        );
      }
      if (thresholds.challengeRisk != null) {
        patch['thresholds.challengeRisk'] = Math.min(
          1,
          Math.max(0, Number(thresholds.challengeRisk)),
        );
      }
      Object.assign(update, patch);
    }

    if (w_anom != null) {
      update.w_anom = Math.min(1, Math.max(0, Number(w_anom)));
    }

    if (rules != null) {
      update.rules = rules;
    }

    const doc = await RulesConfig.findByIdAndUpdate(
      id,
      { $set: update },
      { new: true },
    ).exec();

    if (!doc) {
      return res.status(404).json({ ok: false, msg: 'Config not found' });
    }

    return res.json({ ok: true, config: doc });
  } catch (err) {
    console.error('[rules-config] PUT /:id error:', err);
    return res.status(500).json({ ok: false, msg: 'Failed to update rules config' });
  }
});

module.exports = router;