const router = require('express').Router();
const mongoose = require('mongoose');

const User = require('../../models/User');
const Coupon = require('../../models/Coupon');
const CouponCode = require('../../models/CouponCode');
const Redemption = require('../../models/Redemption');
const RulesConfig = require('../../models/RulesConfig');
const List = require('../../models/List');

const { enrichEvent } = require('../../utils/enrich');
const { evaluateRules } = require('../../utils/rulesEngine');
const { narrateDecision } = require('../../utils/narrator');
const { getAnomalyScore } = require('../../utils/anomalyClient');
const { userAuth } = require('../../middleware/auth');

const AUTO_REMEDIATE = (process.env.AUTO_REMEDIATE || 'true') === 'true';

/**
 * POST /api/public/redemptions
 *
 * Expects an authenticated customer (via userAuth / JWT).
 * Body:
 *  - couponCode (preferred) OR couponId
 *  - orderTotal (number) OR amount (number)
 *  - orderId (string, optional but recommended)
 *  - deviceId (optional; falls back to header / synthetic)
 *  - ip (optional; falls back to request IP)
 *  - failedCouponAttempts10m (optional, number)
 */
router.post('/', userAuth, async (req, res) => {
  try {
    const {
      couponId: bodyCouponId,
      couponCode: bodyCouponCode,
      orderId,
      amount,
      orderTotal,
      deviceId: bodyDeviceId,
      ip: bodyIp,
      failedCouponAttempts10m,
    } = req.body || {};

    // 1) Resolve user from JWT
    const userJwt = req.user || {};
    const userIdRaw = userJwt.sub || userJwt._id || userJwt.id;
    const userEmailRaw = userJwt.email;

    if (!userIdRaw || !userEmailRaw) {
      return res.status(401).json({ ok: false, msg: 'auth_required_user_missing' });
    }

    const userId = new mongoose.Types.ObjectId(userIdRaw);
    const userEmail = String(userEmailRaw).trim().toLowerCase();

    // 2) Determine amount
    const amt = typeof amount === 'number' ? amount : Number(orderTotal);
    if (!bodyCouponId && !bodyCouponCode) {
      return res.status(400).json({ ok: false, msg: 'couponCode or couponId is required' });
    }
    if (!amt || Number.isNaN(amt) || amt <= 0) {
      return res.status(400).json({ ok: false, msg: 'amount/orderTotal must be a positive number' });
    }

    // orderId is nice for audit but not strictly required
    const finalOrderId = orderId || `order-${Date.now()}`;

    // 3) Resolve deviceId and ip (from body, headers, or fallbacks)
    const ip =
      bodyIp ||
      (req.headers['x-forwarded-for'] && req.headers['x-forwarded-for'].toString().split(',')[0].trim()) ||
      req.ip ||
      req.connection?.remoteAddress ||
      null;

    const headerDevice =
      req.headers['x-device-id'] ||
      req.headers['x-client-id'] ||
      null;

    const deviceId = bodyDeviceId || headerDevice || `device-${userId.toString()}`;

    // 4) Ensure user exists in DB
  await User.updateOne(
    { _id: userId },
    {
      $setOnInsert: {
        _id: userId,
        // only fields that are *only* set on first insert
        createdAt: userJwt.createdAt || new Date(),
      },
      $set: {
        // fields we want to keep updated
        email: userEmail,
      },
    },
    { upsert: true },
  );

    // 5) Look up coupon via couponId or couponCode
    let coupon;
    let couponId = bodyCouponId ? new mongoose.Types.ObjectId(bodyCouponId) : null;
    let couponCode = bodyCouponCode || null;

    if (couponId) {
      coupon = await Coupon.findById(couponId).lean();
      if (!coupon) {
        return res.status(400).json({ ok: false, msg: 'Invalid couponId' });
      }
    } else {
      // resolve from couponCode
      const codeDoc = await CouponCode.findOne({ code: couponCode }).lean();
      if (!codeDoc) {
        return res.status(400).json({ ok: false, msg: 'Invalid couponCode' });
      }
      couponId = codeDoc.couponId || codeDoc.coupon || codeDoc.couponId;
      coupon = await Coupon.findById(couponId).lean();
      if (!coupon) {
        return res.status(400).json({ ok: false, msg: 'Coupon not found for code' });
      }
    }

    if (!coupon || coupon.status !== 'active') {
      return res.status(400).json({ ok: false, msg: 'Invalid or inactive coupon' });
    }

    const now = new Date();

    // 6) whitelist / blocklist overrides
    const white = await List.findOne({
      type: 'white',
      value: { $in: [userEmail, deviceId, ip] },
    }).lean();
    const block = await List.findOne({
      type: 'block',
      value: { $in: [userEmail, deviceId, ip] },
      $or: [{ expiresAt: null }, { expiresAt: { $gt: now } }],
    }).lean();

    // 7) Enforce single-use & maxRedemptions
    if (coupon.singleUse) {
      if (!couponCode) {
        return res.status(400).json({ ok: false, msg: 'couponCode is required for single-use coupon' });
      }
      const upd = await CouponCode.updateOne(
        {
          code: couponCode,
          couponId: coupon._id,
          $or: [{ usedBy: null }, { usedBy: { $exists: false } }],
        },
        { $set: { usedBy: userId, usedAt: new Date() } },
      );
      if (upd.matchedCount === 0 || upd.modifiedCount === 0) {
        return res.status(400).json({ ok: false, msg: 'Single-use code already used or invalid' });
      }
    }

    if (coupon.maxRedemptions) {
      const usedCount = await Redemption.countDocuments({ couponId: coupon._id }).exec();
      if (usedCount >= coupon.maxRedemptions) {
        return res.status(400).json({ ok: false, msg: 'Coupon redemption cap reached' });
      }
    }

    // 8) compute enrichment
    const enrichment = await enrichEvent({ userId, deviceId, ip, amount: amt });

    // 9) build evaluation context
    const couponValue =
      coupon.type === 'fixed'
        ? coupon.value
        : +(amt * (coupon.value / 100)).toFixed(2);

    const cfg = await RulesConfig.findOne({ enabled: true }).sort({ createdAt: -1 }).lean();
    const ctx = {
      user: { _id: userId, email: userEmail },
      coupon,
      couponValue,
      acctAgeHours: enrichment.acctAgeHours,
      counters: enrichment.counters,
      deviceId,
      ip,
      failedCouponAttempts10m: failedCouponAttempts10m || 0,
    };

    // 10) evaluate rules → points
    const { hits, rulesPoints } = evaluateRules(ctx, cfg);

    // 11) anomaly score (optional ML) + combined risk
    const anomScore = await getAnomalyScore({
      couponValue,
      acctAgeHours: enrichment.acctAgeHours,
      counters: enrichment.counters,
    });
    const anomaly = { score: anomScore, top: [] };
    const w_anom = cfg && typeof cfg.w_anom === 'number' ? cfg.w_anom : 0.3;
    const riskRaw = (rulesPoints || 0) + w_anom * (anomaly.score || 0);
    const risk = Number(Math.max(0, Math.min(1, riskRaw)).toFixed(4));

    // 12) policy thresholds
    const blockRisk = cfg?.thresholds?.blockRisk ?? 0.8;
    const challengeRisk = cfg?.thresholds?.challengeRisk ?? 0.6;

    let decision = 'ALLOW';
    if (white) {
      decision = 'ALLOW';
    } else if (block) {
      decision = 'BLOCK';
    } else if (risk >= blockRisk || hits.some(h => h.type === 'hard')) {
      decision = 'BLOCK';
    } else if (risk >= challengeRisk) {
      decision = 'CHALLENGE';
    }

    // 13) narration (WHY)
    const narration = narrateDecision({ decision, risk, rulesHits: hits, anomaly });

    // 14) persist redemption (AUDIT)
    const doc = await Redemption.create({
      userId,
      couponId: coupon._id,
      couponCode,
      orderId: finalOrderId,
      amount: amt,
      deviceId,
      ip,
      geo: enrichment.geo,
      acctAgeHours: enrichment.acctAgeHours,
      counters: enrichment.counters,
      rulesHits: hits,
      rulesPoints,
      anomaly,
      risk,
      decision,
      narration,
      entities: { user: userEmail, device: deviceId, ip },
    });

    // 15) auto-remediate if any hard rule fired
    if (AUTO_REMEDIATE && hits.some(h => h.type === 'hard') && !white) {
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000);
      const existing = await List.findOne({
        type: 'block',
        entityType: 'device',
        value: deviceId,
        $or: [{ expiresAt: null }, { expiresAt: { $gt: now } }],
      });
      if (!existing) {
        await List.create({
          type: 'block',
          entityType: 'device',
          value: deviceId,
          reason: 'auto-hard-rule',
          expiresAt,
        });
      }
    }

    return res.json({
      ok: true,
      decision,
      risk,
      anomalyScore: anomaly.score,
      narration,
      hits,
      redemptionId: doc._id,
    });
  } catch (e) {
    console.error('[redemptions] error:', e);
    return res.status(500).json({ ok: false, msg: 'Internal error' });
  }
});

module.exports = router;
