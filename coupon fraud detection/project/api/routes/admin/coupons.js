// api/routes/admin/coupons.js
const router = require('express').Router();
const Coupon = require('../../models/Coupon');
const CouponCode = require('../../models/CouponCode');
const Redemption = require('../../models/Redemption'); // adjust path if different

// helper: attach stats per coupon using Redemption.couponCode
async function attachStats(coupons) {
  if (!coupons.length) return [];

  // group redemptions by couponCode string
  const agg = await Redemption.aggregate([
    {
      $group: {
        _id: '$couponCode',
        totalRedemptions: { $sum: 1 },
        lastRedemptionAt: { $max: '$createdAt' },
      },
    },
  ]).exec();

  const byCode = new Map(agg.map(a => [String(a._id), a]));

  return coupons.map(c => {
    const s = byCode.get(c.code);
    return {
      ...c,
      stats: {
        totalRedemptions: s ? s.totalRedemptions : 0,
        lastRedemptionAt: s ? s.lastRedemptionAt : null,
      },
    };
  });
}

// GET /api/admin/coupons
// list coupons with basic stats
router.get('/', async (req, res) => {
  try {
    const limit = Math.min(Number(req.query.limit) || 50, 100);
    const docs = await Coupon.find({})
      .sort({ createdAt: -1 })
      .limit(limit)
      .lean()
      .exec();

    const withStats = await attachStats(docs);

    return res.json({ ok: true, coupons: withStats });
  } catch (err) {
    console.error('[admin/coupons] GET / error:', err);
    return res.status(500).json({ ok: false, msg: 'Failed to load coupons' });
  }
});

// POST /api/admin/coupons
// create a new coupon
router.post('/', async (req, res) => {
  try {
    const {
      name,
      code,
      type,           // 'fixed' | 'percent'
      value,
      singleUse,
      maxRedemptions,
      minOrder,
      status,         // 'draft' | 'active' | 'paused' | 'archived'
      startAt,
      endAt,
    } = req.body || {};

    if (!name || !code || !type || value == null) {
      return res.status(400).json({
        ok: false,
        msg: 'name, code, type, value are required',
      });
    }

    const coupon = await Coupon.create({
      name: name.trim(),
      code: code.trim().toUpperCase(),
      type,
      value: Number(value),
      singleUse: !!singleUse,
      maxRedemptions: maxRedemptions ? Number(maxRedemptions) : undefined,
      minOrder: minOrder ? Number(minOrder) : undefined,
      status: status || 'draft',
      startAt: startAt ? new Date(startAt) : undefined,
      endAt: endAt ? new Date(endAt) : undefined,
    });

    // also create a CouponCode entry for this public code so redemption flow can find it
    try {
      await CouponCode.create({
        couponId: coupon._id,
        code: coupon.code,
      });
    } catch (codeErr) {
      console.error('[admin/coupons] failed to create CouponCode:', codeErr);
      // do not fail the whole request if CouponCode already exists
      if (codeErr.code !== 11000) {
        throw codeErr;
      }
    }

    return res.status(201).json({ ok: true, coupon });
  } catch (err) {
    console.error('[admin/coupons] POST / error:', err);
    // duplicate code etc.
    if (err.code === 11000) {
      return res
        .status(409)
        .json({ ok: false, msg: 'Coupon code already exists' });
    }
    return res.status(500).json({ ok: false, msg: 'Failed to create coupon' });
  }
});

module.exports = router;