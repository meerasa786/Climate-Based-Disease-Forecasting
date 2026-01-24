const router = require('express').Router();
const Coupon = require('../../models/Coupon');
const CouponCode = require('../../models/CouponCode');

/**
 * POST /api/public/checkout/apply-coupon
 */
router.post('/apply-coupon', async (req, res) => {
  try {
    const { code, orderAmount } = req.body || {};
    if (!code || typeof orderAmount !== 'number') {
      return res.status(400).json({ ok: false, msg: 'code and orderAmount are required' });
    }

    // Try resolving as a single-use CouponCode first
    let couponDoc = null;
    let singleUseMode = false;

    const cc = await CouponCode.findOne({ code }).populate('couponId').lean();
    if (cc && cc.couponId) {
      couponDoc = cc.couponId;
      singleUseMode = true;
      // ensure parent coupon is indeed singleUse
      if (!couponDoc.singleUse) {
        return res.status(400).json({ ok: false, msg: 'This code is not valid for single-use mode' });
      }
      // check if this specific code was already used
      if (cc.usedBy) {
        return res.status(400).json({ ok: false, msg: 'Code already used' });
      }
    }

    //  If not found as single-use, try multi-use 
    if (!couponDoc) {
      couponDoc = await Coupon.findOne({ code }).lean();
      if (!couponDoc) {
        return res.status(404).json({ ok: false, msg: 'Coupon not found' });
      }
      singleUseMode = !!couponDoc.singleUse;
      if (singleUseMode) {
        // coupon is configured as singleUse, but we didn’t find a matching per-code entry
        return res.status(400).json({ ok: false, msg: 'Single-use code not recognized' });
      }
    }

    // Basic gating checks
    if (couponDoc.status !== 'active') {
      return res.status(400).json({ ok: false, msg: 'Coupon is not active' });
    }
    if (couponDoc.startAt && Date.now() < new Date(couponDoc.startAt).getTime()) {
      return res.status(400).json({ ok: false, msg: 'Coupon not started yet' });
    }
    if (couponDoc.endAt && Date.now() > new Date(couponDoc.endAt).getTime()) {
      return res.status(400).json({ ok: false, msg: 'Coupon expired' });
    }
    if (couponDoc.minOrder && orderAmount < couponDoc.minOrder) {
      return res.status(400).json({ ok: false, msg: `Minimum order must be ≥ ${couponDoc.minOrder}` });
    }


    // Calculate discount and final amount
    let discount = 0;
    if (couponDoc.type === 'fixed') discount = couponDoc.value;
    if (couponDoc.type === 'percent') discount = +(orderAmount * (couponDoc.value / 100)).toFixed(2);
    const finalAmount = Math.max(0, +(orderAmount - discount).toFixed(2));

    return res.json({
      ok: true,
      couponId: couponDoc._id,
      singleUse: singleUseMode,
      discount,
      finalAmount
    });
  } catch (e) {
    console.error('[apply-coupon] error:', e);
    return res.status(500).json({ ok: false, msg: 'Internal error' });
  }
});

module.exports = router;
