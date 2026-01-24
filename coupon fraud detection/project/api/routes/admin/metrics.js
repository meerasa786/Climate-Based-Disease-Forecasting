const router = require('express').Router();
const Redemption = require('../../models/Redemption');

/**
 * GET /api/admin/metrics/cards?hours=24
 * Returns counts for the dashboard cards in a given lookback window.
 */
router.get('/cards', async (req, res) => {
  const hours = Math.max(1, Math.min(24*30, parseInt(req.query.hours || '24', 10))); // cap at 30 days
  const since = new Date(Date.now() - hours * 3600 * 1000);

    const [ total, allowed, challenged, blocked ] = await Promise.all([
        Redemption.countDocuments({ createdAt: { $gte: since } }),
        Redemption.countDocuments({ createdAt: { $gte: since }, decision: 'allow' }),
        Redemption.countDocuments({ createdAt: { $gte: since }, decision: 'challenge' }),
        Redemption.countDocuments({ createdAt: { $gte: since }, decision: 'block' })
    ]);

    res.json({
        ok: true,
        windowHours: hours,
        total, allowed, challenged, blocked,
        blockRate: total ? +(blocked/total).toFixed(3) : 0,
        challengeRate: total ? +(challenged/total).toFixed(3) : 0
    });
});

/**
 * GET /api/admin/metrics/top-rules?hours=24&limit=10
 * Unwinds rulesHits and returns the most frequent rule ids in the window.
 */
router.get('/top-rules', async (req, res) => {
    const hours = Math.max(1, Math.min(24*30, parseInt(req.query.hours || '24', 10)));
    const limit = Math.max(1, Math.min(50, parseInt(req.query.limit || '10', 10)));
    const since = new Date(Date.now() - hours * 3600 * 1000);

    const agg = await Redemption.aggregate([
        { $match: { createdAt: { $gte: since }, rulesHits: { $exists: true, $ne: [] } } },
        { $unwind: '$rulesHits' },
        { $group: { _id: '$rulesHits.id', count: { $sum: 1 } } },
        { $sort: { count: -1 } },
        { $limit: limit }
    ]);

    res.json({ ok: true, windowHours: hours, top: agg });
});

module.exports = router;
