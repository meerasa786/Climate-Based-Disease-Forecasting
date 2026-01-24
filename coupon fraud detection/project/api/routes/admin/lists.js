const router = require('express').Router();
const List = require('../../models/List');


router.get('/', async (req, res) => {
  const { type, entityType, active } = req.query || {};
  const q = {};
  if (type) q.type = type;
  if (entityType) q.entityType = entityType;
  
  if (active === 'true') {
    q.$or = [{ expiresAt: null }, { expiresAt: { $gt: new Date() } }];
  }
  if (active === 'false') {
    q.expiresAt = { $lte: new Date() };
  }

  const items = await List.find(q).sort({ createdAt: -1 }).lean();
  res.json({ ok: true, total: items.length, items });
});


router.post('/', async (req, res) => {
    const { type, entityType, value, reason, expiresAt, ttlHours } = req.body || {};
    if (!type || !['block','white'].includes(type)) {
        return res.status(400).json({ ok:false, msg:'type must be "block" or "white"' });
    }
    if (!entityType || !['email','device','ip','address','payment'].includes(entityType)) {
        return res.status(400).json({ ok:false, msg:'invalid entityType' });
    }
    if (!value || typeof value !== 'string') {
        return res.status(400).json({ ok:false, msg:'value is required (string)' });
    }

    let exp = null;
    if (expiresAt) {
        const d = new Date(expiresAt);
        if (isNaN(d.getTime())) return res.status(400).json({ ok:false, msg:'expiresAt must be an ISO date string' });
        exp = d;
    } else if (typeof ttlHours === 'number' && ttlHours > 0) {
        exp = new Date(Date.now() + ttlHours * 3600 * 1000);
    }

    const existing = await List.findOne({
        type, entityType, value,
        $or: [{ expiresAt: null }, { expiresAt: { $gt: new Date() } }]
    });

    let doc;
    if (existing) {
        existing.reason = reason ?? existing.reason;
        existing.expiresAt = exp ?? existing.expiresAt ?? null;
        doc = await existing.save();
    } else {
        doc = await List.create({ type, entityType, value, reason, expiresAt: exp || null });
    }
    res.json({ ok:true, item: doc });
    });


    router.post('/:id/revert', async (req, res) => {
    const { id } = req.params;
    const deleted = await List.findByIdAndDelete(id);
    if (!deleted) return res.status(404).json({ ok:false, msg:'List item not found' });
    res.json({ ok:true, reverted: true });
    });

module.exports = router;
