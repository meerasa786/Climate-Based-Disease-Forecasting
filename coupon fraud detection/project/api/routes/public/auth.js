// api/routes/public/auth.js
const router = require('express').Router();
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../../models/User');

const JWT_SECRET = process.env.JWT_SECRET || 'dev-secret';
const BCRYPT_ROUNDS = parseInt(process.env.BCRYPT_ROUNDS || '10', 10);

function signToken(u) {
  return jwt.sign(
    { sub: u._id.toString(), role: u.role, email: u.email },
    JWT_SECRET,
    { expiresIn: '7d' }
  );
}

/**
 * POST /api/public/auth/register
 * Body: { email, password }
 * Creates a normal customer user (role: "user").
 */
router.post('/register', async (req, res) => {
  try {
    const email = String(req.body?.email || '').trim().toLowerCase();
    const password = String(req.body?.password || '');

    if (!email || !password) {
      return res.status(400).json({ ok: false, msg: 'email_and_password_required' });
    }

    let u = await User.findOne({ email }).exec();
    if (u && u.passwordHash) {
      return res.status(409).json({ ok: false, msg: 'user_exists_use_login' });
    }

    const passwordHash = await bcrypt.hash(password, BCRYPT_ROUNDS);
    if (!u) {
      u = await User.create({ email, passwordHash, role: 'user' });
    } else {
      u.role = 'user';
      u.passwordHash = passwordHash;
      await u.save();
    }

    const token = signToken(u);
    return res.json({ ok: true, token, user: { _id: u._id, email: u.email, role: u.role } });
  } catch (e) {
    console.error('[public/register]', e);
    return res.status(500).json({ ok: false, msg: 'internal_error' });
  }
});

/**
 * POST /api/public/auth/login
 * Body: { email, password }
 */
router.post('/login', async (req, res) => {
  try {
    const email = String(req.body?.email || '').trim().toLowerCase();
    const password = String(req.body?.password || '');

    if (!email || !password) {
      return res.status(400).json({ ok: false, msg: 'email_and_password_required' });
    }

    const u = await User.findOne({ email, role: 'user' }).exec();
    if (!u || !u.passwordHash) {
      return res.status(401).json({ ok: false, msg: 'invalid_credentials' });
    }

    const ok = await bcrypt.compare(password, u.passwordHash);
    if (!ok) {
      return res.status(401).json({ ok: false, msg: 'invalid_credentials' });
    }

    const token = signToken(u);
    return res.json({ ok: true, token, user: { _id: u._id, email: u.email, role: u.role } });
  } catch (e) {
    console.error('[public/login]', e);
    return res.status(500).json({ ok: false, msg: 'internal_error' });
  }
});

module.exports = router;