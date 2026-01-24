const jwt = require('jsonwebtoken');
const JWT_SECRET = process.env.JWT_SECRET || 'dev-secret';
function parseToken(req) {
  const h = req.headers.authorization || '';
  const t = h.replace(/^Bearer\s+/i, '');
  if (!t) return null;
  try { return jwt.verify(t, JWT_SECRET); } catch { return null; }
}

function adminAuth(req, res, next) {
  const p = parseToken(req);
  if (!p || p.role !== 'admin') return res.status(401).json({ ok:false, msg:'admin_auth_required' });
  req.user = p; next();
}
function userAuth(req, res, next) {
  const p = parseToken(req);
  if (!p || (p.role!=='user' && p.role!=='admin')) return res.status(401).json({ ok:false, msg:'user_auth_required' });
  req.user = p; next();
}
module.exports = { adminAuth, userAuth };