const User = require('../models/User');
const Redemption = require('../models/Redemption');

async function enrichEvent({ userId, deviceId, ip, amount }) {
  // 1) account age
  const user = await User.findById(userId).lean();
  const acctAgeHours = user ? Math.max(0, (Date.now() - new Date(user.createdAt).getTime()) / 36e5) : 0;

  // 2) ip->geo
  const geo = { country: ip && ip.startsWith('10.') ? 'IN' : 'US', city: 'Demo' };

  // 3) rolling counters
  const now = new Date();
  const tenMinAgo = new Date(now.getTime() - 10*60*1000);
  const dayAgo = new Date(now.getTime() - 24*60*60*1000);

  // distinct users on same IP in last 10m
  const ipDistinctUsers = await Redemption.distinct('userId', { ip, createdAt: { $gte: tenMinAgo } });
  const ip_uniqueAccounts10m = ipDistinctUsers.length;

  // total redemptions on this device in 24h
  const device_redemptions24h = await Redemption.countDocuments({ deviceId, createdAt: { $gte: dayAgo } }).exec();

  // total redemptions by this user in 24h
  const user_redemptions24h = await Redemption.countDocuments({ userId, createdAt: { $gte: dayAgo } }).exec();

  return {
    acctAgeHours,
    geo,
    counters: {
      ip_uniqueAccounts10m,
      device_redemptions24h,
      user_redemptions24h,
      order_amount: amount
    }
  };
}

module.exports = { enrichEvent };
