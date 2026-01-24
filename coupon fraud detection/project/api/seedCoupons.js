// api/seedCoupons.js
require('dotenv').config();
const mongoose = require('mongoose');
const {connectMongo} = require('./db/mongo'); // your existing mongo.js connector
const Coupon = require('./models/Coupon');
const CouponCode = require('./models/CouponCode');

async function main() {
  try {
    await connectMongo();
    console.log('Connected to Mongo');

    // 1) Create a simple fixed-value coupon
    const welcomeCoupon = await Coupon.create({
      name: 'WELCOME10',
      description: '₹10 off for demo',
      type: 'fixed',         // or "fixed_amount" if your schema uses that
      value: 10,
      singleUse: false,
      maxRedemptions: 100,
      status: 'active',
    });

    // 2) Create a high-value coupon to trigger new_acct_high_value
    const highCoupon = await Coupon.create({
      name: 'BIG50',
      description: '₹50 off high value demo',
      type: 'fixed',
      value: 50,
      singleUse: false,
      maxRedemptions: 50,
      status: 'active',
    });

    // 3) Create codes for each coupon
    const codes = await CouponCode.insertMany([
      {
        code: 'WELCOME10',
        couponId: welcomeCoupon._id,
        usedBy: null,
        usedAt: null,
      },
      {
        code: 'BIG50',
        couponId: highCoupon._id,
        usedBy: null,
        usedAt: null,
      },
    ]);

    console.log('Seeded coupons:');
    console.log('WELCOME10 ->', welcomeCoupon._id.toString());
    console.log('BIG50     ->', highCoupon._id.toString());
    console.log('Codes:', codes.map(c => ({ code: c.code, couponId: c.couponId })));

    await mongoose.connection.close();
    console.log('Done, connection closed.');
    process.exit(0);
  } catch (err) {
    console.error('Seed error:', err);
    process.exit(1);
  }
}

main();