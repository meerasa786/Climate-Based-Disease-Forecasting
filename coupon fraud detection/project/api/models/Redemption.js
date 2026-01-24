const { Schema, model, Types } = require('mongoose');

const RedemptionSchema = new Schema({
  userId: { type: Types.ObjectId, ref: 'User', index: true },
  couponId: { type: Types.ObjectId, ref: 'Coupon', index: true },
  couponCode: String,    
  orderId: String,
  amount: Number,      
  deviceId: String,
  ip: String,
  geo: { country: String, city: String },
  acctAgeHours: Number,

  counters: {
    ip_uniqueAccounts10m: Number,
    device_redemptions24h: Number,
    user_redemptions24h: Number,
    order_amount: Number
  },

  rulesHits: [Schema.Types.Mixed],
  rulesPoints: Number,
  anomaly: { score: Number, top: [String] },
  risk: Number,
  decision: { type: String, enum: ['ALLOW', 'CHALLENGE', 'BLOCK'] },
  narration: String,  

  entities: {         
    user: String,
    device: String,
    ip: String,
    address: String,  
    payment: String   
  }
}, { timestamps: true });

module.exports = model('Redemption', RedemptionSchema);
