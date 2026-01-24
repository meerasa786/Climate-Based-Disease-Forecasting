const { Schema, model, Types } = require('mongoose');

const CouponCodeSchema = new Schema({
    couponId: { type: Types.ObjectId, ref: 'Coupon', index: true },
    code: { type: String, unique: true, index: true },
    assignedTo: { type: Types.ObjectId, ref: 'User' }, 
    usedBy: { type: Types.ObjectId, ref: 'User' },  
    usedAt: Date
}, { timestamps: true });

module.exports = model('CouponCode', CouponCodeSchema);
