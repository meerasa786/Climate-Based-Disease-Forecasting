const { Schema, model } = require('mongoose');

const CouponSchema = new Schema({
    name: { type: String, required: true },
    code: { type: String, required: true, unique: true, index: true }, 
    type: { type: String, enum: ['percent','fixed'], required: true },
    value: { type: Number, required: true },
    singleUse: { type: Boolean, default: false },
    startAt: Date,
    endAt: Date,
    maxRedemptions: Number,
    minOrder: Number,
    status: { type: String, enum: ['draft','active','paused','archived'], default: 'draft' },
    meta: Schema.Types.Mixed
}, { timestamps: true });

module.exports = model('Coupon', CouponSchema);
