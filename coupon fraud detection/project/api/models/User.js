const { Schema, model } = require('mongoose');

const UserSchema = new Schema({
  email: { type: String, required: true, lowercase: true, trim: true, index: true, unique: true },
  passwordHash: { type: String },                   
  role: { type: String, enum: ['user','admin'], default: 'user' },
  createdAt: { type: Date, default: Date.now },
}, { minimize:false });

module.exports = model('User', UserSchema);