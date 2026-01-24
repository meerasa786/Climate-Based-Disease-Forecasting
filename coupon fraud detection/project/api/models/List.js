const { Schema, model } = require('mongoose');

const ListSchema = new Schema({
  type: { type: String, enum: ['block','white'], index: true },
  entityType: { type: String, enum: ['email','device','ip','address','payment'], index: true },
  value: { type: String, index: true },
  reason: String,
  expiresAt: Date, 
  createdBy: String
}, { timestamps: true });

module.exports = model('List', ListSchema);
