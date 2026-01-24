
const { Schema, model } = require('mongoose');

const RulesConfigSchema = new Schema(
  {
    version: { type: Number },                 
    enabled: { type: Boolean, default: true },

    w_anom: { type: Number, default: 0.3, min: 0, max: 1 },

    thresholds: {
      blockRisk: { type: Number, default: 0.8, min: 0, max: 1 },
      challengeRisk: { type: Number, default: 0.6, min: 0, max: 1 }
    },

    rules: [
      {
        id: { type: String, required: true }, 
        points: { type: Number, default: 0 },  
        type: { type: String, default: 'soft', enum: ['soft', 'hard'] },
        params: { type: Schema.Types.Mixed, default: {} } 
      }
    ],

    createdAt: { type: Date, default: Date.now }
  },
  { minimize: false }
);

RulesConfigSchema.path('thresholds.blockRisk').validate(function (b) {
  const c = typeof this.thresholds?.challengeRisk === 'number' ? this.thresholds.challengeRisk : 0.6;
  return b > c;
}, 'thresholds.blockRisk must be greater than thresholds.challengeRisk');

RulesConfigSchema.index({ enabled: 1, createdAt: -1 });


RulesConfigSchema.pre('save', async function (next) {
  if (this.version != null) return next();
  try {
    const latest = await this.constructor.findOne({}).sort({ version: -1, createdAt: -1 }).lean().exec();
    this.version = latest?.version != null ? latest.version + 1 : 1;
    next();
  } catch (e) {
    next(e);
  }
});

module.exports = model('RulesConfig', RulesConfigSchema);