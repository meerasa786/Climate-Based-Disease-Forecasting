const router = require('express').Router();

router.use('/auth', require('./auth'));
router.use('/checkout', require('./checkout'));
router.use('/redemptions', require('./redemptions'));

module.exports = router;
