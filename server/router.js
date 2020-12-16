const express = require('express');
const storage = require('./storage.js');

// Create router.
const router = express.Router();

/**
 * Create response body for POST request.
 * @param {boolean} bSuccess: Success flag.
 * @param {Array} aMetrics: Array of metric entries.
 * @returns {object} Response body.
 * @private
 */
function _createResponseBody(bSuccess, aMetrics) {
    const oBody = {
        success: bSuccess
    };
    if (aMetrics) {
        oBody['payload'] = aMetrics;
    }
    return oBody;
}

/**
 * Create response headers for GET request.
 * @returns {object} Response headers.
 * @private
 */
function _createGetResponseHeaders() {
    return {
        'Connection': 'keep-alive',
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache'
    }
}

/**
 * Convert response body object to event stream string.
 * @param {object} oBody: Response body object.
 * @returns {string} Event stream string.
 * @private
 */
function _convertToEventStream(oBody) {
    return `data: ${JSON.stringify(oBody)}\n\n`;
}

// Get previously stored data provided by 'on_batch' Keras callback.
router.get('/batch/:id', (req, res) => {
    const oResponseHeaders = _createGetResponseHeaders();
    res.set(oResponseHeaders);
    const sId = req.params.id;
    const aBatchMetrics = storage.loadBatchMetrics(sId);
    const oResponseBody = _createResponseBody(true, aBatchMetrics);
    const sBody = _convertToEventStream(oResponseBody);
    res.send(sBody);
});

// Handle data from 'on_batch' Keras callback
// provided by POST request from 'KeruviRemoteMonitor'.
router.post('/batch', (req, res) => {
    const oRequestBody = req.body;
    const bSaveBatchResult = storage.saveBatchMetrics(oRequestBody.id, oRequestBody.metrics);
    const oResponseBody = _createResponseBody(bSaveBatchResult);
    res.json(oResponseBody);
});

// Get previously stored data provided by 'on_epoch' Keras callback.
router.get('/epoch/:id', (req, res) => {
    const oResponseHeaders = _createGetResponseHeaders();
    res.set(oResponseHeaders);
    const sId = req.params.id;
    const aEpochMetrics = storage.loadEpochMetrics(sId);
    const oResponseBody = _createResponseBody(true, aEpochMetrics);
    const sBody = _convertToEventStream(oResponseBody);
    res.send(sBody);
});

// Handle data from 'on_epoch' Keras callback
// provided by POST request from 'KeruviRemoteMonitor'.
router.post('/epoch', (req, res) => {
    const oRequestBody = req.body;
    const bSaveEpochResult = storage.saveEpochMetrics(oRequestBody.id, oRequestBody.metrics);
    const oResponseBody = _createResponseBody(bSaveEpochResult);
    res.json(oResponseBody);
});

// Get previously stored data provided by 'on_train' Keras callback.
router.get('/train/:id', (req, res) => {
    const oResponseHeaders = _createGetResponseHeaders();
    res.set(oResponseHeaders);
    const sId = req.params.id;
    const aTrainMetrics = storage.loadTrainMetrics(sId)
    const oResponseBody = _createResponseBody(true, aTrainMetrics)
    const sBody = _convertToEventStream(oResponseBody);
    res.send(sBody);
});

// Handle data from 'on_train' Keras callback
// provided by POST request from 'KeruviRemoteMonitor'.
router.post('/train', (req, res) => {
    const oRequestBody = req.body;
    const bSaveTrainResult = storage.saveTrainMetrics(oRequestBody.id, oRequestBody.metrics);
    const oResponseBody = _createResponseBody(bSaveTrainResult);
    res.json(oResponseBody);
});

module.exports = router;