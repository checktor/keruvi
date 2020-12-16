const BATCH = 0;
const EPOCH = 1;
const TRAIN = 2;

const oBatchStorage = {};
const oEpochStorage = {};
const oTrainStorage = {};

function saveMetrics(sModelId, iContainerId, oMetrics) {
    let oContainer = null;
    switch (iContainerId) {
        case BATCH:
            oContainer = oBatchStorage;
            break;
        case EPOCH:
            oContainer = oEpochStorage;
            break;
        case TRAIN:
            oContainer = oTrainStorage;
            break;
        default:
            break;
    }
    if (oContainer) {
        if (!oContainer[sModelId]) {
            oContainer[sModelId] = [];
        }
        oContainer[sModelId].push(oMetrics);
        return true;
    } else {
        return false;
    }
}

function loadMetrics(sModelId, iContainerId) {
    let oContainer = null;
    let aMetrics = [];
    switch (iContainerId) {
        case BATCH:
            oContainer = oBatchStorage;
            break;
        case EPOCH:
            oContainer = oEpochStorage;
            break;
        case TRAIN:
            oContainer = oTrainStorage;
            break;
        default:
            break;
    }
    if (oContainer) {
        if (oContainer[sModelId]) {
            aMetrics = oContainer[sModelId];
        }
    }
    return aMetrics;
}

function saveBatchMetrics(sModelId, oMetrics) {
    return saveMetrics(sModelId, BATCH, oMetrics);
}

function loadBatchMetrics(sModelId) {
    return loadMetrics(sModelId, BATCH);
}

function saveEpochMetrics(sModelId, oMetrics) {
    return saveMetrics(sModelId, EPOCH, oMetrics);
}

function loadEpochMetrics(sModelId) {
    return loadMetrics(sModelId, EPOCH);
}

function saveTrainMetrics(sModelId, oMetrics) {
    return saveMetrics(sModelId, TRAIN, oMetrics);
}

function loadTrainMetrics(sModelId) {
    return loadMetrics(sModelId, TRAIN);
}

module.exports = {
    saveBatchMetrics: saveBatchMetrics,
    loadBatchMetrics: loadBatchMetrics,
    saveEpochMetrics: saveEpochMetrics,
    loadEpochMetrics: loadEpochMetrics,
    saveTrainMetrics: saveTrainMetrics,
    loadTrainMetrics: loadTrainMetrics
};