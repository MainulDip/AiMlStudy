const constants = {}

constants.DATA_DIR = "../data"
constants.RAW_DIR = constants.DATA_DIR + "/raw"
constants.DATASET_DIR = constants.DATA_DIR + "/dataset"
constants.JSON_DIR=constants.DATASET_DIR+"/json";
constants.IMG_DIR=constants.DATASET_DIR+"/img";

// samples for feeding into node backend
constants.SAMPLES=constants.DATASET_DIR+"/samples.json";
constants.FEATURES = constants.DATASET_DIR + "/features.json"
constants.JS_OBJECTS="../common/js_objects";

// samples for feeding into web (frontend)
constants.SAMPLES_JS=constants.JS_OBJECTS+"/samples.js";
// full form of samples with features
constants.FEATURES_JS = constants.JS_OBJECTS + "/features.js"

if (typeof module!=='undefined') {
    module.exports = constants
}