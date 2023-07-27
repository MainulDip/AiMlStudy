const constants = require('../common/constants.js')
const featureFunctions = require('../common/featureFunctions.js')

const fs = require('fs')


console.log("Extracting featureFunctions Start")

const samples = JSON.parse(fs.readFileSync(constants.SAMPLES)) // form data/dataset/sample.json

for (const sample of samples) {
    const paths = JSON.parse(fs.readFileSync(constants.JSON_DIR + "/" + sample.id + ".json")) // files inside data/json/session_id.json

    // adding new a prop as inside every sample object, which will change the samples object itself and can be grabbed using the sample object
    sample.point = [
        featureFunctions.getPathCount(paths),
        featureFunctions.getPointCount(paths)
    ]
}

const featureNames = ["Path Count", "Point Count"]

// creating new file on "data/dataset/feature.json" based on these values
fs.writeFileSync( constants.FEATURES, JSON.stringify({
    featureNames, 
    samples : samples.map(s => { // excluding id, session_id, name form sample object
        return {
            point: s.point,
            label: s.label
        }
    })
}))

// saving full form of samples with features for web "/common/js_objects/features.js"
fs.writeFileSync( constants.FEATURES_JS, `const features = ${JSON.stringify({ featureNames, samples })}`)

console.log("Extracting Features Done")